---
title: "70B 模型推理的显存构成"
date: "2026-11-22"
description: "140GB 只是权重，KV cache 和运行时开销才是变量。按 Llama 3 70B 算清 32 个 4K 请求需要多少显存，以及 2 张还是 4 张 80GB 卡。"
tags:
  - "GPU"
  - "推理"
  - "显存"
  - "KV cache"
draft: false
slug: "70b-inference-memory-budget"
---

## 1. 从一个具体问题开始

假设有一台 2×80GB 的 GPU 机器，要服务 32 个用户，每个用户 4K 上下文。模型是 Llama 3 70B，BF16。

先算权重：70B 参数，每个 2 字节，140GB。两张 80GB 卡一共 160GB，看起来能装下。

实际不够。推理时显存不只放权重，还要放 KV cache、prefill 临时 buffer、框架开销和碎片。把这些算进去，总显存会到 190～200GB。

## 2. 显存分四块

| 项 | 32×4K 时的量级 | 由什么决定 |
| --- | --- | --- |
| weights | 140 GB | 参数量、权重精度 |
| KV cache | 约 43 GB | 并发数、上下文长度、层数、KV heads |
| activations | 约 5～10 GB | prefill 长度、kernel workspace |
| overhead + 碎片 | 约 5～10 GB | CUDA context、NCCL、CUDA graphs、allocator |

把显存想成一张桌子：weights 是模型本体，KV cache 是每个请求保留的上下文记录，activations 是临时草稿，overhead 是引擎自己占的位置。

![70B BF16 模型在 32 个 4K 请求下的显存构成](/images/inference-memory-stack.svg)

weights 基本固定。KV cache 会随着并发数和上下文长度涨，是后面最需要算的一块。

## 3. 权重是固定成本

Llama 3 70B 有 70B 参数，BF16 每个参数 2 字节：

```text
70B × 2 B = 140 GB
```

换成 FP8 是 70GB，INT4 约 35GB，还要额外存 scales。权重精度和 KV cache 精度是两件事，权重用 FP8，不代表 KV cache 也自动变成 FP8。

## 4. KV cache 是什么

Attention 里，每个 token 会生成 Q、K、V。生成新 token 时，当前的 Q 要和前面所有 token 的 K 做匹配，再取对应的 V。

如果每一步都重新算历史 K 和 V，开销很大。推理引擎会把它们存下来，这就是 KV cache。每层都要存，每个 KV head 都要存，所以它会随着上下文长度和并发数增长。

它像对话的工作记忆：聊得越久，记忆越大。

## 5. KV cache 有多大

Llama 3 70B 有 80 层、8 个 KV heads、head_dim 128，KV 用 BF16：

```text
2 × 80 × 8 × 128 × 2 = 327,680 B
= 320 KiB / token
```

前面的 2 表示 K 和 V 各存一份。

- 一个 4,096 token 请求：1.25 GiB。
- 32 个 4K 请求：40 GiB。
- 一个 128K token 请求：40 GiB。

![32 个 4K 请求和 1 个 128K 请求的 KV cache 预算](/images/kv-cache-budget.svg)

并发数和上下文长度看着是两个参数，实际都在花同一笔 KV 预算。32 个 4K 用户和一个 128K 用户，总 token 数一样，KV cache 也一样。

Llama 3 70B 用 GQA，8 个 KV heads 被多个 query heads 共享。如果换成每个 query head 都有独立 KV 的 MHA，KV cache 会变成 8 倍。

## 6. 2 张卡为什么不够

除了 weights 和 KV cache，还有两块：

- prefill 处理整段 prompt，临时 buffer 比 decode 大。
- vLLM 这类引擎会预留 CUDA context、通信 buffer、CUDA graphs，并默认留 10% 显存余量。

KV cache 也不是按最大长度一次性整块分配。PagedAttention 把它切成小块，按需分配，浪费主要落在每个请求的最后一个 block。实际规划时，把 80GB 当成 75GB 左右更稳。

总账约 190～200GB：

| 方案 | 单卡可用 | 聚合容量 | 判断 |
| --- | --- | --- | --- |
| 2×80GB BF16 | 约 75 GB | 约 150 GB | 不够 |
| 4×80GB BF16 | 约 75 GB | 约 300 GB | 够，留出 KV 增长空间 |
| 2×80GB FP8 | 约 75 GB | 约 150 GB | 权重 70 GB，加 KV 后够 |

从每卡看更直接：2 卡 BF16 时，权重每卡 70GB，KV cache 每卡 20GiB，再加开销，已经超过 80GB。4 卡时，权重每卡 35GB，KV 每卡 10GiB，能留出余量。

## 7. FP8 和多卡边界

FP8 权重把固定成本降到 70GB。70GB 权重加 43GB KV cache，再加约 10GB 开销，约 123GB，2×80GB 能放下。KV cache 也可以量化到 FP8，进一步减半，但要验证模型质量和引擎支持。

多卡不只是买显存。weights 和 KV cache 被切开后，聚合 HBM 带宽会增加，decode 可能更快；代价是 tensor parallelism 的 all-reduce 通信。2 卡和 4 卡的 decode 速度不会线性翻倍。

## 8. 实际估算顺序

1. 算 weights：参数量 × 每参数字节数。
2. 定最坏情况：并发数 × 最大上下文长度。
3. 算 KV cache：总 token 数 × 每 token 的 KV 大小。
4. 加激活、框架开销和碎片，留 10% 左右余量。
5. 按单卡可用容量拆卡，再用目标引擎和真实请求分布做 profile。

这些只是容量估算，真实部署还要看长上下文 prefill、KV 量化和 tensor parallel 拓扑。
