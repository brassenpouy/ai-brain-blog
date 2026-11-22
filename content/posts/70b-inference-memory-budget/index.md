---
title: "70B 模型推理的显存构成"
date: "2026-11-22"
description: "从权重和 KV cache 讲起，算清 32 个 4K 请求需要多少显存，以及 2 张还是 4 张 80GB 卡够不够。"
tags:
  - "GPU"
  - "推理"
  - "显存"
  - "KV cache"
draft: false
slug: "70b-inference-memory-budget"
---

## 先看一个具体场景

假设有一台 2×80GB 的 GPU 机器，要服务 32 个用户，每个用户 4K 上下文。模型是 Llama 3 70B，BF16。

70B 指 700 亿参数。模型规模通常指参数量，但实际占多少显存，还要看每个参数用几个字节存。BF16 每个参数 2 字节，所以权重是 140GB。两张 80GB 卡一共 160GB，看起来能装下。

这里有两个独立变量。70B 决定模型权重；32 个 4K 请求是工作负载（workload），也就是并发请求数和上下文长度，影响 KV cache 的大小。70B 不会自带 32 个用户，4K 也不是模型属性，只是这次估算的负载假设。

但推理时，显存不只放权重。模型生成每个 token 时还要回看前文，这部分历史信息也要缓存；此外还有临时计算 buffer 和框架开销。把这些算进去，总显存会到 190～200GB。

## 权重是固定的一块

权重是最容易算的部分：

```text
70B × 2 B = 140 GB
```

精度只改变每个参数占多少字节，不改变参数量。70B 仍然是 70B：BF16 每参数 2 字节，权重约 140GB；FP8 每参数 1 字节，约 70GB；FP4 或 INT4 约 0.5 字节，约 35GB，后两者还要额外存 scales。

权重精度和 KV cache 精度是两件事，权重用 FP8，不代表 KV cache 也自动变成 FP8。

常见的 70B 级模型：

- Llama 3.3 70B
- Llama 3.1 70B
- Qwen2.5 72B
- DeepSeek-R1-Distill-Llama-70B
- Hermes 4 70B

70B 在开源模型里算中大，但不是最大一档；这里只讨论 dense 模型，MoE 还要区分 total parameters 和每个 token 的 active parameters。

## KV cache 会随上下文增长

模型生成 token 时，attention 会为每个 token 生成 Q、K、V。新的 token 要用当前的 Q 和前面所有 token 的 K 匹配，再取对应的 V。

每一步都重新算历史 K 和 V，开销太大。推理引擎会把它们存下来，这就是 KV cache。每层都要存，每个 KV head 都要存。它像对话的工作记忆：上下文越长，记住的内容越多；并发请求越多，总量也越大。

Llama 3 70B 有 80 层、8 个 KV heads、head_dim 128，KV 用 BF16。每个 token 的缓存：

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

## 还有激活、开销和碎片

decode 阶段每个请求每步只处理一个 token，激活不大。prefill 阶段处理整段 prompt，临时 buffer 会明显上升，长上下文时更明显。

vLLM 这类引擎会在启动时跑一次测试 forward，测出峰值后预留显存。它还会占 CUDA context、通信 buffer 和 CUDA graphs，并默认留出约 10% 余量。

KV cache 也不是按最大长度一次性整块分配。PagedAttention 把它切成小块，按需分配，浪费主要落在每个请求的最后一个 block。实际规划时，把 80GB 当成 75GB 左右更稳。

![70B BF16 模型在 32 个 4K 请求下的显存构成](/images/inference-memory-stack.svg)

## 需要几张 80GB 卡

把前面的数加起来，32 个 4K 请求大约需要 190～200GB：

| 方案 | 单卡可用 | 聚合容量 | 结果 |
| --- | --- | --- | --- |
| 2×80GB BF16 | 约 75 GB | 约 150 GB | 不够 |
| 4×80GB BF16 | 约 75 GB | 约 300 GB | 够，留出 KV 增长空间 |
| 2×80GB FP8 | 约 75 GB | 约 150 GB | 权重 70 GB，加 KV 后够 |

从每卡看更直接。2 卡 BF16 时，权重每卡 70GB，KV cache 每卡 20GiB，再加开销，已经超过 80GB。4 卡时，权重每卡 35GB，KV 每卡 10GiB，能留出余量。

FP8 权重把固定成本降到 70GB。70GB 权重加 43GB KV cache，再加约 10GB 开销，约 123GB，2×80GB 能放下。KV cache 也可以量化到 FP8，进一步减半，但要验证模型质量和引擎支持。

多卡不只是买显存。weights 和 KV cache 被切开后，聚合 HBM 带宽会增加，decode 可能更快；代价是 tensor parallelism 的 all-reduce 通信。2 卡和 4 卡的 decode 速度不会线性翻倍。

## 实际怎么估

1. 算 weights：参数量 × 每参数字节数。
2. 定最坏情况：并发数 × 最大上下文长度。
3. 算 KV cache：总 token 数 × 每 token 的 KV 大小。
4. 加激活、框架开销和碎片，留 10% 左右余量。
5. 按单卡可用容量拆卡，再用目标引擎和真实请求分布做 profile。

这些只是容量估算，真实部署还要看长上下文 prefill、KV 量化和 tensor parallel 拓扑。
