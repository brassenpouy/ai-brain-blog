---
title: "70B 模型推理，显存到底花在哪"
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

## 1. 先看一个具体问题

假设手上有一台 2×80GB 的 GPU 机器，要服务 32 个用户，每个用户 4K 上下文。模型是 Llama 3 70B，BF16。

先算权重：70B 参数，每个 2 字节，140GB。两张 80GB 卡一共 160GB，看起来能装下。

但推理时显存不只放权重。KV cache、prefill 临时 buffer、CUDA context、通信 buffer 和碎片都要占地方。把 32 个 4K 请求放进去，总显存会到 190～200GB，2 张卡不够。

## 2. 显存花在哪

推理显存可以拆成五块：

```text
total = weights + KV cache + activations/workspace
      + framework overhead + fragmentation
```

| 项 | 32×4K 时的量级 | 由什么决定 |
| --- | --- | --- |
| weights | 140 GB | 参数量、权重精度 |
| KV cache | 约 43 GB | 并发数、上下文长度、层数、KV heads |
| activations | 约 5～10 GB | prefill 长度、kernel workspace |
| overhead | 约 5～10 GB | CUDA context、NCCL、CUDA graphs |
| fragmentation | 取决于 allocator | block size、最大长度、预留策略 |

可以用一张桌子理解：weights 是模型本体，KV cache 是每个请求保留的上下文记录，activations 是临时草稿，overhead 是引擎自己占的位置。

![70B BF16 模型在 32 个 4K 请求下的显存构成](/images/inference-memory-stack.svg)

weights 基本固定。KV cache 会随着并发数和上下文长度涨。激活和框架开销占比不大，但很容易把“刚好够”变成“差一点”。

## 3. 第一块：权重

这是最容易算的一块。Llama 3 70B 有 70B 参数，BF16 每个参数 2 字节：

```text
70B × 2 B = 140 GB
```

换成其他精度：

| 精度 | 权重大小 | 备注 |
| --- | --- | --- |
| BF16 | 140 GB | 训练和推理常用 |
| FP8 | 70 GB | 需要硬件和 kernel 支持 |
| INT4 | 约 35 GB | 还要额外存储 scales，质量和计算开销要单独评估 |

权重精度和 KV cache 精度是两件事。权重用 FP8，不代表 KV cache 也自动变成 FP8。

## 4. 第二块：KV cache 是什么

Attention 里，每个 token 会生成 Q、K、V。生成新 token 时，当前的 Q 要和前面所有 token 的 K 做匹配，再取对应的 V。

如果每一步都重新算历史 K 和 V，开销会很大。推理引擎会把它们存下来，这就是 KV cache。

每层都要存，每个 KV head 都要存。所以 KV cache 会随着层数、上下文长度和并发数增长。它像对话的工作记忆：聊得越久，记忆越大。

## 5. KV cache 到底多大

Llama 3 70B 的关键参数：

```text
layers = 80
KV heads = 8
head_dim = 128
KV dtype = BF16
```

每个 token 的 KV cache：

```text
2 × layers × KV heads × head_dim × bytes
= 2 × 80 × 8 × 128 × 2
= 327,680 B
= 320 KiB
```

前面的 2 表示 K 和 V 各存一份。320 KiB/token 乘以总 token 数，就是 KV cache 总量。

- 一个 4,096 token 请求：1.25 GiB。
- 32 个 4K 请求：40 GiB。
- 一个 128K token 请求：40 GiB。

![32 个 4K 请求和 1 个 128K 请求的 KV cache 预算](/images/kv-cache-budget.svg)

并发数和上下文长度看着是两个参数，实际都在花同一笔 KV 预算。32 个 4K 用户和一个 128K 用户，总 token 数一样，KV cache 也一样。

GQA 在这里很关键。Llama 3 70B 只有 8 个 KV heads，多个 query heads 共享同一组 K/V。如果换成每个 query head 都有独立 KV 的 MHA，KV cache 会变成 8 倍，32 个 4K 请求需要约 320 GiB。仅这一项就足以改变 GPU 数量。

## 6. 第三、四块：激活、开销和碎片

decode 阶段每个请求每步只处理一个 token，激活不大。prefill 阶段处理整段 prompt，临时 buffer 会明显上升，长上下文时更明显。

vLLM 这类引擎会在启动时跑一次测试 forward，测出峰值后预留显存。实际规划时不要只拿公式算，要留出 activation workspace。

框架开销按每张 GPU 计算，常见来源：

- CUDA context。
- NCCL 通信 buffer。
- CUDA graphs。
- 框架自己的 cache 和 allocator。

vLLM 默认把 `gpu_memory_utilization` 设成 0.9，留下约 10% 余量。这部分余量用来吸收峰值和碎片，实际部署要留。

如果按最大长度给每个请求预留一整块 KV，绝大多数位置会空着。并发一高，浪费很快放大。

PagedAttention 把 KV cache 切成小块，按需分配。浪费主要落在每个请求的最后一个 block，不再是一整段最大长度。它解决的是 KV cache 内部碎片，不能消除 allocator 碎片和显存预留。

因此实际可用显存总比标称值少。规划时把 80GB 当成 75GB 左右来算，更接近工程情况。

## 7. 把数带回去：2 张卡为什么不够

聚合显存约 190～200 GB，直接看单卡容量：

| 方案 | 单卡可用 | 聚合容量 | 判断 |
| --- | --- | --- | --- |
| 2×80GB BF16 | 约 75 GB | 约 150 GB | 不够 |
| 4×80GB BF16 | 约 75 GB | 约 300 GB | 够，留出 KV 增长空间 |
| 2×80GB FP8 | 约 75 GB | 约 150 GB | 权重 70 GB，加 KV 后够 |

按 4 卡 tensor parallelism 拆账更直观：

- weights：140 GB ÷ 4 = 35 GB/GPU。
- KV cache：40 GiB ÷ 4 = 10 GiB/GPU。
- 每卡还剩约 30 GB 给激活、通信 buffer、CUDA graphs 和碎片。

2 张卡不够，原因也可以从每卡看：BF16 权重每卡 70 GB，KV cache 每卡 20 GiB，再加开销，已经超过 80GB。

3 张 80GB 从总量看也可能放得下，但 Llama 3 70B 的 8 个 KV heads 不能被 3 整除，4-way TP 更自然。实际选型还要看 NVLink 拓扑、框架支持和并行策略。

## 8. 换精度，能不能压到 2 张卡

FP8 权重把固定成本从 140 GB 降到 70 GB。2 卡 BF16 放不下的 70B，在 FP8 下可以变成 2 卡部署：

```text
70 GB weights + 43 GB KV cache + 约 10 GB 开销
≈ 123 GB
```

2×80GB 的 150 GB 左右可用容量能覆盖这个量级。KV cache 也可以量化到 FP8，进一步减半，但要验证模型质量和引擎支持。

INT4 权重约 35 GB，看起来更省，但还要算 scales、dequant 开销和精度损失。显存只是第一道门槛，吞吐和输出质量还要单独看。

## 9. 多卡不只买显存

增加 GPU 会把 weights 和 KV cache 切开，也会增加聚合 HBM 带宽。decode 是 memory-bandwidth-bound，更多卡可能让每步更快。

代价是 tensor parallelism 的 all-reduce 通信。层数多、卡间互联慢时，通信会把收益吃掉。2 卡和 4 卡的 decode 速度不会线性翻倍。

容量规划先看显存，性能规划还要看：

- 单卡 HBM 带宽和聚合带宽。
- NVLink 或 PCIe 拓扑。
- batch size 和上下文长度分布。
- prefill 与 decode 的配比。
- KV cache 的读写效率。

## 10. 实际估算按这个顺序

先把账分清楚：weights 基本固定，KV cache 随总 token 数增长，显存预算要按最坏情况留。

1. 算 weights：参数量 × 每参数字节数。
2. 定最坏情况：并发数 × 最大上下文长度。
3. 算 KV cache：总 token 数 × 每 token 的 KV 大小。
4. 加激活、框架开销和碎片，给 10% 左右余量。
5. 按单卡可用容量拆卡，再用目标引擎和真实请求分布做 profile。

拿上面的例子：70B BF16 权重约 140 GB，KV cache 约 320 KiB/token，32 个 4K 请求约 40 GiB，加开销后总额约 190～200 GB。BF16 需要 4×80GB，FP8 可以压到 2×80GB。

这些只是容量估算，真实部署要用目标引擎和实际请求分布做 profile。长上下文 prefill、KV 量化和 tensor parallel 拓扑都会改变结果。
