---
title: "70B 模型推理要多少显存：140GB 只是权重"
date: "2026-11-22"
description: "以 Llama 3 70B 为例，把 weights、KV cache、激活和碎片分开算，判断 2 张还是 4 张 80GB GPU。"
tags:
  - "GPU"
  - "推理"
  - "显存"
  - "KV cache"
draft: false
slug: "70b-inference-memory-budget"
---

70B 模型推理要多少显存，很多人第一反应是 140GB。这个数只算了权重，KV cache、prefill 的临时 buffer 和框架开销都没进去。下面按 Llama 3 70B、BF16 把账算完，再看 2 张和 4 张 80GB 卡怎么选。

## 1. 结论

70B BF16 模型的权重约 140 GB。32 个请求、每个 4K token 时，KV cache 约 40 GiB，折合约 43 GB。加上 prefill 激活、CUDA context、NCCL buffer、CUDA graphs 和碎片，总显存约 190～200 GB。

两张 80GB 卡一共 160 GB，不够，常规选择是 4 张 80GB GPU 做 tensor parallelism。换成 FP8 权重后，70 GB 权重加 43 GB KV cache 和开销约 120 GB，两张 80GB 卡可以放下。

这张图是 32 个 4K 请求的显存账：

![70B BF16 模型在 32 个 4K 请求下的显存构成](/images/inference-memory-stack.svg)

## 2. 总账

推理显存可以拆成五项：

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

把显存想成一张桌子：weights 是模型本体，KV cache 是每个请求保留的上下文记录，activations 是临时草稿，overhead 是引擎自己占的位置。

weights 基本固定。KV cache 会随着并发数和上下文长度涨。激活和框架开销占比不大，但很容易把“刚好够”变成“差一点”。

## 3. 权重

Llama 3 70B 有 70B 参数。BF16 每个参数 2 字节：

```text
70B × 2 B = 140 GB
```

不同精度的权重占用：

| 精度 | 权重大小 | 备注 |
| --- | --- | --- |
| BF16 | 140 GB | 训练和推理常用 |
| FP8 | 70 GB | 需要硬件和 kernel 支持 |
| INT4 | 约 35 GB | 还要额外存储 scales，质量和计算开销要单独评估 |

权重精度和 KV cache 精度是两件事。权重用 FP8，不代表 KV cache 也自动变成 FP8。

## 4. KV cache

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

GQA 在这里很关键。Llama 3 70B 只有 8 个 KV heads。如果换成 64 个 KV heads 的 MHA，KV cache 会变成 8 倍，32 个 4K 请求需要约 320 GiB。仅这一项就足以改变 GPU 数量。

## 5. 激活和框架开销

decode 阶段每个请求每步只处理一个 token，激活不大。prefill 阶段处理整段 prompt，临时 buffer 会明显上升，长上下文时更明显。

vLLM 这类引擎会在启动时跑一次测试 forward，测出峰值后预留显存。实际规划时不要只拿公式算，要留出 activation workspace。

框架开销按每张 GPU 计算，常见来源：

- CUDA context。
- NCCL 通信 buffer。
- CUDA graphs。
- 框架自己的 cache 和 allocator。

vLLM 默认把 `gpu_memory_utilization` 设成 0.9，留下约 10% 余量。这部分余量用来吸收峰值和碎片，实际部署要留。

## 6. 碎片

如果按最大长度给每个请求预留一整块 KV，绝大多数位置会空着。并发一高，浪费很快放大。

PagedAttention 把 KV cache 切成小块，按需分配。浪费主要落在每个请求的最后一个 block，不再是一整段最大长度。它解决的是 KV cache 内部碎片，不能消除 allocator 碎片和显存预留。

因此实际可用显存总比标称值少。规划时把 80GB 当成 75GB 左右来算，更接近工程情况。

## 7. 32 个 4K 请求要几张卡

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

3 张 80GB 从总量看也可能放得下，但 Llama 3 70B 的 8 个 KV heads 不能被 3 整除，4-way TP 更自然。实际选型还要看 NVLink 拓扑、框架支持和并行策略。

## 8. FP8 和 INT4 的边界

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

## 10. 怎么定卡数

先把账分清楚：weights 基本固定，KV cache 随总 token 数增长，显存预算要按最坏情况留。

拿上面的例子：

- 70B BF16 权重约 140 GB。
- Llama 3 70B 的 KV cache 约 320 KiB/token。
- 32 个 4K 请求约 40 GiB KV cache。
- 加激活、框架开销和碎片，总额约 190～200 GB。
- BF16 需要 4×80GB，FP8 可以压到 2×80GB。

这些只是容量估算，真实部署要用目标引擎和实际请求分布做 profile。长上下文 prefill、KV 量化和 tensor parallel 拓扑都会改变结果。
