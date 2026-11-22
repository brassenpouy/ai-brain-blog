---
title: "MoE 与循环 Transformer：架构跟着硬件走"
date: "2026-11-22"
description: "从一条 x.com 帖子出发，解释 MoE 为什么靠大 batch 摊薄专家权重，而 looped transformer 为什么用重复计算换小内存。"
tags:
  - "AI 架构"
  - "MoE"
  - "推理"
  - "硬件"
draft: false
slug: "moe-vs-looped-transformer"
---

## 一条帖子把问题说穿了

Zhengyao Jiang 在 x.com 上把两种模型架构放在一起比较：

> MoE is a datacenter architecture. Looped transformers are a laptop architecture.

（[原帖](https://x.com/zhengyaojiang/status/2124292767274713218)）

MoE 是 Mixture of Experts，混合专家模型。这句话不是说 MoE 只能跑在机房，也不是说 looped transformer 只能跑在笔记本。它说的是两种架构的**设计取舍**：一个把成本压在内存上，一个把成本压在计算上。

![MoE 与 looped transformer 的硬件取舍](/images/moe-vs-looped-transformer.svg)

帖子引用的是 Thinking Machines 的 Inkling 2 Mini：一个 26B 参数的 looped transformer，目标是让它在笔记本上运行；同系列的 Small 是 594B（[原帖](https://x.com/thinkymachines/status/2123871643797995872)，[发布说明](https://thinkingmachines.ai/news/inkling-2-mini/)）。这里真正值得看的，不是 26B 这个数字，而是它为什么选择 looped 结构。

## MoE：把内存花在专家上

普通 dense 模型每处理一个 token，都要用整张权重矩阵。

MoE 换了一种做法：模型里有很多组专家，每个 token 只通过路由器选中少数几个专家。总参数可能非常大，但每个 token 真正参与计算的参数很少。

这带来一个很直接的结果：

- 内存要装下所有专家，所以内存占用高。
- 每个 token 只算少数专家，所以单 token 的计算量相对低。
- 路由器和专家调度增加了系统复杂度。

假设有 8 组专家，每个 token 只用 2 组。权重内存接近 8 份，计算量接近 2 份。MoE 用更大的内存换更少的计算，这就是“memory for compute”。

这种交换在数据中心很划算。GPU 显存大，可以放下所有专家；batch 大，不同 token 可以走不同专家，专家和计算单元更容易被同时利用。

## 为什么 MoE 喜欢大 batch

MoE 的优势不是无条件成立的。

一个 token 只激活少数专家，意味着权重读取很稀疏。如果一次只处理一个 token，GPU 上很多计算单元可能闲着；你仍然要为大模型准备内存，却没有足够的 token 把专家利用率拉起来。

数据中心通常有更大的 batch。很多请求一起进来，路由器把不同 token 分给不同专家，专家权重被更多 token 复用，GPU 的并行能力也更容易发挥。

本地笔记本的情况相反。大多数时候只有一个用户、一个请求，batch 基本就是 1；没有请求时甚至是 0。显存和内存容量有限，专家权重却一个都不能少。MoE 的内存优势在这里变成了负担。

## Looped Transformer：一套权重反复用

普通 Transformer 是一叠不同的层：第 1 层、第 2 层、第 3 层……每层都有自己的权重。层数越多，参数和内存越大。

Looped transformer 换了一种堆深度的方法：只保留一套（或很少几套）block，让同一套权重反复通过 N 次。

它的取舍正好相反：

- 权重只有一份，内存占用低。
- 同一套计算要重复 N 次，计算量更高。
- 增加的是“有效深度”，不是参数数量。

在笔记本上，内存容量往往是硬约束，计算可以用更多时间换；looped transformer 用更多计算换更小权重，正好符合这个约束。

当然，26B 也不是“随便一台笔记本就能跑”。实际部署通常还要 4-bit 量化，把权重压到十几 GB；是否流畅还取决于内存带宽、CPU/GPU 和上下文长度。

## 为什么 batch size 会改变答案

把两种架构放在一起看，会发现它们优化的不是同一个变量：

| | MoE | Looped transformer |
| --- | --- | --- |
| 权重内存 | 高，所有专家都要在 | 低，重复使用少量 block |
| 单 token 计算 | 只算激活的专家 | 重复通过 block |
| 适合的 batch | 大 | 小，甚至 1 |
| 典型部署 | 数据中心 | 笔记本、边缘设备 |
| 主要瓶颈 | 显存容量和带宽 | 计算和延迟 |

MoE 把内存换成计算，适合“内存充足、batch 大、追求吞吐”的场景。

Looped transformer 把计算换成内存，适合“内存紧张、batch 小、能接受更多计算”的场景。

这也解释了帖子里的“datacenter architecture”和“laptop architecture”：它们描述的是设计中心，不是物理限制。

## 这不是绝对规则

不要把这套分类当成定律。

- 小规模的 MoE 也可以跑在个人工作站上。
- 大型 looped 模型也可以部署在数据中心。
- 量化、蒸馏和推理框架会改变内存和计算的实际比例。
- batch 大小、延迟目标和功耗，可能比参数量更能决定最终选择。

真正要问的是：模型在目标硬件上，内存够不够、带宽够不够、batch 能不能放大、延迟能不能接受。

## 对部署的启发

一条模型架构的选择，背后其实是一组部署约束：

- 显存/内存容量
- 显存带宽
- 典型 batch 大小
- 首 token 延迟和每 token 延迟
- 功耗和散热

数据中心有更大的内存和 batch，能用 MoE 换吞吐；笔记本只有一个用户，内存和功耗都紧，looped transformer 更自然。

所以，模型架构不是脱离硬件独立进化的。它要回答的，始终是同一个问题：**在目标硬件上，用哪种方式组织计算最划算。**
