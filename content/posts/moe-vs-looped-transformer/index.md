---
title: "MoE 与循环 Transformer：架构跟着硬件走"
date: "2026-11-22"
description: "从一条 x.com 帖子出发，先解释参数、token、batch 和显存，再看 MoE 和 looped transformer 为什么适合不同硬件。"
tags:
  - "AI 架构"
  - "MoE"
  - "推理"
  - "硬件"
draft: false
slug: "moe-vs-looped-transformer"
---

## 帖子里的两种架构

Zhengyao Jiang 在 x.com 上把两种模型架构放在一起比较：

> MoE is a datacenter architecture. Looped transformers are a laptop architecture.

（[原帖](https://x.com/zhengyaojiang/status/2124292767274713218)）

大意是：MoE 更像数据中心的架构，looped transformer 更像笔记本的架构。

这句话描述的是设计取向，不是物理限制。MoE 和 looped transformer 都把一部分成本放到了另一侧：一个更吃内存，一个更吃计算。

## 先把几个词说清

- **参数和权重**：模型里可训练的数字。`26B` 指 260 亿个参数；`594B` 指 5940 亿个。
- **token**：模型处理文本的单位，可以是一个字，也可以是一个词片段。
- **batch**：一次并行处理多少请求或 token。数据中心会凑大 batch，本地通常只有 1 个请求。
- **显存和内存**：放权重、缓存和中间结果的地方；计算单元负责做乘加运算。

MoE 是 Mixture of Experts，混合专家模型。普通 dense 模型没有专家选择，每个 token 都要用全部权重；MoE 里有很多组专家，路由器为每个 token 选少数几组。

专家不是人，是一小组可以训练的权重。路由器就是一个选择器。

![MoE 与 looped transformer 的硬件取舍](/images/moe-vs-looped-transformer.svg)

帖子引用的是 Thinking Machines 的 Inkling 2 Mini：一个 26B 参数的 looped transformer，目标是让它在笔记本上运行；同系列的 Small 是 594B（[原帖](https://x.com/thinkymachines/status/2123871643797995872)，[发布说明](https://thinkingmachines.ai/news/inkling-2-mini/)）。

## MoE：把内存花在专家上

假设有 8 组专家，每个 token 只走 2 组。

- 权重内存接近 8 份，因为所有专家都要能随时被选中。
- 单个 token 的计算量接近 2 份，因为只算被选中的专家。
- 路由和调度会增加系统复杂度。

MoE 用更大的内存换更少的计算，这就是帖子里的 “memory for compute”。

在数据中心，这个交换很划算。GPU 显存大，可以放下所有专家；请求多，不同 token 可以走不同专家，计算单元更容易忙起来。

## 为什么 MoE 喜欢大 batch

MoE 的优势依赖 batch 和硬件。

一次只处理一个 token 时，这个 token 只会选中少数专家。GPU 上很多计算单元可能闲着，但显存里仍然要装下全部专家。内存付了，计算能力却没充分利用。

batch 变大后，不同 token 可以分给不同专家，专家权重被更多 token 使用，GPU 的并行能力更容易发挥。

本地笔记本通常只有一个用户、一个请求，batch 基本就是 1；没有请求时 GPU 甚至闲着。MoE 的大权重在这里很难摊薄。

## Looped Transformer：一套权重反复用

普通 Transformer 是一叠不同的层：第 1 层、第 2 层、第 3 层……每层都有自己的权重。层数越多，参数和内存越大。

Looped transformer 换了一种堆深度的方法：只保留一套（或很少几套）block，让同一套权重反复通过 N 次。

block 可以理解为一层 Transformer。普通模型每层权重不同；looped 模型只有一套，重复使用。

- 权重只有一份，内存占用低。
- 同一套计算要重复 N 次，计算量更高。
- 计算上更深，但参数没有跟着变多。

在笔记本上，内存容量往往是硬约束，计算可以用更多时间换；looped transformer 用更多计算换更小权重，正好符合这个约束。

26B 也不是随便一台笔记本就能跑。实际部署通常还要 4-bit 量化，把每个权重用更少的 bit 存储，压到十几 GB；是否流畅还取决于内存带宽、CPU/GPU 和上下文长度。

## 为什么 batch size 会改变答案

两种架构优化的变量不同：

| | MoE | Looped transformer |
| --- | --- | --- |
| 权重内存 | 高，所有专家都要在 | 低，重复使用少量 block |
| 单 token 计算 | 只算激活的专家 | 重复通过 block |
| 适合的 batch | 大 | 小，甚至 1 |
| 典型部署 | 数据中心 | 笔记本、边缘设备 |
| 主要瓶颈 | 显存容量和带宽 | 计算和延迟 |

MoE 适合内存充足、batch 大、追求吞吐的场景。

Looped transformer 适合内存紧张、batch 小、能接受更多计算的场景。

帖子里的 “datacenter architecture” 和 “laptop architecture” 说的是设计取向，不是硬性分类。

## 这不是绝对规则

- 小规模 MoE 也可以跑在个人工作站上。
- 大型 looped 模型也可以部署在数据中心。
- 量化、蒸馏和推理框架会改变内存和计算的实际比例。
- batch 大小、延迟目标和功耗，可能比参数量更能决定最终选择。

## 部署时先看什么

选择模型架构前，先看这些约束：

- 显存和内存容量
- 显存带宽
- 典型 batch 大小
- 首 token 延迟和每 token 延迟
- 功耗和散热

数据中心有更大的内存和 batch，可以用 MoE 换吞吐；笔记本通常只有一个用户，内存和功耗都紧，looped transformer 更自然。

模型架构最终要回答的问题是：在目标硬件上，用哪种方式组织计算最划算。
