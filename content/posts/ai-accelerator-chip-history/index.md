---
title: "AI 算力芯片谱系：从通用 GPU 到专用 ASIC"
date: "2026-11-22"
description: "按公司和代际整理 NVIDIA、Google、AWS、AMD、Intel、华为、寒武纪等 AI 加速器的主要产品、公开规格、软件栈和适用场景。"
tags:
  - "AI 芯片"
  - "GPU"
  - "TPU"
  - "数据中心"
  - "国产芯片"
draft: false
slug: "ai-accelerator-chip-history"
---

## 先限定范围

这不是一张“所有 AI 芯片”的清单。本文只整理数据中心里的训练/推理加速器，按公司和代际选代表型号；消费级 GPU、手机 NPU、FPGA、自动驾驶芯片和同一芯片的各种 OEM SKU 不展开。

同一颗芯片，放在 PCIe 卡、SXM 模块、整机或云实例里，频率、功耗和显存配置都可能不同。下表的算力大多是厂商公开的芯片级峰值，稀疏/密集、FP16/FP8/FP4、芯片/系统的口径并不统一，不能直接横比。未找到可靠公开值的写 `n/a`。

## 规格怎么读

在比较芯片前，先弄清这些词。

| 词 | 含义 | 容易误解的地方 |
| --- | --- | --- |
| TFLOPS | 每秒万亿次浮点运算 | 可能是 FP64、FP32、FP16 或 Tensor Core 峰值，不能混着比 |
| TOPS | 每秒万亿次整数运算 | 常见于 INT8 推理，和浮点算力不是同一把尺子 |
| PFLOPS | 每秒千万亿次浮点运算 | 1 PFLOPS = 1,000 TFLOPS；机架级数字还要除以芯片数 |
| FP16 / BF16 | 16 位浮点 | BF16 动态范围更大，FP16 尾数更细；训练常用 BF16 |
| FP8 / FP4 | 8 位 / 4 位浮点 | 带宽和算力更高，但需要缩放和校准，精度会受影响 |
| INT8 | 8 位整数 | 常用于推理；峰值 TOPS 不能直接换算成浮点模型速度 |
| HBM / HBM3e | 高带宽显存 | 容量决定能装多大模型，带宽决定数据喂得多快 |
| 显存带宽 | 每秒能搬多少字节 | LLM decode 经常先卡在这里，不是先卡在峰值算力 |
| TDP | 热设计功耗 | 不代表实际功耗，也不代表性能；散热和电力要一起看 |
| SM | NVIDIA 的 Streaming Multiprocessor | GPU 内部的计算块，不是型号；SM 数不等于性能 |
| SXM / OAM | 加速器模块封装 | NVIDIA 用 SXM，AMD/Intel/国产多用 OAM；同一颗芯片可以有 PCIe 版本 |
| 芯片 / 板卡 / 系统 | 三种统计范围 | 同一颗芯片做成不同整机，峰值和功耗会不同 |
| 稀疏 / 密集 | 是否利用 2:4 等稀疏结构 | 稀疏峰值通常是密集峰值的 2 倍，不能直接横向比较 |
| MFU | 模型 FLOPs 利用率 | 实际训练能达到峰值算力的比例，比峰值更接近真实体验 |

训练看三件事：显存容量、显存带宽和互联。推理还要加一项延迟：首 token 多久出来、之后每个 token 多快。峰值算力只是上限，数据供不上时会先停。

## 同一颗芯片，不同形态

同一颗 GPU 不会只有一种卖法。它可以做成 PCIe 插卡、SXM/OAM 模块、整机，或者只在云上以实例出租。型号后面经常带 `SXM`、`PCIe`、`OAM`，说的就是形态，不是另一颗芯片。

| 形态 | 常见厂商 | 特点 |
| --- | --- | --- |
| PCIe 卡 | NVIDIA、AMD、Intel、国产 GPU | 标准服务器插槽，部署灵活，功耗和互联通常受限 |
| SXM | NVIDIA | 高功耗模块，配合 NVLink 和 HGX/NVL 系统 |
| OAM | AMD、Intel、国产加速器 | OCP 加速器模块，通常插在 UBB 基板上 |
| 整机 / 机架 | GB200 NVL72、TPU Pod、Trainium 集群 | 厂商按整机或集群给算力，不能直接除以单卡数 |
| 云实例 | Google Cloud、AWS | 不零售芯片，按实例规格提供算力 |

以 NVIDIA 为例，SM 数和常见形态如下。SM 是芯片内部的计算块，SXM/PCIe 是板卡形态；两者不是一回事。

| 产品 | 常见形态 | SM 数（常见配置） |
| --- | --- | --- |
| P100 | SXM2 / PCIe | 56 |
| V100 | SXM2 / PCIe | 80 |
| A100 | SXM4 / PCIe | 108 |
| H100 | SXM5 / PCIe | 132 / 114 |
| H200 | SXM | 132 |
| B200 | SXM | n/a |
| GB200 | NVL72 整机 | n/a |

SM 数多，代表芯片内部可调度的计算块多；实际性能还取决于频率、显存带宽、Tensor Core、互联和软件。不同厂商的“计算块”名字也不同：AMD 叫 CU，Google TPU 用矩阵乘法阵列，不能只数块数。

## 五个阶段

- 2015～2017：TPU v1 使用专用矩阵乘法阵列；P100、V100 进入深度学习训练。
- 2018～2020：AWS、华为、Intel 推出自研加速器；A100 成为通用训练主力。
- 2021～2022：TPU v4、MI250X、H100、Gaudi2、BR100 等发布，训练规模继续扩大。
- 2023～2024：H200、MI300X、Trainium2、Trillium、Gaudi3 提升显存容量和带宽。
- 2025～2026：Blackwell、Ironwood、TPU 8t/8i、Trainium3、MI350、Ascend 910C 进入系统级竞争；scale-up 网络和软件栈与芯片同样重要。

## NVIDIA：通用 GPU 的主线

软件栈：CUDA、TensorRT、NCCL、cuBLAS/cuDNN，是目前最完整的加速计算生态。

| 年份 | 产品 | 显存 / 带宽 | 峰值算力（厂商口径） | 定位 |
| --- | --- | --- | --- | --- |
| 2016 | P100 | 16GB HBM2 / 732GB/s | FP16 21.2 TFLOPS | 深度学习与 HPC |
| 2017 | V100 | 16/32GB HBM2 / 900GB/s | FP16 Tensor 125 TFLOPS | 训练 |
| 2020 | A100 | 40/80GB HBM2e / 2.0TB/s | BF16 312 TFLOPS；稀疏 624 | 训练/推理 |
| 2022 | H100 | 80GB HBM3 / 3.35TB/s | FP16 989 TFLOPS；FP8 1.98 PFLOPS | LLM 训练/推理 |
| 2024 | H200 | 141GB HBM3e / 4.8TB/s | 与 H100 同档算力 | 大模型推理 |
| 2024 | H20 | 96GB HBM3 / 4.0TB/s | FP16 148 TFLOPS；FP8 296 TFLOPS | 中国市场推理 |
| 2024 | B200 | 192GB HBM3e / 8TB/s | FP4 约 10 PFLOPS | 万亿参数训练/推理 |
| 2024 | GB200 | 2×B200，384GB HBM3e / 16TB/s | FP4 约 20 PFLOPS | 机架级系统 |
| 2025 | B300/GB300 | 288GB HBM3e / 8TB/s（单 GPU） | Blackwell Ultra，FP4 较 B200 提升 1.5x | 推理/训练 |

NVIDIA 的优势来自芯片、NVLink、InfiniBand、CUDA 和整机的组合。到了 Blackwell，单卡变成双 die 封装，竞争单位从卡转向机架。

## Google TPU：为矩阵乘法做 ASIC

软件栈：XLA、JAX、Pathways；主要通过 Google Cloud 使用。

| 年份 | 产品 | 显存 / 带宽 | 峰值算力（厂商口径） | 定位 |
| --- | --- | --- | --- | --- |
| 2015 | TPU v1 | 8GB DDR3 / 34GB/s | 92 TOPS INT8 | 推理 |
| 2017 | TPU v2 | 16GB HBM / 600GB/s | 45 TFLOPS BF16 | 训练 |
| 2018 | TPU v3 | 32GB HBM / 900GB/s | 约 123 TFLOPS BF16 | 训练 |
| 2021 | TPU v4 | 32GB HBM / 1.2TB/s | 275 TFLOPS BF16 | 训练 |
| 2023 | TPU v5e | 16GB HBM / 819GB/s | 197 TFLOPS BF16 | 成本型推理 |
| 2023 | TPU v5p | 95GB HBM / 2.76TB/s | 459 TFLOPS BF16 | 大模型训练 |
| 2024 | Trillium（v6e） | 32GB HBM / 1.6TB/s | 926 TFLOPS BF16 | 训练/推理 |
| 2025 | Ironwood（v7） | 192GB HBM3e / 7.37TB/s | 4.6 PFLOPS FP8 | 推理与思考型负载 |
| 2026 | TPU 8t / 8i | 216GB / 288GB HBM3e | 12.6 / 10.1 PFLOPS FP4 | 训练 / 推理分线 |

TPU 是“芯片 + Pod + 编译栈”一起设计的典型。Ironwood 之后，Google 又把训练和推理拆成 8t、8i 两条线，和 NVIDIA 的通用 GPU 路线形成对比。

## AWS：Inferentia 与 Trainium

软件栈：Neuron SDK；芯片只在 AWS 云上通过实例提供，不单独零售。

| 年份 | 产品 | 显存 / 带宽 | 峰值算力（厂商口径） | 定位 |
| --- | --- | --- | --- | --- |
| 2018 | Inferentia | 16GB DDR4 / n/a | 128 TOPS INT8 | 推理 |
| 2020 | Trainium | 32GB HBM / 820GB/s | 190 TFLOPS FP16 | 训练 |
| 2022 | Inferentia2 | 32GB HBM / 820GB/s | 380 TOPS INT8 | 推理 |
| 2024 | Trainium2 | 96GB HBM3 / 2.9TB/s | 1.3 PFLOPS FP8 | 大模型训练 |
| 2025 | Trainium3 | 3nm，公开细节有限 | 约为 Trainium2 的 2 倍 | 训练 |

AWS 用 Neuron 把训练和推理成本压到自家云实例里，没有跟进通用 GPU 生态。代价是软件迁移成本。

## AMD 与 Intel：两条追赶路线

AMD 走通用 GPU + ROCm；Intel 走 Gaudi 加速器 + oneAPI。

| 年份 | 产品 | 显存 / 带宽 | 峰值算力（厂商口径） | 定位 |
| --- | --- | --- | --- | --- |
| 2020 | AMD MI100 | 32GB HBM2 / 1.23TB/s | FP16 184.6 TFLOPS | HPC/AI |
| 2021 | AMD MI250X | 128GB HBM2e / 3.2TB/s | FP16 383 TFLOPS | HPC/训练 |
| 2023 | AMD MI300X | 192GB HBM3 / 5.3TB/s | FP16 1.3 PFLOPS；FP8 2.6 PFLOPS | LLM 训练/推理 |
| 2024 | AMD MI325X | 256GB HBM3e / 6TB/s | FP16 约 1.3 PFLOPS | 大模型推理 |
| 2025 | AMD MI350X/355X | 288GB HBM3e / 8TB/s | FP4 最高 9.2 PFLOPS（355X） | 训练/推理 |
| 2019 | Intel Gaudi | 32GB HBM2 / n/a | n/a | 训练 |
| 2022 | Intel Gaudi2 | 96GB HBM2e / 2.45TB/s | BF16 432 TFLOPS | 训练/推理 |
| 2024 | Intel Gaudi3 | 128GB HBM2e / 3.7TB/s | BF16 约 1.8 PFLOPS | 训练/推理 |

AMD 的卖点是大显存和相对开放的 ROCm；Intel 的 Gaudi 则更强调性价比和以太网 scale-out。两者都还没有追上 CUDA 的生态厚度。

## 中国公司：Ascend、MLU、BR100 和更多路线

中国厂商的公开规格普遍不如 NVIDIA/AMD 完整，很多数字来自发布会、招股书或第三方拆解。下表把能确认的代际和大致能力放在一起，未公开或不稳定的写 `n/a`。

| 年份 | 产品 | 显存 / 带宽 | 峰值算力（公开口径） | 定位 |
| --- | --- | --- | --- | --- |
| 2018 | 华为 Ascend 310 | n/a | 16 TOPS INT8 | 边缘推理 |
| 2019 | 华为 Ascend 910 | 32GB HBM / n/a | 256 TFLOPS FP16 | 训练 |
| 2023 | 华为 Ascend 910B | 64GB HBM2e / n/a | 约 376 TFLOPS FP16 | 训练/推理 |
| 2025 | 华为 Ascend 910C | 双 die，公开细节有限 | n/a | 训练/推理 |
| 2017 | 寒武纪 MLU100 | 16GB / n/a | 16 TFLOPS FP16 | 推理/训练 |
| 2018 | 寒武纪 MLU270 | 16GB / n/a | 128 TOPS INT8 | 推理 |
| 2021 | 寒武纪 MLU370 | 48GB / n/a | 256 TOPS INT8 | 推理 |
| 2022 | 寒武纪 MLU590 | 48GB HBM2 / n/a | 约 512 TOPS INT8 | 训练/推理 |
| 2022 | 壁仞 BR100 | 64GB HBM2e / n/a | 约 1.9 PFLOPS FP16 | 训练 |
| 2022 | 壁仞 BR104 | 32GB HBM2e / n/a | n/a | 推理 |
| 2021 | 昆仑芯 P800 | 32GB / n/a | 256 TOPS INT8 | 推理/训练 |
| 2019 | 阿里平头哥 Hanguang 800 | n/a | 78.6 TFLOPS FP16 | 推理 |
| 2018 | 百度昆仑 1 | n/a | 260 TOPS INT8 | 推理 |
| 2021 | 百度昆仑 2 | n/a | 256 TOPS INT8 | 推理 |
| 2022 | 燧原 S60 | 32GB / n/a | n/a | 训练/推理 |
| 2023 | 摩尔线程 MTT S4000 | 48GB GDDR6 / n/a | n/a | 训练/推理 |
| 2026 | 摩尔线程 MTT S5000 | 80GB / 1.6TB/s | FP8 1,000 TFLOPS | 训练/推理 |
| 2024 | 沐曦 C500 | 64GB HBM2e / n/a | n/a | 训练/推理 |

软件栈同样分散：华为 CANN/MindSpore，寒武纪 BANG，壁仞 BIRENSUPA，昆仑芯接 PaddlePaddle，阿里和百度各自有云侧工具链。对客户来说，迁移成本往往比峰值算力更难忽略。

## 其他加速器：不只 GPU 和 TPU

| 年份 | 产品 | 形态 / 关键规格 | 能力 |
| --- | --- | --- | --- |
| 2023 | Microsoft Maia 100 | 5nm，64GB HBM2e | Azure 内部 AI 负载 |
| 2023 | Meta MTIA v1 | 内部推理 ASIC | 推荐/排序推理 |
| 2024 | Meta MTIA v2 | 内部推理 ASIC | 推荐/排序推理 |
| 2021 | Tesla Dojo D1 | 7nm，训练专用 | 特斯拉内部训练 |
| 2021 | GroqChip | 230MB SRAM，确定性执行 | 低延迟推理 |
| 2019 | Cerebras WSE-1 | 晶圆级，400,000 核 | 训练/推理 |
| 2021 | Cerebras WSE-2 | 7nm，850,000 核 | 训练/推理 |
| 2024 | Cerebras WSE-3 | 5nm，900,000 核，44GB SRAM | 训练/推理 |
| 2021 | SambaNova SN10 | RDU + 三层存储 | 训练/推理 |
| 2023 | SambaNova SN40L | 三层存储，面向大模型 | 训练/推理 |
| 2018 | Graphcore GC2 | IPU，MIMD 架构 | 训练/推理 |
| 2020 | Graphcore GC200 | 7nm，1,472 IPU 核，900MB SRAM | 训练/推理 |
| 2021 | Tenstorrent Grayskull | Tensix + RISC-V | 训练/推理 |
| 2022 | Tenstorrent Wormhole | 以太网 scale-out | 训练/推理 |
| 2024 | Tenstorrent Blackhole | 新一代 Tensix | 训练/推理 |

这些公司用架构差异换空间：Groq 用 SRAM 和确定性执行换低延迟，Cerebras 用晶圆级集成换带宽，Graphcore 用 IPU 换不同的并行模型，Tenstorrent 用 RISC-V 和开放软件换可定制性。

## 横向看：算力之外的决定因素

- **显存容量和带宽**：LLM 推理先看能不能装下权重和 KV cache，再看峰值算力。
- **互联和集群**：NVLink、TPU Pod、Trainium 集群、CloudMatrix 决定多卡效率。
- **软件栈**：CUDA 的生态优势仍然最大；XLA、Neuron、ROCm、CANN 都在补课。
- **精度路线**：FP16/BF16 → FP8 → FP4，硬件峰值提高，但量化和数值稳定性成为新问题。
- **供货与合规**：出口管制、制程限制和云服务可用性，会直接影响客户能不能拿到卡。
- **系统交付**：从卖芯片转向卖机架、Pod、整机和云实例，竞争单位已经变了。

## 数据说明

表中的时间、显存和峰值算力主要来自厂商规格页、产品发布资料和公开论文。可参考 [NVIDIA 数据中心 GPU](https://www.nvidia.com/en-us/data-center/)、[Google Cloud TPU](https://cloud.google.com/tpu/docs)、[AWS Neuron](https://aws.amazon.com/machine-learning/neuron/)、[AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html)、[Intel AI 加速器](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators.html)、[昇腾](https://www.hiascend.com/)、[寒武纪](https://www.cambricon.com/) 和 [摩尔线程](https://www.mthreads.com/)。中国厂商和初创公司的部分型号公开资料有限，`n/a` 表示没有找到可靠公开值。

厂商峰值不是实测性能：稀疏与密集、芯片与板卡、训练与推理的口径都不同。选型时，要把模型、batch、序列长度、精度、互联和软件栈放进同一个测试里比较。
