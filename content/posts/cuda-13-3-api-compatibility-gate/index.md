---
title: "CUDA 13.3 API 兼容验收门槛"
date: "2026-11-26T22:13:00+08:00"
lastmod: "2026-11-26T22:15:58+08:00"
description: "以 CUDA 13.3 Runtime/Driver API 为边界，定义源码级兼容验收硬门槛；不覆盖 ABI/二进制兼容。"
tags:
  - "CUDA"
  - "兼容性"
  - "测试"
  - "GPU"
draft: false
slug: "cuda-13-3-api-compatibility-gate"
---

## 验收边界

CUDA 13.3 API 兼容验收只回答三件事：版本和 API 契约是否对齐，声明支持的符号能否使用，行为和错误语义是否一致。

cuBLAS、cuDNN、NCCL、PyTorch 和模型精度不在这一层验收。API 测试通过，不能推出完整软件栈兼容。

本文的兼容指源码/API 兼容：应用使用 13.3 头文件和 API，重新编译后运行。已有 CUDA 二进制不重编译直接运行、soname、符号版本、结构体布局和 driver ABI 与 NVIDIA 一致，都不在范围内。

先声明范围。覆盖 13.3 全部公开 Runtime/Driver API，才可以叫“全量兼容”；只覆盖子集，只能标“部分兼容”。

## 参考环境

以 CUDA 13.3 GA 为基准：驱动 610.43.02，cudart 13.3.29。参考环境用于差分测试，不要求厂商驱动号与 NVIDIA 相同。

13.3 的 API 变化以 [CUDA 13.3 Release Notes](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html)、[Runtime API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/index.html) 和 [Driver API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/index.html) 为准。

## 四道硬门槛

### 版本与 API 契约

- `cudaRuntimeGetVersion()` 和 `cuDriverGetVersion()` 返回 13.3 对应版本号 13030。
- 头文件 `CUDA_VERSION` 为 13030；使用 13.3 头文件可以编译，API 参数、枚举和回调签名与 13.3 一致。
- 声明支持的符号在编译和链接阶段可用；`cuGetProcAddress` 的版本化入口可用。

### API 覆盖

- 声明范围内的符号覆盖率 100%。
- 不支持的能力返回文档定义的错误，不能返回成功，也不能静默降级。
- 13.3 新增 API 逐项给出实现或不支持结论。

只支持子集时，不能对外写“完全兼容 CUDA 13.3”。

### 行为语义

P0/P1 用例通过率必须为 100%。

- P0：初始化、context、module、memory、stream、event、kernel launch、错误模型、版本管理。
- P1：graph、VMM、async allocator、P2P/IPC、MPS/MIG、green context、PTX JIT。
- P2：interop、texture/surface、profiler、checkpoint 等，可以有不支持项，但必须记录。

硬性结果：

- 静默错误、崩溃、挂起、内存越界、context 损坏为 0。
- 错误码类别、异步错误暴露时机、资源生命周期与 13.3 文档一致。
- PTX 和 cubin 分开测试。13.3 文档明确 PTX 在旧驱动上受限，不能混在一起判断。
- 版本混用按 13.3 的 version mixing rules 测试，不假设任意组合都可用。

### 增量与回归

13.3 新增或变化的能力必须单独设用例：

- `cuStreamBeginRecaptureToGraph`：拓扑匹配、非法重捕获、失败后的 graph 状态。
- Green Contexts：默认 stream 可选，SM/workqueue 资源划分。
- MPS 静态分区错误隔离：分区创建、非法分区 ID、跨分区故障隔离。
- DMA-BUF mmap：设备属性、export、`mmap` 和同步协议；不支持平台明确报错。
- CUDA 13.3 新 API 的错误路径和资源清理。

回归要求：

- 13.2 已通过的 API 用例在 13.3 上不能退化。
- 13.3 deprecated 符号仍能编译、链接、运行。
- NVIDIA 13.3 release notes 中的 known issues 要映射到测试豁免，不能把已知 bug 当作兼容通过。

## 判定

满足以下条件，才能判定 CUDA 13.3 API 兼容验收通过：

1. 声明范围内 API 覆盖率 100%。
2. P0/P1 用例通过率 100%。
3. 静默错误、崩溃、挂起、内存破坏为 0。
4. 13.3 新增 API 全部有实现或不支持结论。
5. 13.2 回归无退化。
6. 兼容矩阵、测试日志、未支持清单可审计。

不满足时，只能标注“CUDA 13.3 部分兼容”。性能、框架、模型和应用需要在上层单独验收。
