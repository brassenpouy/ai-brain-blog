---
title: "CUDA 13.3 API 兼容验收门槛"
date: "2026-11-26T22:13:00+08:00"
lastmod: "2026-11-26T22:19:31+08:00"
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

本文的兼容指源码/API 兼容：应用使用 13.3 头文件和 API，重新编译后运行。已有 CUDA 二进制不重编译直接运行、soname、符号版本、结构体布局和 driver ABI 与 NVIDIA 一致，都不在范围内。

cuBLAS、cuDNN、NCCL、PyTorch 和模型精度也不在这一层验收。API 测试通过，不能推出完整软件栈兼容。

先声明范围。覆盖 13.3 全部公开 Runtime/Driver API，才可以叫“全量兼容”；只覆盖子集，只能标“部分兼容”。

## 参考版本

以 CUDA 13.3 GA 为基准：驱动 610.43.02，cudart 13.3.29。参考环境用于差分测试，不要求厂商驱动号与 NVIDIA 相同。

13.3 的 API 变化以 [CUDA 13.3 Release Notes](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html)、[Runtime API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/index.html) 和 [Driver API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/index.html) 为准。

## 测试面

### API 契约

- `cudaRuntimeGetVersion()` 和 `cuDriverGetVersion()` 返回 13.3 对应版本号 13030。
- 使用 13.3 头文件可以编译，API 参数、枚举、回调和常量与 13.3 一致。
- 声明支持的符号在编译和链接阶段可用；`cuGetProcAddress` 的版本化入口可用。
- Runtime API 和 Driver API 分开测试。Runtime 封装 Driver，但默认 stream、primary context 和错误语义有自己的规则。
- deprecated 符号仍能编译、链接、运行。

### 状态与生命周期

- context：primary context 的 retain/release、多线程 current context、reset 后资源失效。
- module/function：PTX、cubin、fatbin 加载，JIT 选项，符号查找，卸载后调用。
- memory：malloc/free、pitched、UVM、VMM、async pool、OOM、非法指针、double free。
- stream/event：default stream、non-blocking stream、优先级、回调、跨 stream 依赖、event 计时。
- graph：显式构图、capture、instantiate、update、销毁，以及失败后的 graph 状态。

### 行为与并发

- 同步语义：`cudaDeviceSynchronize`、`cudaStreamSynchronize`、event sync 的覆盖范围。
- 异步错误：kernel launch 后的错误在哪个同步点暴露。
- 错误模型：非法参数、越界、错误 handle、sticky error、last error 清除规则。
- 多线程、多 stream、多卡、P2P、IPC 和 MPS/MIG。
- PDL：测试 `cudaLaunchAttributeProgrammaticStreamSerialization`、`cudaLaunchAttributeProgrammaticEvent`、`cudaTriggerProgrammaticLaunchCompletion` 和 `cudaGridDependencySynchronize`。覆盖 successor 提前启动、trigger 条件、事件触发、无 trigger 的退化路径、stream/event 顺序，以及 Driver API 的 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_*` 映射。
- PTX 和 cubin 分开测试。13.3 文档明确 PTX 在旧驱动上受限，不能混在一起判断。
- 版本混用按 13.3 的 version mixing rules 测试，不假设任意组合都可用。

### 故障注入

每个高风险用例独立进程运行，避免 sticky error 和 context 损坏污染后续测试。

- 注入 OOM、资源耗尽和分配失败。
- 在 stream 执行中销毁 event、graph、memory 或 context。
- 构造非法 handle、错误 device、错误 flag 和越界参数。
- 观察崩溃、挂起、内存越界和静默成功，而不只是看返回值。

## 优先级

| 级别 | 范围 |
| --- | --- |
| P0 | 初始化、context、module、memory、stream、event、kernel launch、错误模型、版本管理 |
| P1 | graph、PDL、VMM、async allocator、P2P/IPC、MPS/MIG、green context、PTX JIT |
| P2 | interop、texture/surface、profiler、checkpoint 等 |

P0/P1 必须全部通过。P2 可以有不支持项，但必须记录在兼容矩阵中。

## 13.3 的增量

13.3 新增或变化的能力要单独设用例：

- `cuStreamBeginRecaptureToGraph`：拓扑匹配、非法重捕获、失败后的 graph 状态。
- Green Contexts：默认 stream 可选，SM/workqueue 资源划分。
- MPS 静态分区错误隔离：分区创建、非法分区 ID、跨分区故障隔离。
- DMA-BUF mmap：设备属性、export、`mmap` 和同步协议；不支持平台明确报错。
- CUDA 13.3 新 API 的错误路径和资源清理。

NVIDIA 13.3 release notes 中的 known issues 要映射到测试豁免，不能把已知 bug 当作兼容通过。13.3 的 cuBLAS PDL known issue 也应在库层复现，确认 API 层不会把错误的依赖顺序误判为通过。

## 硬门槛

- 声明范围内 API 覆盖率 100%。
- P0/P1 用例通过率 100%。
- 静默错误、崩溃、挂起、内存越界、context 损坏为 0。
- 不支持的能力返回文档定义的错误，不能返回成功，也不能静默降级。
- 13.3 新增 API 全部有实现或不支持结论。
- 13.2 已通过的 API 用例在 13.3 上不能退化。

判定以对象行为为准。比较错误类别、同步点、结果和资源状态；不比较 handle 数值、地址和错误字符串。

满足以上硬门槛，才能判定 CUDA 13.3 API 兼容验收通过；缺少任一项，只能标注“CUDA 13.3 部分兼容”。性能、框架、模型和应用需要在上层单独验收。
