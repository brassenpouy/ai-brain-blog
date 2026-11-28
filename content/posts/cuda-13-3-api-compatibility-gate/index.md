---
title: "CUDA 13.3 API 兼容验收标准"
date: "2026-11-26T22:13:00+08:00"
lastmod: "2026-11-28T11:34:25+08:00"
description: "GPU 厂商对齐 CUDA 时，以 CUDA 13.3 为例的源码级 API 兼容验收标准；不覆盖 ABI/二进制兼容。"
tags:
  - "CUDA"
  - "兼容性"
  - "测试"
  - "GPU"
draft: false
slug: "cuda-13-3-api-compatibility-gate"
---

## 背景与边界

不少 GPU 厂商在软件栈上对齐 NVIDIA：头文件、Runtime/Driver API、常用库和框架接口尽量保持 CUDA 形态，降低客户的迁移成本。这个策略有效，但“兼容 CUDA”经常停留在市场描述，缺少可验收的边界。

本文以 CUDA 13.3 为例，给出一套源码级 API 兼容验收标准，把“能不能叫兼容”拆成可执行的条件。

这里只讨论源码/API 兼容：应用使用 13.3 头文件和 API，重新编译后运行。已有 CUDA 二进制不重编译直接运行、soname、符号版本、结构体布局和 driver ABI 与 NVIDIA 一致，不在范围内。

cuBLAS、cuDNN、NCCL、PyTorch 和模型精度也不在这一层验收。API 通过，不等于软件栈兼容。

兼容范围按声明处理：覆盖 13.3 全部公开 Runtime/Driver API，才可以叫“全量兼容”；只覆盖子集，只能标“部分兼容”。

## 13.3 基准

以 CUDA 13.3 GA 为基准：驱动 610.43.02，cudart 13.3.29。参考环境用于差分测试，不要求厂商驱动号与 NVIDIA 相同。

API 变化以 [CUDA 13.3 Release Notes](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html)、[Runtime API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/index.html) 和 [Driver API](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/index.html) 为准。

兼容矩阵至少包含：GPU 架构、OS、驱动版本、Runtime 版本、MPS/MIG/Green Context 模式、PTX/cubin 路径。

## 验收范围

### Host API

Driver API 至少覆盖：

- 初始化、版本、设备和 primary context
- context、module、library
- memory、VMM、async allocator、multicast
- stream、event、stream memory operations
- graph、tensor map、peer access、IPC
- green context、logical endpoint、coredump、checkpoint

Runtime API 至少覆盖：

- 设备管理、错误处理、版本管理
- memory、memory pool、unified memory
- stream、event、execution control
- graph、external interop、peer access
- occupancy、texture/surface、library、profiler
- green context、logical endpoint

Runtime 封装 Driver，但默认 stream、primary context 和错误语义有自己的规则，两条线分开测试。

### 设备侧 API

- Device Runtime API：动态并行、device malloc/free、memcpy/memset、stream/event、device synchronize、嵌套 launch。测试需启用 relocatable device code 和 `cudadevrt`。
- PDL：`cudaLaunchAttributeProgrammaticStreamSerialization`、`cudaLaunchAttributeProgrammaticEvent`、`cudaTriggerProgrammaticLaunchCompletion`、`cudaGridDependencySynchronize`。
- Cooperative Groups：block/grid/cluster sync、`memcpy_async`、reduce。
- Cluster/DSMEM：cluster 同步、共享内存映射；只在支持架构上测。
- libcu++：`cuda::atomic`、`cuda::barrier`、`cuda::pipeline`、`memcpy_async`。
- Tensor Core wrapper：`nvcuda::wmma`、MMA/WGMMA/tcgen05 相关接口。
- Device Graph Launch：设备侧 graph launch 和更新。
- 内建与 intrinsics：`threadIdx`、`blockIdx`、`__syncthreads`、`__syncwarp`、shuffle/ballot、`__ldg`、atomics、`printf`/`assert`。

设备侧测试要单独看设备堆 OOM、嵌套 launch 失败、cluster 尺寸、device graph 生命周期和错误传播。

### 优先级

| 级别 | 范围 |
| --- | --- |
| P0 | 初始化、context、module、memory、stream、event、kernel launch、错误模型、版本管理 |
| P1 | graph、PDL、device runtime、cooperative groups、cluster/DSMEM、device graph、VMM、async allocator、P2P/IPC、MPS/MIG、green context、PTX JIT |
| P2 | interop、texture/surface、profiler、checkpoint 等 |

P0/P1 必须全部通过。声明全量兼容时，P2 也纳入验收；只做部分兼容时，P2 可以列不支持项，但必须记录在兼容矩阵中。

## 测试方法

### 接口

- `cudaRuntimeGetVersion()` 和 `cuDriverGetVersion()` 返回 13.3 对应版本号 13030。
- 使用 13.3 头文件可以编译，API 参数、枚举、回调和常量与 13.3 一致。
- 声明支持的符号在编译和链接阶段可用；`cuGetProcAddress` 的版本化入口可用。
- deprecated 符号仍能编译、链接、运行。

### 状态与并发

- context：primary context 的 retain/release、多线程 current context、reset 后资源失效。
- module/function：PTX、cubin、fatbin 加载，JIT 选项，符号查找，卸载后调用。
- memory：malloc/free、pitched、UVM、VMM、async pool、OOM、非法指针、double free。
- stream/event：default stream、non-blocking stream、优先级、回调、跨 stream 依赖、event 计时。
- graph：显式构图、capture、instantiate、update、销毁，以及失败后的 graph 状态。
- 同步与错误：device/stream/event sync 的覆盖范围，异步错误暴露点，sticky error 和 last error 清除规则。
- 多线程、多 stream、多卡、P2P、IPC、MPS/MIG。
- PDL：后继 kernel 提前启动、trigger 条件、事件触发、无 trigger 的退化路径、stream/event 顺序，以及 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_*` 映射。
- PTX 和 cubin 分开测试。13.3 文档明确 PTX 在旧驱动上受限，不能混在一起判断。
- 版本混用按 13.3 的 version mixing rules 测试，不假设任意组合都可用。

### 故障注入

每个高风险用例独立进程运行，避免 sticky error 和 context 损坏污染后续测试。

- 注入 OOM、资源耗尽和分配失败。
- 在 stream 执行中销毁 event、graph、memory 或 context。
- 构造非法 handle、错误 device、错误 flag 和越界参数。
- 观察崩溃、挂起、内存越界和静默成功，而不只是看返回值。

### 判定依据

以 13.3 参考环境做差分测试。比较错误类别、同步点、结果和资源状态；不比较 handle 数值、地址和错误字符串。

文档有明确规定的，以文档为准；文档不明确时，用参考环境确认可观察行为。

## 13.3 的增量

- `cuStreamBeginRecaptureToGraph`：拓扑匹配、非法重捕获、失败后的 graph 状态。
- Green Contexts：默认 stream 可选，SM/workqueue 资源划分。
- MPS 静态分区错误隔离：分区创建、非法分区 ID、跨分区故障隔离。
- DMA-BUF mmap：设备属性、export、`mmap` 和同步协议；不支持平台明确报错。
- PDL 与 cuBLAS：复现 13.3 known issue 中的依赖顺序问题，确认 API 层不会把错误的依赖顺序误判为通过。
- 13.3 新 API 的错误路径和资源清理。

NVIDIA 13.3 release notes 中的 known issues 要映射到测试豁免，不能把已知 bug 当作兼容通过。

## 通过标准

- 声明范围内 API 覆盖率 100%。
- P0/P1 用例通过率 100%。
- 静默错误、崩溃、挂起、内存越界、context 损坏为 0。
- 不支持的能力返回文档定义的错误，不能返回成功，也不能静默降级。
- 13.3 新增 API 全部有实现；只做部分兼容时，才允许记录“不支持结论”。
- 13.2 已通过的 API 用例在 13.3 上不能退化。
- 兼容矩阵、测试日志和未支持清单可审计。

满足以上标准，才能判定 CUDA 13.3 API 兼容验收通过；缺少任一项，只能标注“CUDA 13.3 部分兼容”。性能、框架、模型和应用需要在上层单独验收。
