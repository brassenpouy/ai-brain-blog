---
title: "CUDA exception 进程隔离：边界、交互和测试"
date: "2026-11-23"
lastmod: "2026-11-23T23:45:16+08:00"
description: "梳理 CUDA 中 exception 进程隔离的错误层级、context/MPS/MIG 边界、相关 feature 交互和验收标准。"
tags:
  - "CUDA"
  - "进程隔离"
  - "GPU 驱动"
  - "MPS"
  - "MIG"
draft: false
slug: "cuda-exception-process-isolation"
---

## 先定义隔离目标

CUDA 没有一项单独名为 “exception process isolation” 的官方 API。它更像一组语义：一个进程触发 GPU exception 后，错误只终止或污染该进程的 CUDA 状态，其他进程继续运行；错误能归因到责任进程；必要时只重建受影响的 context 或分区。

这个目标至少要拆成三层：

- **进程隔离**：进程崩溃、被杀或 CUDA 状态损坏，不影响其他进程。
- **context 隔离**：每个 context 有独立地址空间、队列和调度状态。
- **设备隔离**：多租户、MPS、MIG、SR-IOV 或虚拟机之间，连硬件错误都不互相传播。

三层不是一回事。context 有 fault isolation，不代表 MPS 客户端之间能隔离 fatal fault；MPS 有独立地址空间，也不代表它有完整的 error isolation。

## CUDA 错误分几层

CUDA runtime 的错误不是同一种严重程度。设计隔离功能前，先把错误分类。

| 错误类型 | 例子 | 作用范围 | 恢复方式 |
| --- | --- | --- | --- |
| API / 参数错误 | `cudaErrorInvalidValue` | 当前调用 | 修正参数，context 继续用 |
| 启动配置错误 | 资源不足、block 配置非法 | 当前 launch | 调整配置后重试 |
| 致命执行错误 | `cudaErrorIllegalAddress`、`cudaErrorAssert`、`cudaErrorLaunchFailure` | 当前进程的 CUDA 状态 | 官方要求终止并重启进程 |
| context 失效 | `cudaErrorContextIsDestroyed` | 当前线程的 context | 创建新 context 或重启进程 |
| 设备级错误 | Xid、ECC、GPU reset | 整卡或更大范围 | 设备复位、隔离节点 |

关键在第三类。NVIDIA 对 [`cudaErrorIllegalAddress`、`cudaErrorAssert`、`cudaErrorLaunchFailure`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html) 的说明都是：设备进入不一致状态，后续 CUDA 工作会继续返回同一错误，要恢复 CUDA 必须终止并重新启动进程。

`cudaDeviceReset()` 只销毁当前设备、当前进程里的分配和状态，不是整卡 reset（[CUDA runtime 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)）。它解决不了其他进程的 context 污染，更解决不了硬件级 Xid。

## context、MPS、MIG 的边界

### CUDA context

NVIDIA 在 [MIG 文档](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html) 里把 GPU context 类比为 CPU process：独立地址空间、独立调度、fault isolation。一个 context 的 fatal error 不应直接破坏另一个 context。

这不等于“exception process isolation 已经完成”。一个进程可以有多个 context，一个 context 的 fatal error 仍可能要求整个进程退出；驱动、MPS server 和硬件级错误会跨过 context 边界。

### MPS

MPS 经常被当成多进程隔离方案。它让多个进程共享 GPU，并给每个 client 独立 GPU 地址空间；NVIDIA 的 [MPS 文档](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html) 也明确写了“limited form of error containment”。

但它的隔离粒度不是 per-client：

- 一个 client 触发 fatal GPU fault，错误会限制在共享该 GPU 的 client 子集内。
- 错误会通知该子集内的所有 client，但不指出是哪个 client 触发的。
- 受影响的 client 需要自行退出；MPS server 进入 `FAULT`，拒绝新 client，直到相关 client 全部退出后重建 context。
- 其他 GPU 上的 client 不受影响。
- client 未同步就异常退出，可能让 MPS server 和其他 client 进入未定义状态，出现挂起、异常失败或数据损坏。

MIG 文档的对比表把 MPS 的 Error Isolation 标为 `No`。这里的 `No` 指没有 MIG 那种按实例的 error isolation；MPS 仍有有限的 fault containment。对“exception 进程隔离”来说，MPS 是需要纳入设计的共享拓扑，不能作为隔离保证。

Reddit r/CUDA 的一条 [GPU 分片讨论](https://www.reddit.com/r/CUDA/comments/1v0cqot/how_we_do_fractional_gpu_slicing_without_nvidia/) 也提到，MPS 下一个进程的 invalid memory access 会影响整个 MPS server 和其 client context。这和官方文档的 limited containment 是一致的：地址空间独立，不等于错误隔离。

### MIG

MIG 把 GPU 划成 GPU Instance 和 Compute Instance，计算、显存和带宽物理分区。[MIG 文档](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html) 给出的 Error Isolation 是 `Yes`，memory bandwidth QoS 也是 `Yes`。

代价是粒度粗、配置受 profile 限制、跨分区 IPC 受限。MIG 适合强隔离租户，不适合把一个进程的异常隔离成轻量 context 级操作。

## 与现有 feature 的交互

这个 feature 涉及整条 CUDA 栈：

| feature | 交互点 | 需要明确的行为 |
| --- | --- | --- |
| CUDA runtime / driver | primary context、显式 context、错误码 | 故障进程收到的错误码保持不变；其他进程不被污染 |
| streams / events | fatal error 会污染整个 context | 同 context 的流全部失效；跨 context 不传播 |
| async / copy engine | 在途 memcpy、DMA、GPUDirect | 故障时取消或 drain，避免悬挂 fence |
| CUDA Graphs | graph exec、memory pool、capture | launch 失败如何归因；graph 资源是否回收 |
| Unified Memory | page fault、迁移、OOM | fault storm、迁移失败和进程退出如何隔离 |
| CUDA IPC / P2P | 共享内存 handle、event、peer access | 一端崩溃时 handle 生命周期和错误传播 |
| NCCL / 多机 | 通信域、rank 失败 | 单进程隔离不等于作业隔离；集合通信可能整体失败 |
| MPS | server/client、共享 GPU 子集 | fatal fault 的报告范围、server `FAULT`、重启策略 |
| MIG | GI/CI、profile、有限 IPC | 分区内故障、跨分区错误、设备级 reset |
| vGPU / SR-IOV | VM 边界、VF reset | VM 内隔离与整卡错误边界 |
| 容器 / K8s | PID namespace、device plugin、cgroup | 进程身份、设备清理、重启与调度 |
| ECC / Xid / reset | 硬件错误、驱动日志；Xid 是驱动写到内核日志的错误报告（[文档](https://docs.nvidia.com/deploy/xid-errors/latest/introduction.html)） | 无法隔离的设备级错误如何升级处理 |
| 调试 / profiling | compute-sanitizer、cuda-gdb、Nsight、CUPTI | 故障进程可诊断，其他进程不被误杀 |

几个容易漏的点：

**Unified Memory。** 页错误发生在 GPU 访问未驻留页时，驱动需要处理迁移、映射和 OOM。故障进程的 page fault 不能拖住其他 context；同时要防止 fault storm 把驱动队列打满。

**CUDA IPC 和 P2P。** 地址空间隔离不等于共享对象隔离。导出的内存 handle、event 和 peer mapping 都有生命周期，一端异常退出后，另一端可能拿到失效 handle。

**CUDA Graphs。** graph 把多个 kernel 和拷贝封装成一次 launch。异常发生时，要能判断是 graph 定义、实例化、参数更新还是执行阶段失败，并回收 graph 私有内存池。

**MPS 的多用户模式。** MPS 可以配置成不同 UID 共享一个 server，这会放弃用户间隔离。fatal fault 可能影响其他用户的 client，不能把这种部署当成安全隔离。

## feature 设计要先定语义

### 故障域

先定义 fault domain：进程、context、MPS client group，还是 MIG instance。不同域的恢复动作不同。

- 进程级：终止故障进程，清理它的 context、VA、队列、事件和 IPC 资源。
- context 级：销毁故障 context，保留进程和其他 context。
- 客户端组级：MPS 下按受影响 GPU 子集处理。
- 设备级：Xid、ECC、硬件 reset，只能升级为设备复位或节点隔离。

### 错误归因

每个 exception 至少要能关联：

- PID、UID、namespace
- context ID、device ID
- channel / queue / engine
- 触发时的 CUDA API 和 kernel
- Xid、ECC、reset 事件

MPS 目前对非内存类 fatal fault 不一定能指出具体 client。新 feature 如果承诺进程级归因，需要补上这条链路。

### 恢复动作

故障进程的 CUDA 状态通常不可恢复。隔离功能能做的是：

1. 停止该 fault domain 的新提交。
2. 取消或 drain 在途工作。
3. 等待 fence 和 DMA 完成或安全终止。
4. 释放 VA、内存、队列、event、graph 和 IPC handle。
5. 销毁 context，必要时重建 MPS/MIG 资源。
6. 把错误返回给责任进程，其他域继续运行。

设备级 fatal error 不能走这条路径。必须明确升级条件，避免把“进程隔离”误宣传成“硬件故障隔离”。

### CUDA API 语义

隔离功能不能改变 CUDA API 的既有语义：

- 致命错误在故障进程内仍应表现为对应的 CUDA 错误。
- `cudaGetLastError` / `cudaPeekAtLastError` 的可见性不应跨进程串扰。
- `cudaDeviceReset` 仍只作用于当前进程的当前设备。
- MPS/MIG 模式下的错误码、server 状态和 IPC 行为要逐项写清。
- fork、多线程、多个 context、driver/runtime 混用都要覆盖。

## 开发难点

**错误归因。** GPU 错误是异步上报的。CUDA API 可能在后一次调用才返回前一个 kernel 的错误；MPS 对非内存类 fatal fault 不指出具体 client；Xid 是驱动日志，不天然带 PID 和 context。要把异常限制到进程，driver 必须维护 process、context、channel、queue 和 fault event 的对应关系。

**故障域恢复。** CUDA fatal execution error 不是“清一个错误码”就能恢复。`cudaErrorIllegalAddress`、`cudaErrorAssert`、`cudaErrorLaunchFailure` 都要求进程终止。隔离功能要决定是杀进程、销毁 context，还是重建 MPS client group；每种动作的资源清理边界不同。

**在途工作。** 异常发生时可能还有 kernel、async copy、DMA、fence、event 和 graph 在执行。直接杀进程会留下悬挂队列、未完成 fence 和 IPC handle。driver 需要先停止新提交，再取消或 drain 旧工作，最后回收 VA、内存、队列和事件。

**MPS 的共享状态。** MPS client 有独立地址空间，但共享 MPS server 和 GPU 子集。一个 client 的 fatal fault 会让 server 进入 `FAULT`，影响共享同一 GPU 的多个 client；非内存 fault 还不一定指出责任方。想在这个拓扑上做 per-process 隔离，必须改 MPS 的故障处理和归因链路。

**MIG 的粒度。** MIG 有实例级 error isolation，但配置受 profile 限制，跨分区 IPC 受限，也不能把一个进程的异常轻量地隔离成子分区。隔离功能要识别 MIG 拓扑，避免把设备级故障误判成进程级故障。

**跨 feature 状态。** Unified Memory 的 page fault、CUDA Graphs 的私有内存池、IPC/P2P 的共享 handle、NCCL 的通信域，都有独立生命周期。异常清理要覆盖这些对象，否则进程退出后仍会留下脏状态。

**兼容性和性能。** 每个 CUDA 版本、驱动版本、MPS/MIG 配置和容器环境都要保持错误码和 API 行为一致。隔离路径还会增加跟踪、fence 和清理开销，不能把正常路径拖慢。

**安全边界。** 恶意进程可以主动触发 fault。隔离功能要保证它不能借异常影响其他租户，也不能通过错误信息、handle 或 telemetry 泄露其他进程的数据。

## 涉及哪些软件栈模块

这个 feature 不能只做在 CUDA runtime。最小闭环要跨 user-space driver、kernel driver、firmware 和硬件；MPS/MIG 还要各自处理。

| 模块 | 负责什么 | feature 需要改什么 |
| --- | --- | --- |
| CUDA runtime | API 语义、错误码、context 初始化、`cudaDeviceReset` | 保持错误可见性和进程退出行为；通常不负责真正的隔离 |
| CUDA driver（user space） | context、stream、queue、channel、IPC、UVM、CUDA Graph | fault domain、错误归因、资源清理、MPS 接口 |
| kernel driver | 中断、MMU fault、channel teardown、reset、Xid | 故障路由、进程退出处理、设备隔离 |
| firmware / GSP | 调度、engine reset、异常上报 | 故障范围、reset 范围、上报格式 |
| GPU 硬件 | VMID、channel、engine、MMU、MIG 分区 | per-process 故障归因和隔离能力 |
| MPS | 共享 server、client、GPU 子集 | client 故障组、责任方归因、server `FAULT` 恢复 |
| MIG / NVML | GI/CI 分区、拓扑、遥测 | 识别隔离边界、分区 reset、错误上报 |
| 容器 / K8s | 进程生命周期、设备分配、重启 | 故障进程清理、设备重新调度、状态回收 |

CUDA runtime 是应用能看到的接口层；真正决定隔离能力的是 driver、kernel driver、firmware 和硬件。硬件不能提供 per-channel 或 per-VMID 的错误归因时，软件只能做到“事后杀进程”，不能保证其他进程完全不受影响。

MPS 又特殊一层：MPS server 是用户态进程，但 fault containment 依赖 driver 和后续 context 重建。MIG 则把隔离边界下移到硬件分区，但配置、IPC 和 reset 仍要 NVML/driver 配合。

## 测试场景

| 故障 | 拓扑 | 期望 |
| --- | --- | --- |
| illegal address | 两进程、无 MPS | 故障进程退出；另一进程继续 |
| assert / launch failure | 同进程多 context | 故障 context 失效；其他 context 可继续（若产品承诺） |
| OOM | 多进程共享显存 | 责任进程失败；其他进程不被 OOM 连带 |
| Unified Memory page fault | UVM + 多进程 | fault 归因正确；无 fault storm |
| CUDA Graph 执行失败 | graph + 多流 | graph 资源回收；错误可归因 |
| MPS fatal fault | 同 GPU 多 client | 受影响 client 收到错误；其他 GPU client 继续 |
| MPS client 未同步退出 | 多 client | server 不进入未定义状态；可恢复 |
| MIG fatal fault | 不同 GI/CI | 分区隔离；跨分区 IPC 行为明确 |
| Xid / ECC | 整卡 | 按设备级错误升级；不能假装进程级隔离 |
| 进程被 kill -9 | 多进程 | 资源回收；无残留 channel/fence |

### MPS 的 blast radius

MPS 测试不能只断言“故障 client 退出”。要按拓扑定义受影响集合：

- 同 GPU 子集内的 client 是否收到 fatal error。
- 其他 GPU 上的 client 是否继续运行。
- MPS server 是否进入 `FAULT`，新 client 是否收到 `CUDA_ERROR_MPS_SERVER_NOT_READY`。
- 受影响 client 全部退出后，server 是否回到 `ACTIVE`，新 client 能否正常创建 context。
- 非内存 fatal fault 是否仍然不指出责任 client；如果产品承诺进程级归因，这里就是缺口。

测试 harness 要记录每个 client 的 PID、context、device、错误码和存活状态。故障注入覆盖 idle、kernel 执行中、async copy 中、Graph launch 中，以及 `kill -9`、assert、illegal address、OOM。并发触发两个 client 的异常，观察 server 状态和恢复路径。

MPS 的校验目标不是一个进程的退出码，而是 blast radius：哪些 client 应该受影响、哪些必须存活、server 如何恢复、新请求何时可以重新进入。这个用例也可以作为进程隔离 feature 的负向基线：如果 feature 声称隔离 MPS client，这个场景必须通过；如果做不到，就要在支持矩阵里明确。

还要做并发和压力测试：两个进程同时触发异常、异常和 reset 竞争、反复创建销毁 context、MPS server 反复进入 `FAULT`、容器重启、长时间 soak。隔离功能最容易在竞态和清理路径上出问题。

## 验收指标

- 其他进程存活率：故障后非责任进程应全部继续运行。
- 错误归因准确率：PID、context、device、Xid 能对应到故障源。
- 资源回收：VA、内存、队列、event、graph、IPC handle 无泄漏。
- 恢复时间：故障进程退出到新请求可服务的耗时。
- API 语义：错误码和 API 行为与基线一致。
- 性能开销：正常路径的提交、同步和显存开销可测。
- 长稳：反复故障注入后没有死锁、悬挂 fence 或 server 状态卡死。

## 常见误区

**“MPS 有独立地址空间，所以就是进程隔离。”** 地址空间隔离不等于错误隔离。MPS 的 fatal fault 会通知共享同一 GPU 子集的多个 client，且不指出责任 client。

**“context 有 fault isolation，所以进程一定安全。”** fatal execution error 仍要求进程退出；MPS server 和硬件级错误会跨过 context。

**“MIG 能隔离所有故障。”** MIG 提供实例级 error isolation，但设备级 Xid、ECC、PCIe 或固件问题仍可能影响整卡。

**“`cudaDeviceReset` 会重置整张 GPU。”** 它只清理当前进程、当前设备的 CUDA 状态；不能修复其他进程或硬件。

`exception 进程隔离` 要解决的是：把错误限制在明确的 fault domain 内，同时保持 CUDA API 语义一致、资源可回收、错误可归因。只做异常捕获，不算进程隔离。
