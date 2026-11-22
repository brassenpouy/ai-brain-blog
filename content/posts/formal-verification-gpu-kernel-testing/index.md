---
title: "GPU kernel 的形式化验证对软件栈测试的启发"
date: "2026-11-22"
description: "从 Daniel Lemire 的帖子和 TensorLean 论文出发，解释 PTX 级形式化验证能补上测试的哪一层，以及如何落到 AI 软件栈验证。"
tags:
  - "形式化验证"
  - "GPU"
  - "软件栈"
  - "测试"
  - "Lean"
draft: false
slug: "formal-verification-gpu-kernel-testing"
---

## 一条帖子背后的论文

Daniel Lemire 在 x.com 上说，他正在把越来越多的高性能代码拿去做形式化验证。他的原话很直接：这些项目为了性能“带着剪刀奔跑”，测试和验证已经很多，但形式化验证仍然值得加（[原帖](https://x.com/lemire/status/2124175237419815389)）。

帖子配的论文是 Columbia 和 CertiK 的 **From Math to Machine, in Minutes: Formal Verification of Production GPU Kernels**，工具叫 TensorLean（[项目页](https://tensorlean.ai/)，[论文](https://arxiv.org/html/2611.04447v1)）。

论文给出的结果很具体：

- 在 PTX 层面验证了 149 个真实 kernel 实例。
- 83 个证明正确，66 个发现问题，合计找到 38 个 bug。
- 覆盖 FlashAttention-2 到 FlashAttention-4、DeepGEMM、Liger Kernel 等生产 kernel。
- 这些问题在此前的测试里藏了几个月到几年。

这对软件栈测试的启发是：**测试和形式化验证覆盖不同层面，应该叠起来用**。

## 测试为什么抓不到这些 bug

普通 kernel 测试通常长这样：准备一组输入，调用 kernel，和 PyTorch/CPU 参考实现比较，误差在容差内就算通过。

它能发现很多问题，但有三个天然盲区：

**输入覆盖有限。** 测试用例来自人选或随机生成，不可能枚举所有 shape、stride、dtype、边界值和异常输入。

**线程调度覆盖有限。** GPU kernel 的正确性依赖同步、异步拷贝、内存序和线程交错。测试跑到的只是其中一部分调度，换一张卡、换一个编译器版本，可能走到另一条路径。

**测试的是源码行为，不一定是编译后的行为。** 编译器会重排指令、融合循环、改变寄存器使用和内存访问。源码正确，生成的 PTX 仍可能出错。

形式化验证补的正是这三块：它针对编译后的 PTX，要求在所有允许的输入和所有执行下都满足规格。测试是抽样，证明是全称。

## TensorLean 验证了什么

TensorLean 的核心设计是把 kernel 正确性拆成两层：

- **数学层**：这个 kernel 应该计算什么，例如 `O = softmax(QKᵀ/√d)V`。
- **内存层**：每个 tensor 元素对应哪块 GPU 内存，stride 和 layout 是什么。

规格写在 tensor 层，工具再自动降到 PTX 内存层。这样一份 attention 规格可以覆盖多个 FlashAttention 变体，不需要为每个 kernel 手写一份 PTX 级证明。

TensorLean 验证的是编译后的 PTX；CUDA 源码不直接进入证明。证明要求：

- 对所有满足前置条件的输入；
- 对所有允许的线程调度；
- 输出在给定浮点容差内；
- 不改动不该改的内存。

这带来一个很重要的区别：测试告诉你“这组输入没问题”，形式化验证告诉你“在这套 PTX 语义和前置条件下，不存在反例”。

论文也明确写了边界：它假设 `ptxas` 和硬件正确实现 PTX，不证明 kernel 一定终止，也不处理所有 PTX 特性。证明是“部分正确性”：完成的执行不会给出错误结果，但不保证一定结束。

## 对软件栈测试的五个启发

### 1. 规格应该成为测试资产

传统测试的预期值通常写在测试代码里，分散且隐式。形式化验证要求把“正确”写成明确规格：数学表达式、内存布局、前置条件、浮点容差。

这份规格可以直接复用：

- 生成参考实现；
- 作为 differential testing 的 oracle；
- 作为编译器改写的检查目标；
- 作为 kernel 交接和 review 的文档。

规格写错，证明就没有意义。所以规格本身也要测试，通常拿 PyTorch/CPU 参考实现做交叉验证。

### 2. 要验证编译后的代码

只测 Python 层或 CUDA 源码，会漏掉 codegen 和指令调度问题。TensorLean 直接验证 PTX，说明软件栈测试不能只停在框架层。

对 AI 芯片软件栈来说，这对应一个很实际的门禁：kernel 库、编译器后端、PTX 生成器每次改动，都应该有一层针对生成代码的语义检查；端到端跑模型不能代替这一层。

### 3. 形式化验证适合高风险、可复用的 kernel

不适合把所有 kernel 都拿去做证明。论文里 454 条 PTX 指令对应约 34,000 行证明，平均每条指令约 75 行；这还没有算规格和共享 lemma。

更现实的策略是分层：

- 高频、核心、容易静默出错的 kernel 做形式化验证，例如 attention、softmax、layernorm、GEMM、量化 kernel、MoE 路由。
- 其余 kernel 用参考实现对比、property-based testing、fuzzing 和 sanitizer 覆盖。
- 编译器 pass 用 translation validation 或差分测试，检查优化前后语义是否一致。
- 运行时、分布式和硬件层继续用 API contract、trace replay、故障注入和长稳测试。

### 4. 证明不能替代性能和集成测试

形式化验证证明的是功能正确性，不是速度。一个 kernel 可以证明完全正确，同时在目标 GPU 上很慢，或者在多卡通信、显存压力、批处理变化下表现很差。

所以测试金字塔不会变成“证明 + 结束”，而是多一层：

| 层次 | 主要手段 | 回答的问题 |
| --- | --- | --- |
| Kernel 语义 | 形式化验证、差分测试、fuzzing | 算得对吗？所有输入和调度都成立吗？ |
| 编译器 / runtime | translation validation、API contract、trace replay | 生成的代码和运行时行为一致吗？ |
| 模型集成 | 端到端模型、评测集、收敛/CUDA Graph | 模型整体还对吗？ |
| 性能 | 微基准、profiling、SLO 压测 | 跑得快吗？延迟和吞吐达标吗？ |
| 硬件 / 集群 | soak、ECC、故障注入、多机扩展 | 能长期稳定运行吗？ |

### 5. 证明的信任边界要写清楚

形式化验证的结论有前提。TensorLean 的证明依赖：

- PTX 语义模型；
- Lean 证明检查器；
- 前置条件和内存布局；
- 浮点容差；
- `ptxas` 和硬件正确实现 PTX 的假设。

这些假设本身就是测试对象。验收报告要列出证明了什么、没证明什么、依赖什么版本，不能只写一句“已形式化验证”。

## GPU driver 是软件栈的核心层

GPU driver 属于 host 侧软件栈的核心层。向上承接 framework、runtime 和 compiler，向下通过 ioctl、MMIO 和中断管理 GPU 的资源、队列、内存、同步和错误恢复。TensorLean 证明 PTX kernel；driver 管这些 kernel 怎么被提交、同步和回收。两者层级不同，但都在软件栈里。

形式化方法可以借用，证明对象要换成 driver 的 contract。先做局部契约，不要一上来覆盖整个驱动。

| driver 部件 | 适合的手段 | 重点 |
| --- | --- | --- |
| UAPI / ioctl | contract test、model-based testing、syzkaller | 参数校验、错误码、状态转换 |
| GPU VA / 页表 | TLA+、Alloy、故障注入 | map/unmap、pin/unpin、迁移、回收 |
| command ring / 提交队列 | Loom、Verus、Kani、CBMC | 生产者消费者、内存序、回绕 |
| fence / event / 中断 | TLA+、Spin、压力测试 | 完成顺序、丢失/重复中断、超时 |
| reset / recovery / suspend | TLA+、故障注入 | 状态机、队列和内存恢复 |
| 硬件 / 固件 | 仿真、FPGA、真实卡 | MMIO、固件、硬件行为 |

driver 的很多 bug 不在单条指令，而在状态机和并发交错：一个 ioctl 的失败路径没有回滚，fence 等待和 reset 撞在一起，多进程同时 map 同一块 VA，suspend 时还有队列没排空。这些场景适合模型检查和小范围代码验证。

工具上，TLA+ 适合协议和状态机，Loom 适合 Rust 并发，Kani/Verus/CBMC 适合 C/Rust 的局部数据结构和边界路径；syzkaller、KASAN、KCSAN、lockdep 和故障注入负责把真实系统里的异常路径跑出来。

driver 形式化有明显信任边界。证明通常假设 MMIO、GPU 固件和硬件行为符合模型；这些假设要靠仿真、FPGA、真实卡和长稳测试覆盖。

从一个小状态机和一个无锁队列开始，比从“整个 driver 形式化”开始更现实。

## 如果测的是 CuTe DSL

CuTe DSL 的测试对象比手写 CUDA 多几层。按 NVIDIA 文档，它是 Python 嵌入式 DSL：`@jit` / `@kernel` 经过 AST 重写和 tracing，生成 IR，再 JIT 编译成 PTX，最后由 `ptxas` 生成 SASS（[CuTe DSL 介绍](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)）。

TensorLean 在 PTX 层做证明。它不覆盖 Python frontend、IR lowering 和 JIT 缓存。CuTe DSL 的测试要按这些层分开。

| 层 | 测什么 | 方法 |
| --- | --- | --- |
| Layout algebra | shape、stride、cosize、composition、swizzle | 与 C++ CuTe 参考实现做 property-based testing；必要时用 SMT 检查整数约束 |
| DSL frontend | AST 重写、tracing、static/dynamic layout、类型推断 | 同一 kernel 用不同写法生成 IR/PTX，做差分测试 |
| IR / lowering | tiled copy、MMA、TMA、cluster、pipeline | IR 语义检查、translation validation、与参考 kernel 对比 |
| PTX | 关键 kernel 的功能正确性 | TensorLean 式形式化验证，覆盖所有输入和线程调度 |
| JIT / runtime | 缓存键、架构、编译选项、动态 shape | 缓存失效回归、ABI 测试、`keep-ptx` 检查生成物 |
| 性能 | 吞吐、延迟、占用、显存 | benchmark、Nsight Systems/Compute；证明不覆盖性能 |

CuTe DSL 和 TensorLean 在方法上有一个直接对应：**数学计算和 layout 分开写规格**。

CuTe 的 layout algebra 本身就是独立对象。可以先证明 layout 映射正确，再证明这个 layout 上的 kernel 数学正确。测试时也应该分开：

- layout 层检查 `shape/stride/cosize`、对齐、越界、swizzle 和 tile 映射。
- kernel 层检查 `O = softmax(QKᵀ/√d)V` 这类数学结果，容差单独定义。

CuTe DSL 还有一些手写 CUDA 不会遇到的测试点：

- **JIT 缓存键**：架构、编译选项、layout、dtype 变化后是否重新编译；缓存命中是否拿到旧 kernel。
- **动态 shape**：文档说明 CuTe layout algebra 目前只支持 32 位 shape/stride，边界值要单独覆盖。
- **生成物检查**：用 `keep-ptx` 保留 PTX，检查关键指令、寄存器、共享内存和预期是否一致。
- **异步 pipeline**：TMA、async copy、warp specialization 会改变线程调度和内存序，普通随机输入测不到这些路径。

形式化验证在这里的落点要往后放：先做 layout algebra 和 frontend 的差分测试，再选一个高风险 kernel 对生成的 PTX 做证明。论文明确说它验证的部分 PTX 是手写的；CuTe 生成的真实编译产物包含内联和优化，证明难度更高。

这层仍然要保留信任边界。CuTe DSL frontend、IR lowering、`ptxas` 和硬件都不在 TensorLean 的证明范围内。PTX 证明通过，只能说明这段 PTX 符合规格，不能说明 CuTe 编译器没有错误地生成它。

## 怎么落到 AI 芯片软件栈

一个可执行的路线：

1. 画出层间契约：framework/runtime、driver UAPI、MMIO/firmware 各层负责什么。
2. 选 3～5 个高风险 kernel，写 tensor 级规格，用 PyTorch/CPU 参考实现校验规格。
3. 对编译后的 PTX 做形式化验证，记录 PTX ISA、编译器版本、GPU 架构和容差。
4. 在 driver 里选 1～2 个状态机（reset/recovery、VM map/unmap）和 1 个无锁结构（提交队列或 fence），做局部模型检查。
5. 把证明和 driver 检查接入 CI，与差分测试、fuzzing、sanitizer、故障注入一起跑。
6. 端到端模型、性能、硬件和长稳测试继续保留，不用证明替代。
7. 把规格、证明、未证明假设和已知 bug 一起交给客户或下游团队。

这条路不会便宜，但比让一个静默 kernel bug 进入训练集群要便宜。

## 边界

TensorLean 的结果很强，但不是通用解决方案：

- 论文验证的部分 PTX 是手写的，真实编译器的内联和优化会让证明更难。
- 没有证明终止性，也没有证明 ISA 级 undefined behavior 不存在。
- PTX 语义、`ptxas` 和硬件本身不在证明范围内。
- 浮点容差和规格需要人来定，规格错误不会被证明检查器发现。
- 性能、分布式通信、数据管线和硬件可靠性仍然要靠传统测试。

对软件栈测试的启发是：**把“算得对”从抽样判断升级成可检查的规格，把验证对象从源码下移到生成代码和 driver 契约，然后让形式化证明、差分测试、集成测试和性能测试各管一层。** 形式化验证是其中一层，不能替代其他层。
