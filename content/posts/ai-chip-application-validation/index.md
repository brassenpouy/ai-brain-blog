---
title: "AI 芯片交付前的应用层验证"
date: "2026-11-22"
description: "从算子正确性、模型收敛、单机与集群性能，到推理 SLO 和稳定性，整理一套 AI 算力芯片与软件栈交付前的应用层测试框架。"
tags:
  - "AI 芯片"
  - "测试"
  - "软件栈"
  - "性能"
draft: false
slug: "ai-chip-application-validation"
---

## 先看一个交付场景

一颗新 AI 芯片跑 GEMM 微基准，峰值算力很漂亮。客户拿 Llama 训练脚本一跑，step time 抖动，loss 曲线和参考平台对不上；换成推理服务，吞吐上去了，p99 延迟却频繁超标。

问题通常不是某一个 kernel 太慢，而是交付前没有从应用层把整条链路验证完整。

这里不谈封装、良率、DFT（可测试性设计），也不谈单颗芯片的电气测试。那些是芯片团队的基础工作。本文只谈一件事：客户拿到卡、驱动和软件栈之后，能不能把模型跑对、跑快、跑稳。

## 应用层验证的五个层次

应用层验证不是“跑几个模型看看”。它要从算子一直走到客户真实 workload，每一层回答不同的问题。

![AI 芯片应用层验证的五个层次](/images/ai-chip-validation-layers.svg)

| 层次 | 要回答的问题 | 典型产出 |
| --- | --- | --- |
| 算子与数值 | 每个算子算得对吗？ | op 覆盖率、误差报告、特殊值结果 |
| 模型正确性 | 模型能收敛、输出质量一致吗？ | loss 曲线、评测分数、端到端输出对比 |
| 单机与系统性能 | 单卡、单机能跑多快？ | 微基准、step time、tokens/s、利用率 |
| 集群与服务 | 多机扩展、推理 SLO（服务等级目标）能守住吗？ | scaling efficiency、TTFT、p99、goodput |
| 客户验收 | 客户自己的模型和流量能跑吗？ | 客户 workload 报告、验收结论 |

越往下越基础，越往上越接近客户。底层不过关，上层数据没有意义；只过底层，也不代表客户能用。

## 正确性：先保证算对

第一层是算子和数值。需要覆盖客户会用到的算子，不只是模型里最常见的那些。

- 和参考实现对比，按数据类型设置合理容差。FP8、FP4 不能按 BF16 的误差标准要求。
- 检查 NaN、Inf、极大值、极小值、空张量和异常 shape。
- 检查动态 shape、变长序列、广播、转置和非连续内存。
- 检查有没有算子悄悄回退到 CPU 或低效路径。回退后结果可能对，但性能已经不是一个产品。

第二层是模型正确性。只对比单个算子不够，模型会把误差放大。

- 训练侧看 loss 曲线、梯度范数、最终评测指标，而不只是“能跑完”。
- 推理侧看 logits、生成结果和评测集分数，确认量化、编译和 kernel 替换没有改变质量。
- 同一模型要覆盖 eager、图编译、不同 batch 和不同精度，确认结果在容差内。
- 对训练还要验证 checkpoint 保存和恢复；恢复后的 loss 应该接得上，而不是重新开始。

工具上，PyTorch 的 `torch.testing.assert_close`、ONNX Runtime、`compute-sanitizer` 和框架自带的算子测试能覆盖不少基础问题。真正的模型清单，还是要来自客户和内部模型仓库。

软件栈本身也要一起验收。PyTorch/JAX 的 eager、`torch.compile`/XLA、Hugging Face、vLLM、SGLang、TensorRT-LLM 等常用路径要能直接跑；动态 shape、量化、图捕获、多机启动要覆盖；日志和 profiler 要能定位到算子。最关键的是确认没有悄悄回退到 CPU 或慢路径。

## 性能：从微基准到客户 SLO

微基准要做，但只能回答局部问题。GEMM、attention、memcpy、all-reduce 这些测试能快速定位峰值算力、显存带宽和通信瓶颈。

更重要的是模型级和端到端指标：

| 场景 | 关键指标 | 还要看 |
| --- | --- | --- |
| 训练 | step time、samples/s、time-to-accuracy、scaling efficiency、MFU（模型 FLOPs 利用率） | loss 收敛、显存余量、checkpoint 时间 |
| 推理 | TTFT（首 token 延迟）、TPOT（每输出 token 时间）、tokens/s、goodput、p99 延迟 | 并发、KV cache、量化、冷启动 |

训练不只看“每秒多少样本”，还要看多久达到目标质量。推理不只看“每秒多少 token”，还要看延迟分布和满足 SLO 的有效吞吐，也就是 goodput。

单机之后是集群。多机训练要测 all-reduce、all-gather、reduce-scatter 的带宽和延迟，测 DP、TP、PP、EP 等并行策略（数据、张量、流水、专家并行）的扩展效率，也要测节点故障后的恢复。很多“芯片性能问题”最后落在网络、存储或数据加载上。

推理服务还要测不同并发、不同输入长度、不同输出长度和不同 batch 策略下的表现。平均吞吐漂亮但 p99 超标，客户仍然不能用。

比较性能时，至少保持模型、精度、batch、输入长度、框架版本和 warmup 策略一致。固定频率跑一组数据可以看峰值，默认功耗和散热下跑一组数据才接近客户现场。

## 稳定性：能不能连续跑

实验室里跑十分钟不出错，不等于能进生产。

- 做长时间稳定性测试（soak test），观察降频、温度、功耗、ECC 和错误日志。
- 做故障注入：拔链路、制造 ECC 错误、重启节点、模拟存储抖动，验证训练能否从 checkpoint 恢复、推理服务能否摘除坏节点。
- 关注 SDC（silent data corruption）：结果错了但不报错，比直接崩溃更难发现。
- 多租户场景要验证 MIG、SR-IOV 或类似隔离机制下的 QoS，避免一个租户把显存带宽和缓存打满。
- 检查 telemetry：温度、功耗、时钟、利用率、XID/设备错误是否能被监控系统采集。

## 成熟方法：标准基准加客户 workload

业界最成熟的一套方法是 [MLPerf](https://mlcommons.org/benchmarks/)。它规定了模型、数据集、精度、质量目标和运行规则，训练看 time-to-train，推理看不同场景下的吞吐和延迟。它的价值是可比较、可复现，适合做行业基线。

但 MLPerf 不是全部。它覆盖的是有限模型和有限场景，不能替代客户自己的 workload。

更完整的做法是几层方法叠加：

- **标准基准**：MLPerf Training、Inference、Storage，建立可对外比较的基线。
- **Golden model CI**：把客户 Top N 模型和内部主力模型做成固定回归集，每次驱动、编译器、库升级都跑一遍。
- **客户 workload replay**：用真实请求 trace、真实输入长度和真实并发模式压测，而不是只跑合成数据。
- **SLO load test**：按客户延迟和吞吐目标做阶梯加压，记录 p50、p99、错误率和 goodput。
- **A/B 对比**：同一 workload 在参考平台和新平台上对比，拆开算力、带宽、通信、存储和软件开销。
- **Chaos / fault injection**：主动制造故障，验证恢复路径，而不是等生产环境第一次出事。

## 测试左移：从应用 workload 反推下层测试

应用层测试贵，不是因为它难写，而是它慢、依赖多、失败后定位远。测试左移的目标不是把验收测试搬到下层，而是把应用层暴露出的 workload 拆成更便宜、更早跑的下层用例。

真实应用至少要往下抽四类东西：

- API 调用序列：allocation、kernel launch、stream、event、同步和错误处理。
- kernel 启动参数：shape、stride、dtype、layout、动态 shape 和边界值。
- 图与算子组合：融合、布局转换、量化、回退路径和编译器 pass 顺序。
- 数值结果：参考输出、容差、NaN/Inf、收敛曲线和质量指标。

这些素材可以在 compiler、runtime/driver 和 kernel 层变成更早的测试：

| 下层 | 更早发现什么 | 常用手段 |
| --- | --- | --- |
| Compiler | 图覆盖、算子回退、数值偏差、codegen 错误 | LLVM lit/FileCheck、MLIR verifier、StableHLO/XLA 测试、IR fuzzing、与参考后端做 differential testing |
| Runtime / driver | API 语义、错误码、并发、内存生命周期、版本兼容 | API contract test、CUPTI trace replay、CUDA/HIP 测试、OpenCL/SYCL/Vulkan CTS、故障注入 |
| Kernel / library | 边界 shape、stride、dtype、精度和越界 | 与参考 kernel 对比、property-based testing、compute-sanitizer、microbenchmark |
| Virtual platform | 芯片回来前的功能与接口问题 | 功能模型、FPGA 原型、RTL 仿真；验证语义和覆盖，不替代最终性能 |

这里有两个关键做法。

一是 contract testing。上一层和下一层约定好 API 的行为：错误码是什么、stream 是否保序、异步拷贝什么时候完成、内存什么时候可以释放。下层测试按 contract 测，不必每次都拉起完整模型。

二是 trace replay。用 CUPTI、ROCm tracer 或类似机制抓一段真实应用的 API 和 kernel trace，在新 driver/runtime 上重放，比较调用是否成功、输出是否在容差内、耗时有没有明显回归。它比完整端到端测试轻，又比手写单元测试更接近真实 workload。

芯片还没回来时，可以用功能模型、FPGA 原型或 RTL 仿真跑 compiler、runtime 和 driver 测试。它们不能替代最终性能测试，但能在流片前抓出接口、语义和覆盖问题。

测试左移不意味着只做下层测试。下层测试便宜、定位快，但容易漏掉集成问题；应用层验收慢、成本高，但最接近客户。好的做法是让每一层都跑，把快的下层测试放进每次提交，把模型和客户 workload 放进 nightly 或发布前回归。

需要说明的是，这些是通用工程实践的组合，不是某一家 GPU/TPU 厂商公开的统一流程。不同团队会按产品阶段、客户 workload 和软件栈成熟度，在 CTS、trace replay、虚拟平台和 CI 闸门上做不同取舍。

## 工具地图

工具不是方法论，但能把验证做得可重复。

| 层面 | 常用工具 | 主要看什么 |
| --- | --- | --- |
| 标准基准 | MLPerf Training / Inference / Storage | 行业基线、time-to-train、SLO |
| 算子与数值 | PyTorch testing、ONNX Runtime、compute-sanitizer | 覆盖率、误差、越界和非法访问 |
| Compiler | LLVM lit/FileCheck、MLIR verifier、StableHLO/XLA 测试、Triton/torch.compile 测试 | 图覆盖、算子回退、数值和 codegen |
| Runtime / driver | CUPTI、nsys、CUDA/HIP API 测试、OpenCL/SYCL/Vulkan CTS | API 语义、错误码、并发和版本兼容 |
| Profiling | Nsight Systems、Nsight Compute、PyTorch Profiler、rocprof、XProf | kernel 时间、访存、占用率、通信等待 |
| 分布式 | nccl-tests、OSU Micro-Benchmarks、torchrun | all-reduce 带宽、扩展效率、通信瓶颈 |
| 推理服务 | vLLM benchmark、SGLang benchmark、genai-perf、perf_analyzer、k6/Locust | TTFT、TPOT、p99、并发上限 |
| 质量评测 | lm-evaluation-harness、OpenCompass | 量化或编译后的质量变化 |
| 稳定性 | DCGM、dcgm diag、gpu-burn、Prometheus/Grafana | 温度、功耗、ECC、XID、长稳趋势 |
| 数据与存储 | MLPerf Storage、fio、DALI | 数据加载是否拖慢训练 |

GPU、TPU、NPU 的工具名字不同，但分层是一样的：正确性、性能、稳定性、可运维性，一层都不能少。

## 交付前验收

交付前至少要过这几项：

- 正确性：客户模型和主力模型结果在容差内，训练能收敛，推理质量不下降。
- 性能：相对参考平台有明确的性能结论，微基准、模型级和端到端数据能互相对上。
- 扩展：单机到多机的 scaling efficiency 可解释，网络、存储和数据加载没有明显短板。
- 稳定：长时间运行、故障注入和恢复演练通过。
- 生态：客户常用的框架、库、模型和服务引擎能直接跑，不需要大规模改代码。
- 成本：性能不只看峰值，还要看每瓦性能、每美元性能和实际资源利用率。

最容易犯的错误，是拿一个漂亮的微基准当交付结论。客户买的不是 GEMM 峰值，而是自己的模型能不能按时训练完、推理服务能不能守住 SLO。

应用层验证的目标，是把“芯片能算”变成“客户的 workload 能稳定地跑完”。这两件事之间，隔着算子、模型、集群、服务和一整套软件栈。
