---
title: "大模型 API trace 的 capture 与 replay"
date: "2026-11-22"
description: "从 HTTP 请求、token 序列到 CUDA API/kernel 和 CUDA Graph，解释 trace 抓取与重放的原理、做法和边界。"
tags:
  - "LLM"
  - "CUDA"
  - "Trace"
  - "性能"
draft: false
slug: "llm-api-trace-capture-replay"
---

## 先分清四种 trace

“抓一个大模型的 API trace”这句话，至少可能指四种不同的东西：

![大模型 trace 的四个层次](/images/llm-trace-layers.svg)

| 层次 | 抓什么 | 主要用途 |
| --- | --- | --- |
| HTTP / JSON API | 请求、响应、时间戳、工具调用 | 回放用户请求、做回归和压测 |
| Token trace | prompt、输出 token、logprob | 比较不同推理配置的数值结果 |
| CUDA API / kernel trace | runtime/driver 调用、kernel、memcpy、stream | 分析 GPU 时间线、定位性能问题 |
| CUDA Graph | 被捕获的 GPU 计算图 | 重复 launch，降低 CPU 启动开销 |

抓错层，问题就答不对。想看用户请求怎么变慢，抓 HTTP trace；想看 kernel 为什么空转，抓 CUDA trace；想减少 decode 的 launch 开销，用 CUDA Graph。

这里的“API”有两层含义。大模型服务对外通常暴露 HTTP/JSON 接口；一次请求从 HTTP 进入，经过 tokenizer、调度器、推理引擎，最后落到 CUDA kernel。HTTP trace 记录入口，CUDA trace 记录设备侧执行，两边可以用同一个 request ID 或 trace ID 关联。

## 基础概念

**trace** 是一次执行留下的事件记录：什么时间、调用了什么、参数是什么、执行了多久。

**capture** 是抓取过程。可以在 HTTP 代理层抓，在 SDK 里埋点抓，也可以在 CUDA runtime/driver 层抓。

**replay** 是把记录下来的事件重新发一遍。不同层的 replay 目标不同：HTTP 重放是重新发请求，token 重放是强制模型走同一条输出序列，CUDA Graph 重放是重新执行同一张 GPU 计算图。

几个 GPU 侧的词：

- **CUDA runtime / driver API**：应用调用 CUDA 的接口，如分配显存、启动 kernel、拷贝数据。
- **kernel**：真正跑在 GPU 上的函数。
- **stream**：一串按顺序执行的 GPU 操作。
- **CUDA Graph**：把 kernel、memcpy、事件等操作连成一张有依赖的图，定义一次，多次执行。

## 抓取 HTTP trace

这一层最像普通后端监控。

在应用和模型服务之间放一个代理，或者在 SDK 里加 middleware，记录：

- 请求方法、路径、模型名
- prompt、messages、工具定义
- 采样参数：temperature、top_p、max_tokens
- 响应、token 用量、首 token 时间、总耗时
- 调用链和错误信息

常见做法是用 `mitmproxy` 做 HTTP 代理，用 OpenTelemetry 或 Langtrace/Langfuse 这类工具做应用层 tracing。x.com 上 [Langtrace](https://x.com/langtrace_ai) 的定位就是记录每次 LLM API 调用、成本和性能指标。

API trace 里可能有密钥、用户隐私和完整 prompt。抓之前要先脱敏，尤其是 `Authorization` 头和业务数据。

## 抓取 token trace

vLLM 提供了另一种更贴近模型的 [trace replay](https://docs.vllm.ai/en/latest/serving/online_serving/trace_replay/)：抓一段之前生成的 token 序列，让推理引擎按这个序列输出，同时计算模型真实的 logprob。

开启方式：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enable_trace_replay=True)

params = SamplingParams(
    trace_decode_token_ids=[15, 284, 1026, 374],
    logprobs=5,
)
```

它会把输出固定成 `[15, 284, 1026, 374]`，但每个位置仍然返回模型当前配置下的真实概率。用途不是压测，而是比较量化、TP 布局、attention backend，或者训练框架和推理框架之间的数值差异。

这属于模型输出层的 trace，不是 CUDA trace，也不是 HTTP trace。

## 抓取 CUDA API / kernel trace

NVIDIA 的 [CUPTI](https://docs.nvidia.com/cupti/) 是专门做这件事的库。它通过 Activity API 异步记录 CUDA 活动，通过 Callback API 在特定 CUDA 事件发生时通知工具。

CUPTI 能记录：

- CUDA runtime 和 driver API 调用
- kernel 启动和执行时间
- memcpy、memset
- stream、event 和同步关系

[Nsight Systems](https://developer.nvidia.com/nsight-systems) 就是建立在 CUPTI 之上的工具。一个常见命令是：

```bash
nsys profile --trace=cuda,nvtx -o trace ./app
```

抓到的 trace 可以看时间线：哪个 kernel 最慢、CPU 和 GPU 之间的空隙在哪里、memcpy 是否拖住了流水线。

要注意：CUPTI trace 主要用于**分析**，不是一个通用的 replay 文件。它记录的是发生过什么，不会自动帮你把整段 CUDA 调用重新执行一遍。要 replay，通常需要自己写回放器，或者改用 CUDA Graph。

## CUDA Graph：真正可重放的 capture

[CUDA Graph](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) 把“定义”和“执行”分开：

1. 定义图：kernel、memcpy、依赖关系。
2. instantiate：生成可执行图。
3. launch：反复执行。

抓取方式之一是 stream capture：

```cuda
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

launch_work(stream);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(exec, stream);
```

关键点是：**capture 期间的操作不会真的执行**，而是被记录到图里。捕获结束后，`cudaGraphLaunch()` 才真正运行。

PyTorch 里也能用：

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = model(x)
g.replay()
```

vLLM 在推理侧大量使用 CUDA Graph（[设计说明](https://docs.vllm.ai/en/stable/design/cuda_graphs/)），通过 `cudagraph_mode` 在 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE` 之间选择。decode 阶段的 batch 形状相对稳定，适合把一整串 kernel 捕获成图，之后每一步直接 replay，减少 CPU 启动开销。

但 CUDA Graph 不是万能的：

- 捕获期间不能随意同步或查询 stream。
- 形状和内存地址要稳定，动态 shape 往往需要 padding 或重新捕获。
- 不同 stream 的依赖要正确连接。
- 图会占额外显存，模式选错可能变慢。
- 它重放的是 GPU 计算图，不包含 Python 逻辑、HTTP 请求和业务状态。

Reddit 的 r/CUDA 有专门的[帖子](https://www.reddit.com/r/CUDA/comments/1o2fl3g/cuda_graphs_vs_kernel_fusion_are_we_solving_the/)讨论 CUDA Graphs 和 kernel fusion 的边界。两者解决的问题不同：fusion 减少 kernel 数量，Graph 减少 launch 开销，不能互相替代。

## 怎么重放

不同 trace 的 replay 方式不一样。

### HTTP replay

两种模式：

- **缓存回放**：把响应缓存下来，重放时直接返回，适合做确定性回归。
- **在线回放**：把同样的请求重新发给新模型或新服务，适合做模型升级对比和压测。

在线回放要保留请求的到达节奏和并发数，否则测出来的吞吐和延迟没有意义。工具上可以用 `mitmproxy`、VCR.py、Polly.js、WireMock，或者自建 replay client。

### Token replay

把 trace 里的输出 token 作为强制输入，比较不同配置下的 logprob、top-k 和最终质量。它适合验证数值一致性，不适合评估生成速度。

### CUDA replay

通用的 CUPTI trace 不能直接拿来重放。工程上有两条路：

- 用 CUDA Graph 重新捕获可重放的 GPU 工作。
- 自己写一层 replay driver，把记录到的 API 调用按顺序重新发出。

后者更接近“API replay”，但要处理句柄、显存地址、错误码、同步和版本差异，维护成本高。

## 一个可落地的流程

1. 先定义问题：要查性能、数值差异、用户请求回归，还是故障复现？
2. 选择层次：HTTP、token、CUDA trace，还是 CUDA Graph。
3. 预热后抓取，记录模型、驱动、框架、batch、精度和硬件配置。
4. 先在同一环境 replay 一次，确认输出、kernel 数量和耗时能对上。
5. 再换目标环境 replay，比较差异。
6. 保存 trace 的元数据和脱敏后的样本，别只保存一个二进制文件。

比较指标也要分层：

- HTTP：首 token 延迟、总延迟、tokens/s、错误率、并发。
- Token：logprob 差异、top-k 变化、评测分数。
- CUDA：kernel 时间、GPU 空闲、memcpy、occupancy。
- CUDA Graph：launch 开销、显存占用、decode tokens/s。

## 边界和常见误区

**trace 不是免费的。** CUPTI 和代理都会增加开销，抓得太细会改变程序行为。用采样、过滤器和 warmup 控制影响。

**trace 可能被篡改。** x.com 上有一条讨论提到，LLM agent 可以删除或修改自己的 trace，甚至学会在有利于奖励时隐藏行为（[原帖](https://x.com/askalphaxiv/status/2104187549250207863)）。如果 trace 用于审计，采集器要独立于 agent，存储要 append-only。

**trace 可能包含敏感数据。** prompt、工具返回值、API key 都可能出现在 trace 里。采集前脱敏，采集后控制访问权限。

**CUDA Graph 不是 kernel fusion。** 它优化的是提交方式，不会自动把两个 kernel 合成一个。

**不同层的 trace 不能混用。** HTTP trace 解释不了 kernel 为什么慢；CUDA trace 也不知道用户请求的业务语义。

## 工具地图

| 层次 | 抓取 | 重放 |
| --- | --- | --- |
| HTTP / JSON | mitmproxy、OpenTelemetry、Langtrace、Langfuse | 缓存回放、在线 replay client、WireMock |
| Token | vLLM `enable_trace_replay` | 强制 token，比较 logprob |
| CUDA API / kernel | CUPTI、Nsight Systems | 分析为主；自定义 replay driver |
| CUDA Graph | stream capture、PyTorch CUDAGraph、vLLM `cudagraph_mode` | `cudaGraphLaunch()` / `g.replay()` |

[CUDA 官方文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)把 CUDA Graph 的收益讲得很直接：图的定义、实例化和执行分离后，重复 launch 的 CPU 开销可以大幅降低。CUPTI 则负责把实际执行记录下来。两者一个偏“重放”，一个偏“观察”，经常配合使用。

先确定你要回答哪一层的问题，再选 capture 和 replay 的方式。把所有东西都叫“API trace”，很容易抓了一堆数据，却没用在对的地方。
