# Infra 八股

## 1. 大模型参数规模很大时，分布式训练有哪些常见策略？

**答案：** 常见策略包括数据并行 DP/DDP/FSDP/ZeRO、张量并行 TP、流水线并行 PP、专家并行 EP、序列并行、激活重计算、CPU/NVMe offload 和混合精度。实际训练往往组合使用：用 FSDP/ZeRO 降低参数、梯度、优化器状态显存，用 TP/PP 让单层或整模型放得下，用 activation checkpointing 换显存，用更好的通信拓扑减少跨节点开销。  
> **面试表达：** 分布式训练的核心是在显存、通信、吞吐和实现复杂度之间取平衡。  

**相关：** [[训练系统工程/FSDP ZeRO 和并行训练|FSDP ZeRO 和并行训练]]、[[训练系统工程/大模型资源估算|大模型资源估算]]

---

## 2. Tensor Parallel、Pipeline Parallel、Data Parallel 分别是什么？

**答案：** Data Parallel 是复制多份模型，每张卡处理不同 batch，再同步梯度；Tensor Parallel 是把单层矩阵乘或 attention/FFN 的张量切到多张卡上，让一层模型合起来计算；Pipeline Parallel 是按层切分模型，不同 GPU 负责不同层，通过 micro-batch 流水线提高利用率。DP 扩吞吐，TP 解决单层太大，PP 解决层数整体太深或整模型放不下。  
> **面试表达：** DP 切 batch，TP 切层内张量，PP 切层。  

**相关：** [[训练系统工程/FSDP ZeRO 和并行训练|FSDP ZeRO 和并行训练]]

---

## 3. TP 开大为什么可能导致训练变慢？TP 中通信开销来自哪里？

**答案：** TP 把单层计算切到多卡后，每层 attention/MLP 都需要 all-reduce、all-gather 或 reduce-scatter 来拼接中间结果。TP 越大，单卡计算变少，但同步通信更多，尤其跨节点 TP 会受网络带宽和延迟限制。TP 太大还会导致 kernel 变小、计算通信比下降，GPU 等通信，整体反而变慢。  
> **面试表达：** TP 不是越大越好，超过单机高速互联范围后，通信可能比省下的计算更贵。  

**相关：** [[训练系统工程/FSDP ZeRO 和并行训练|FSDP ZeRO 和并行训练]]、[[大模型推理/Ray|Ray]]

---

## 4. 训练时如何选择 TP / PP 等分布式策略？

**答案：** 先看单卡是否放得下模型层和激活：单层太大优先 TP，层数多或整模型太大再加 PP，参数/优化器状态太大用 FSDP/ZeRO。再看硬件拓扑：TP 尽量放在 NVLink/NVSwitch 内，PP 可以跨节点但要控制 pipeline bubble，DP/FSDP 更适合扩 batch。还要结合 sequence length、micro-batch、checkpoint、吞吐和容错成本做实验验证。  
> **面试表达：** 选择并行策略要先解决“放得下”，再优化“跑得快”。  

**相关：** [[训练系统工程/FSDP ZeRO 和并行训练|FSDP ZeRO 和并行训练]]、[[训练系统工程/大模型资源估算|大模型资源估算]]

---

## 5. RL 训练中 rollout 和 actor update 如何分布式调度？

**答案：** 一般把 rollout 和 update 拆成不同 worker。rollout worker 用 vLLM/SGLang/HF generate 批量生成样本；reward worker 做规则验证、RM 打分或环境执行；reference worker 计算 KL/logprob；trainer worker 用 FSDP/Megatron/DeepSpeed 更新 actor。调度上可以分离资源池，也可以 colocate 节省传输；每轮 update 后要把新权重同步给 rollout engine，避免样本过旧。  
> **面试表达：** RL 后训练不是一个普通 dataloader，而是“在线生成数据 + 分布式训练”的闭环系统。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]、[[大模型推理/vLLM SGLang|vLLM SGLang]]

---

## 6. veRL / Ray 这类框架大致如何组织分布式 RL？

**答案：** Ray 负责把不同角色封装成分布式 actor/task，并管理资源、placement group、对象传输和调度。veRL 这类 RLHF 框架会把 actor rollout、reference logprob、reward、advantage 计算、actor update、权重同步等步骤拆成可组合 worker，用统一 batch 数据结构传递 input_ids、mask、logprob、reward、advantage 等字段。  
> **面试表达：** Ray 管“分布式编排”，veRL 管“RL 数据流和训练角色”。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[大模型推理/Ray|Ray]]

---

## 7. 是否看过 veRL 的代码实现？核心模块有哪些？

**答案：** 面试中可以从模块职责回答：rollout 模块负责调用推理引擎生成 response；actor/trainer 模块负责计算 loss 和更新权重；reference/reward/critic 模块负责 logprob、奖励和 value 相关计算；数据协议模块承载 prompt、response、mask、logprob、reward、advantage；Ray orchestration 模块负责 worker 创建、资源放置、远程调用和权重同步。PPO 会多 critic，GRPO/DAPO 通常重点在 group rollout、reward 和 actor update。  
> **面试表达：** 不要只说“看过”，要能按 RL 数据流说出每个模块输入输出。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]

---

## 8. SGLang 和 vLLM 在推理服务中分别适合什么场景？

**答案：** vLLM 更像高吞吐通用 serving engine，核心优势是 PagedAttention、continuous batching、KV Cache 管理和 OpenAI-compatible 服务，适合大规模补全、对话和 rollout serving。SGLang 更强调结构化 LLM program、复杂 Agent/RAG、多轮调用、prefix/cache 复用和受约束生成，适合有复杂控制流和共享前缀的应用。二者能力边界在持续重叠，例如 vLLM 也支持结构化输出和 tool calling，SGLang 也可以作为高性能 serving backend；面试时更应该讲“默认侧重点”，不要讲成互斥。  
> **面试表达：** vLLM 偏“高吞吐模型服务底座”，SGLang 偏“复杂 LLM 程序执行 runtime”，但生产选型要看请求形态、缓存复用和工程生态。  

**相关：** [[大模型推理/vLLM SGLang|vLLM SGLang]]

---

## 9. LLM Generator 和 vLLM 请求模型各有什么优缺点？

**答案：** 如果在训练代码里直接用 HF generate 或自研 LLM Generator，通常集成简单，容易拿到 logits/logprob，和训练代码耦合更紧，适合小规模实验和调试；缺点是吞吐、连续批处理、KV 管理和多请求调度通常不如专门推理引擎。通过 vLLM/SGLang 这类请求式推理服务生成 rollout，吞吐高、并发强、KV 管理成熟，适合大规模 rollout；缺点是多一层服务/RPC 和权重同步复杂度，logprob、采样参数、版本一致性要仔细对齐。  
> **面试表达：** 小实验重调试便利，大规模 rollout 重吞吐和调度。  

**相关：** [[大模型推理/vLLM SGLang|vLLM SGLang]]、[[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]

---

## 10. 训推分离有什么收益？

**答案：** 训推分离可以让 rollout 使用推理优化后端，actor update 使用训练优化后端，二者独立扩缩容。这样能提高 GPU 利用率，减少训练 worker 等生成数据的时间，也能把推理服务的 KV Cache、continuous batching 和训练侧反向传播/FSDP 分片解耦。代价是权重同步、样本版本管理和系统复杂度上升。  
> **面试表达：** 训推分离的收益是让推理按推理方式跑、训练按训练方式跑。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[大模型推理/vLLM SGLang|vLLM SGLang]]

---

## 11. fully async 训练会带来什么问题？

**答案：** fully async 会提高吞吐，但会带来 policy lag、off-policy 数据、reward/logprob 版本不一致、训练不稳定、队列堆积、straggler、失败重试难追踪和指标延迟。算法上要限制样本最大滞后、重新计算 logprob 或降权旧样本；系统上要做 backpressure、版本追踪、超时控制和可观测性。  
> **面试表达：** 全异步不是免费加速，它把一部分等待时间变成了样本陈旧和调试复杂度。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 12. 沙箱执行速度如何优化？

**答案：** 可以预构建镜像和依赖，预热 sandbox pool，限制每次执行的 CPU/内存/时间，复用只读环境，缓存编译结果和测试数据，把轻量 verifier 本地化，批量调度同类任务，减少网络 IO，并对超时、失败和资源占用做细粒度监控。代码类任务还要隔离副作用，保证重试幂等。  
> **面试表达：** 沙箱优化的关键是预热、复用、限额、缓存和隔离。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]

---

## 13. 工具/API 并发、超时、失败重试如何影响 Agentic RL 训练效率？

**答案：** 并发太低会让 rollout 等工具，并发太高会触发限流或让尾延迟变差；超时设置太短会误杀可成功轨迹，太长会拖慢整批训练；失败重试能提高成功率，但也会增加成本、制造重复副作用和 reward 噪声。需要按工具设置并发池、超时预算、幂等 key、指数退避、最大重试次数和失败类型标签。  
> **面试表达：** 工具链路的 p95/p99 延迟和失败率会直接决定 Agentic RL 的有效吞吐。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]
