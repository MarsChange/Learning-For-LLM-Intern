# Agentic RL 八股

## 1. 给定一个 query，Agentic RL 的完整训练流程是什么？

**答案：** 先把 query 放入任务环境，policy 按 agent loop 生成思考、工具调用或最终答案；运行时执行工具并把 observation 拼回上下文；多轮循环直到成功、失败或达到预算；收集完整 trajectory，包括 token、action、observation、tool result、reward、mask 和版本信息；再用最终奖励或过程奖励计算 advantage，屏蔽非模型生成 token，对 policy 生成的有效 token 做 PPO/GRPO/DAPO 类更新；最后用任务成功率、工具调用正确率、成本和安全指标评估。  
> **面试表达：** Agentic RL 比普通 RL 多了环境交互和轨迹管理，本质是“rollout 一条工具使用轨迹，再把轨迹转成可训练 token loss”。  

**相关：** [[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]、[[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[输入表示/Chat Template 和 Tool Calling|Chat Template 和 Tool Calling]]

---

## 2. Agent loop 如何运行：模型输出、工具调用、工具返回、上下文拼接、继续生成分别怎么衔接？

**答案：** 运行时维护一个 messages/state 列表。模型看到 system、user、历史 assistant、tool observation 和工具 schema 后，输出普通文本或结构化 action；如果是 action，runtime 校验参数、执行工具、拿到 observation；observation 以 tool message 或约定格式追加到上下文；模型再基于新上下文继续生成，直到输出 final answer 或触发停止条件。  
> **面试表达：** Agent loop 是一个状态机：model 产 action，runtime 执行 action，observation 回填状态，再交给 model 决策下一步。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[输入表示/Chat Template 和 Tool Calling|Chat Template 和 Tool Calling]]

---

## 3. Agent 训练中哪些 token 参与 loss，哪些 token 需要 mask 掉？

**答案：** 一般只让 policy 自己生成的 assistant token 参与 loss，比如工具调用 JSON、行动参数、可训练的推理文本和最终答案。user/system token、tool schema、tool observation、外部环境返回、padding、被截断的无效部分都要 mask 掉，因为它们不是模型动作。如果不希望模型学习暴露私有 CoT，也可以只训练 action/final answer，或把过程推理替换为可控的中间状态标签。  
> **面试表达：** loss 只应该惩罚或奖励“模型能控制的动作”，不能让模型去拟合用户输入和工具返回。  

**相关：** [[输入表示/Chat Template 和 Tool Calling|Chat Template 和 Tool Calling]]、[[后训练/SFT/LoRA 和 QLoRA|LoRA 和 QLoRA]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 4. 多轮 Agentic RL 里，什么时候停止 rollout？

**答案：** 停止条件通常包括：模型输出 final answer，任务环境判断成功，工具或 verifier 返回不可恢复失败，达到最大 turn 数、最大 token 数、最大耗时或最大工具调用次数，检测到重复 action/observation 死循环，或触发安全策略。停止后要给 trajectory 标注终止原因，方便 reward、错误分析和训练过滤。  
> **面试表达：** Agent rollout 必须有明确预算和终止原因，否则容易把训练成本耗在无效长轨迹上。  

**相关：** [[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]、[[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]

---

## 5. 模型陷入死循环或 turn 数远超预期时，如何处理？

**答案：** 在线 rollout 时应立即触发 stop condition，给失败或长度惩罚，并记录重复模式。训练数据处理时可以过滤明显无效轨迹，或把“循环检测后停止”的负样本保留下来让模型学会结束。系统侧可以加入重复 action 检测、相同 observation 检测、工具调用预算、状态摘要和强制 final answer 提示。  
> **面试表达：** 死循环不能只靠模型自觉，要在环境、奖励和训练数据里一起约束。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 6. 多轮 Agentic RL 是否遇到过推理耗时和长尾轨迹？有哪些优化方法？

**答案：** 很容易遇到。长尾来自长 CoT、工具慢、失败重试、循环和复杂任务。优化方法包括限制 max turns/max tokens，设计长度或成本惩罚，给工具设置 timeout 和重试上限，对 observation 做结构化摘要，缓存可复用工具结果，使用 process reward 鼓励更短路径，按任务难度 curriculum，训练时单独监控 p95/p99 轨迹长度和耗时。  
> **面试表达：** Agentic RL 优化的不只是准确率，还要优化“成功需要花多少 token、多少工具调用、多少时间”。  

**相关：** [[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]、[[后训练/RL 算法/DAPO/DAPO 算法原理|DAPO 算法原理]]、[[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]

---

## 7. 如果某些轨迹 rollout 特别长，应该丢弃、截断、异步处理，还是复用？各有什么代价？

**答案：** 丢弃最干净，但会浪费难样本并让训练偏向短任务；截断能控成本，但可能把失败归因搞错，产生有偏 reward；异步处理能提升 GPU 利用率，但会带来 policy lag 和样本陈旧；复用长轨迹能摊薄成本，但多次更新会越来越 off-policy。实际常按终止原因分层处理：明显循环丢弃，接近成功保留，超时轨迹给惩罚，复用时限制版本滞后和更新轮数。  
> **面试表达：** 长轨迹不是一刀切丢掉，关键看它有没有学习信号，以及复用会不会破坏 on-policy 假设。  

**相关：** [[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]、[[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]

---

## 8. rollout 很贵时，能否一批 rollout 数据多次更新？这会带来什么 off-policy 问题？

**答案：** 可以有限次数复用，这也是 PPO-style 方法会做多 epoch minibatch update 的原因。但复用越多，当前 policy 与生成样本的 old policy 差距越大，importance ratio 方差会变大，clip fraction 和 KL 可能上升，最终样本不再代表当前策略分布。解决办法是限制 update epoch、用 ratio clipping/KL、监控 policy lag、及时刷新 rollout。  
> **面试表达：** rollout 复用是在“省采样成本”和“保持 on-policy”之间做权衡。  

**相关：** [[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 9. 训练和推理分离、异步 rollout 会带来什么收益和风险？

**答案：** 收益是推理和训练可以用不同后端、不同并行策略和不同资源池，rollout worker 不必等待 trainer，GPU 利用率和吞吐更高。风险是 rollout 使用的权重可能落后于 trainer，reward、logprob、KL 和当前 policy 不再完全匹配，产生 policy lag；系统还会更难调试，失败重试和数据顺序也更复杂。  
> **面试表达：** 异步能提高吞吐，但代价是样本新鲜度下降，算法上要补偿 policy lag。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 10. 异步 Agentic RL 中 policy lag / off-policy 问题如何补偿？

**答案：** 常见补偿包括给每条 trajectory 记录 policy version，限制最大版本滞后；训练前重新计算当前 policy logprob；使用 old logprob 做 importance ratio 和 clipping；加 KL/reference penalty；限制每批数据复用次数；对过旧样本降权或丢弃；周期性同步 rollout engine 权重；必要时设置同步 barrier。  
> **面试表达：** 不能只说“异步更快”，还要能说明如何知道样本旧了、旧到什么程度就不能用。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 11. Agentic RL 中影响训练效率的因素有哪些？

**答案：** 主要因素包括 rollout 生成速度、工具/API 延迟、reward/verifier 速度、轨迹长度、group size、采样温度、并发调度、权重同步成本、GPU 利用率、无效样本比例、重试比例和数据传输开销。算法上要看有效 reward 差异和 advantage 方差；系统上要看 trainer 是否等 rollout、rollout 是否等工具、reward worker 是否成为瓶颈。  
> **面试表达：** Agentic RL 的效率瓶颈经常不在反向传播，而在 rollout、工具和 reward 链路。  

**相关：** [[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]、[[大模型推理/vLLM SGLang|vLLM SGLang]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 12. 工具调用、API 请求、沙箱执行成为瓶颈时，可以怎么优化？

**答案：** 可以做请求缓存和去重，批量调用 verifier，设置超时和幂等重试，预热沙箱和依赖环境，复用 sandbox pool，把慢工具拆成异步任务，对高频工具做本地 mock 或规则 verifier，限制无效重试，按资源类型隔离队列，并把 tool latency、error rate、timeout rate 纳入训练监控。  
> **面试表达：** 工具瓶颈要按生产系统处理：缓存、并发、限流、超时、重试、隔离和可观测性都要有。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[训练系统工程/veRL Ray 后训练系统|veRL Ray 后训练系统]]

---

## 13. 为什么多轮 Agentic RL 会有 credit assignment 问题？

**答案：** 因为最终成功或失败往往由多步推理、多个工具调用和多个 observation 共同决定，只有 final reward 时，很难判断是哪一步 action 真正贡献了结果。把同一个 reward 分给所有 token 会让无关 token 也被强化或惩罚，导致学习信号噪声很大。多轮越长、工具越多、分支越复杂，credit assignment 越难。  
> **面试表达：** Agentic RL 的难点是奖励延迟：结果在最后出现，但错误可能发生在中间某一步。  

**相关：** [[后训练/RL 基础/Reward Return Value Advantage|Reward Return Value Advantage]]、[[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]

---

## 14. process reward 的优势是什么？可以从哪些维度设计？

**答案：** process reward 能给中间步骤提供更密集的训练信号，减少只靠 final reward 的高方差，并帮助模型学会正确的工具选择、参数构造、状态更新和错误恢复。设计维度可以包括子目标完成度、工具选择正确性、参数合法性、observation 使用是否正确、推理步骤正确性、安全合规、成本/长度、是否避免重复调用，以及最终答案是否可追溯到工具结果。  
> **面试表达：** final reward 评价“最后对不对”，process reward 评价“过程有没有走在正确路线上”。  

**相关：** [[后训练/RL 基础/Reward Return Value Advantage|Reward Return Value Advantage]]、[[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]
