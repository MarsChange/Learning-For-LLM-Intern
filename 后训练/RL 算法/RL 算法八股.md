# RL 算法八股

## 1. RLHF 的三个主要阶段是什么？

**答案：** 经典 RLHF 包括三步：先做 SFT，让模型学会基本指令跟随和回答格式；再用 chosen/rejected 偏好对训练 reward model；最后用 PPO 等 RL 算法优化 policy，让模型在 reward model 和 KL 约束下生成更符合偏好的回答。现代变体中，DPO 省掉显式 reward model 和在线 RL，GRPO/DAPO 常用规则奖励替代 reward model。  
> **面试表达：** SFT 负责冷启动，RM 负责把偏好变成可优化信号，RL 负责真正更新策略。  

**相关：** [[后训练/RL 基础/RLHF 总览|RLHF 总览]]、[[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]

---

## 2. PPO、GRPO、DPO、DAPO 的基本原理是什么？

**答案：** PPO 用 old policy 采样，通过新旧策略概率比裁剪限制更新幅度，并用 critic 估计 advantage。GRPO 去掉 critic，对同一 prompt 采样多条回答，用组内相对 reward 估计 advantage。DPO 直接用偏好对优化 policy/reference 的 log-ratio margin，不做在线 rollout。DAPO 是 GRPO 思路上的大规模 reasoning RL 改造，引入动态采样、上下界解耦裁剪、token-level loss 和过长惩罚。  
> **面试表达：** PPO 是带 critic 的在线 RL，GRPO 是无 critic 的组内相对优化，DPO 是离线偏好分类式优化，DAPO 是面向长推理的 GRPO 工程增强。  

**相关：** [[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]、[[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 算法/DPO/DPO 算法原理|DPO 算法原理]]、[[后训练/RL 算法/DAPO/DAPO 算法原理|DAPO 算法原理]]

---

## 3. GRPO 相比 PPO 做了哪些变化？DAPO 相比 GRPO 的核心改进是什么？

**答案：** GRPO 相比 PPO 最大变化是去掉 value/critic，不再估计每个状态的 value，而是对同一 prompt 采样一组回答，用组内 reward 均值作为 baseline。DAPO 在 GRPO 上解决长推理训练的工程问题：dynamic sampling 保证 group 里有正负差异，Clip-Higher 给正 advantage 更大更新空间，token-level loss 避免长度归一偏置，overlong reward shaping 控制过长回答。  
> **面试表达：** GRPO 是“省 critic”，DAPO 是“让 GRPO 在大规模长推理上更有效、更稳”。  

**相关：** [[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 算法/DAPO/DAPO 算法原理|DAPO 算法原理]]、[[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]

---

## 4. DAPO 的动态采样为什么能提升训练效率？

**答案：** GRPO/DAPO 的学习信号来自同一 prompt 下回答之间的 reward 差异。如果一组全对或全错，advantage 接近无效，训练几乎没有方向。动态采样会优先保留既有正确又有错误的 group，把训练预算集中在“模型会一点但不稳定”的题目上，从而提高有效样本密度和梯度信号质量。  
> **面试表达：** 动态采样不是为了拿更多数据，而是为了让每个 group 都有可学习的对比信号。  

**相关：** [[后训练/RL 算法/DAPO/DAPO 算法原理|DAPO 算法原理]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

## 5. DAPO 过滤无区分度 group 后，会不会限制模型学习困难样本？有什么解决思路？

**答案：** 会有这个风险，但要说准确：DAPO 的 dynamic sampling 主要过滤的是同一 prompt 下全对或全错、组内 reward 没有差异的 group，而不是简单删除所有零奖励样本。如果全错 group 被大量重采样或丢弃，模型可能接触不到真正困难的题，训练会偏向中等难度样本。解决思路包括设置最大重采样次数后保留部分难样本、按难度 curriculum 混入 hard prompts、引入 process reward 提供中间信号、保留少量全错 group 做负例、或者用更强 verifier/RM 区分“完全无效”和“过程接近正确”的样本。  
> **面试表达：** DAPO 动态采样提升信号密度，但要防止训练集只剩“半会不会”的题；它过滤的是无对比信号的 group，不是机械删除所有错误样本。  

**相关：** [[后训练/RL 算法/DAPO/DAPO 算法原理|DAPO 算法原理]]、[[后训练/RL 基础/Reward Return Value Advantage|Reward Return Value Advantage]]

---

## 6. GRPO / DAPO 是 on-policy 还是 off-policy 思路？

**答案：** 它们通常是 near on-policy：用当前或刚冻结的 old policy 生成 rollout，再用这些 rollout 做有限轮 policy update。严格说，更新时 policy 已经变了，所以需要 old logprob、ratio clipping 和 KL 来控制偏移。如果 rollout 被反复复用、异步队列太长或权重同步滞后，就会越来越 off-policy，importance ratio 方差和训练不稳定性都会上升。  
> **面试表达：** GRPO/DAPO 的设计假设是“样本别太旧”，复用过度就变成 off-policy 问题。  

**相关：** [[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]、[[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 算法/DAPO/DAPO 算法原理|DAPO 算法原理]]

---

## 7. DPO 和 PPO 的区别、优势和劣势分别是什么？

**答案：** DPO 用离线偏好对训练，不需要 reward model、critic 和在线 rollout，工程简单、稳定、成本低；但它依赖偏好数据质量，探索能力弱，不适合需要环境交互或规则验证的复杂任务。PPO 是在线 RL，可以优化任意 reward，能处理环境反馈和探索，但工程成本高、超参敏感，需要维护 policy、old policy、reference、reward model、critic 等组件。  
> **面试表达：** 有高质量偏好对、目标偏静态时优先 DPO；目标需要在线探索或工具环境反馈时，PPO/GRPO 更合适。  

**相关：** [[后训练/RL 算法/DPO/DPO 算法原理|DPO 算法原理]]、[[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]、[[后训练/RL 基础/RLHF 总览|RLHF 总览]]

---

## 8. GRPO 中 old policy、reference model、policy model 分别起什么作用？

**答案：** policy model 是正在训练的模型，负责生成当前 logprob 并被梯度更新；old policy 是采样 rollout 时的策略快照，用来计算新旧概率比，保证 PPO-style clipping 有参照；reference model 通常是冻结的 SFT 或 base 模型，用 KL/reference penalty 约束 policy 不要为了 reward 偏离语言质量、安全边界和原始能力太远。  
> **面试表达：** policy 负责学，old policy 负责量更新步子，reference 负责守住分布边界。  

**相关：** [[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]

---

## 9. GRPO 中 KL 或 reference penalty 的作用是什么？

**答案：** KL/reference penalty 是防止 policy 只追 reward 而发生分布漂移。没有约束时，模型可能 reward hacking、输出风格突变、语言质量下降、长度失控或安全边界变差。KL 约束让模型在提升 reward 的同时，仍尽量靠近 reference model 的合理输出分布。  
> **面试表达：** reward 告诉模型往哪里走，KL 告诉模型不要走太远。  

**相关：** [[后训练/RL 基础/RLHF 总览|RLHF 总览]]、[[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]

---

## 10. 如果给一个真实 query，让你代码实现 GRPO 全流程，你会如何组织 rollout、reward、logprob、advantage、loss 和 update？

**答案：** 一个清晰的实现流程是：对 prompt batch 中每个 query 用 old policy 采样 G 条 response；用规则、RM 或环境执行得到 reward；保存 input_ids、attention_mask、response_mask、old logprobs、reward 和必要的 ref logprobs；对同一 query 的 G 个 reward 做组内均值或标准化，得到每条 response 的 advantage；把 sequence-level advantage broadcast 到有效 response token；当前 policy 重新前向得到 new logprobs，计算 ratio、clipped loss、KL/reference penalty；最后反向更新 policy，并周期性刷新 old policy 或同步 rollout engine 权重。  
> **面试表达：** GRPO 代码的核心数据结构是一批 group，每个 group 里有多条 response、reward、old logprob、ref logprob 和 mask。  

**相关：** [[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]、[[后训练/RL 基础/Reward Return Value Advantage|Reward Return Value Advantage]]
