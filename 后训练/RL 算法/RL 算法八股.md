# RL 算法八股

## 0. 常见算法对比八股速查

这一部分优先回答“面试官让你比较两个算法”时最常见的问题。先抓住四个维度：**是否在线采样、是否需要 reward model、是否需要 critic、奖励/更新粒度是否匹配**。

### 0.1 PPO、DPO、GRPO、DAPO、Dr.GRPO、GSPO 总览

| 算法      | 核心思想                                           | 是否在线采样 | reward 来源                | 是否需要 critic | 主要优势                           | 主要代价               |
| ------- | ---------------------------------------------- | -----: | ------------------------ | ----------: | ------------------------------ | ------------------ |
| PPO     | 用新旧策略概率比裁剪限制更新幅度                               |      是 | RM、规则、环境奖励               |           是 | 通用、可优化任意 reward、能在线探索          | 多模型组件、显存高、超参敏感     |
| DPO     | 用偏好对直接优化 policy/reference log-ratio margin     |      否 | chosen/rejected 偏好对      |           否 | 简单稳定、成本低、像 SFT 一样训练            | 依赖离线偏好数据，探索弱       |
| GRPO    | 用同 prompt 多回答的组内相对 reward 替代 critic            |      是 | 规则奖励或 RM                 |           否 | 省 critic，适合数学/代码 RLVR          | 依赖 group 内差异，采样成本高 |
| DAPO    | 在 GRPO 上加入动态采样、非对称 clip、长度控制                   |      是 | 多为规则奖励                   |           否 | 更适合长 CoT 推理训练                  | 工程复杂，重采样增加成本       |
| Dr.GRPO | 修正 GRPO 的长度归一和 std 归一偏置                        |      是 | 多为规则奖励                   |           否 | 降低错误回答变长和难度偏置                  | 解决的是目标偏置，不解决所有采样问题 |
| GSPO    | 把 token-level ratio/clipping 改成 sequence-level |      是 | 多为 sequence-level reward |           否 | reward 粒度和 update 粒度更一致，MoE 更稳 | 算法较新，实现和适用边界要看任务   |

> **速记：** PPO 是经典在线 RLHF；DPO 是离线偏好优化；GRPO 是省 critic 的在线 RLVR；DAPO/Dr.GRPO/GSPO 都是在修 GRPO 在长推理里的不同问题。

### 0.2 GRPO 相比 PPO 有什么优缺点？

**答案：** GRPO 可以看作 PPO 在 LLM reasoning 场景下的轻量化变体。PPO 用 critic/value model 估计 advantage，而 GRPO 对同一个 prompt 采样多条回答，用组内平均 reward 作为 baseline，从而去掉 critic。  

| 维度 | PPO | GRPO |
|---|---|---|
| advantage 估计 | 训练 critic 估计 $V(s_t)$ | 组内相对 reward |
| 模型组件 | policy、old policy、reference、RM、critic | policy、old policy、reference、reward/verifier |
| 显存和工程成本 | 高 | 较低 |
| 采样方式 | prompt batch 生成 rollout | 每个 prompt 采样 $G$ 条回答 |
| 适合任务 | 开放式偏好、环境反馈、多步任务 | 数学、代码、格式验证等 RLVR |
| 主要风险 | critic 不稳、RM hacking、KL/clip 超参敏感 | 全对/全错 group 信号弱，依赖 group size 和 reward 质量 |

**GRPO 的优点：**

- 去掉 critic，显存、实现和训练稳定性压力更小。
- 对可验证任务很自然，同一题多条回答可以形成相对比较。
- 不一定需要人工偏好 RM，规则/verifier 就能给 reward。

**GRPO 的缺点：**

- 每个 prompt 要采多条回答，rollout token 成本不低。
- 如果 group 全对或全错，advantage 信号很弱。
- sequence-level reward 配 token-level ratio，长 CoT 或 MoE 场景可能有噪声。

> **面试表达：** GRPO 不是“比 PPO 全面更强”，而是在可验证推理任务里用组内比较替代 critic，换来更低显存和更简单训练；代价是更依赖采样组质量和 verifier 信号。

**相关：** [[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]、[[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 基础/Reward Return Value Advantage|Reward Return Value Advantage]]

---

### 0.3 GRPO 相比 DPO 有什么优缺点？

**答案：** GRPO 和 DPO 的核心差异是：GRPO 是在线采样的强化学习，DPO 是离线偏好对优化。GRPO 可以让模型在当前策略下探索新解法，并用规则/verifier 奖励更新；DPO 则直接利用已有 chosen/rejected 数据，不做 rollout。

| 维度 | DPO | GRPO |
|---|---|---|
| 数据来源 | 静态偏好对 | 当前策略在线生成 |
| 训练信号 | chosen 相对 rejected 更好 | 同 prompt 组内 reward 高低 |
| reward model | 不需要显式 RM | 可用规则奖励或 RM |
| critic | 不需要 | 不需要 |
| 探索能力 | 弱，受离线数据覆盖限制 | 较强，可持续采样新回答 |
| 工程成本 | 低，接近 SFT | 中等，需要 rollout/verifier/old logprob |
| 典型任务 | 通用偏好对齐、风格/安全/有用性偏好 | 数学、代码、工具调用、可验证推理 |

**GRPO 的优势：**

- 能优化规则奖励，不依赖人工偏好对。
- 能通过在线采样发现新的推理路径。
- 更适合“答案可自动判定”的 reasoning RL。

**GRPO 的劣势：**

- 比 DPO 更贵，需要生成多条 response 并计算 reward/logprob。
- reward 规则一旦有漏洞，容易 reward hacking。
- 训练稳定性受 KL、clip、group size、采样温度影响。

> **面试表达：** DPO 是“给定偏好数据后怎么学”，GRPO 是“让模型自己采样、再用 verifier 反馈怎么学”。有高质量偏好对时 DPO 更省；有可验证奖励和探索需求时 GRPO 更合适。

**相关：** [[后训练/RL 算法/DPO/DPO 算法原理|DPO 算法原理]]、[[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 基础/Rollout 和采样|Rollout 和采样]]

---

### 0.4 PPO 相比 DPO 有什么优缺点？

**答案：** PPO 是经典 RLHF 的在线 RL 算法，DPO 是把偏好建模和策略优化合并后的离线偏好优化算法。PPO 更通用，但工程更重；DPO 更简单稳定，但探索能力弱。

| 维度 | PPO | DPO |
|---|---|---|
| 优化方式 | 在线 rollout + reward 最大化 | 离线 chosen/rejected 分类式优化 |
| reward model | 通常需要单独训练 RM | 不需要显式 RM |
| critic/value | 需要 | 不需要 |
| reference model | 用 KL 约束 | 用 log-ratio margin 体现 reference |
| 能否探索 | 可以，取决于采样策略 | 基本不能，只学习数据里已有偏好 |
| 工程复杂度 | 高 | 低 |
| 失败模式 | reward hacking、KL 爆炸、critic 不稳 | 偏好数据噪声、风格过拟合、覆盖不足 |

> **面试表达：** PPO 更像完整 RL 系统，能优化任意 reward；DPO 更像偏好监督学习，工程简单但依赖数据覆盖。不是谁替代谁，而是目标和资源不同。

**相关：** [[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]、[[后训练/RL 算法/DPO/DPO 算法原理|DPO 算法原理]]、[[后训练/RL 基础/RLHF 总览|RLHF 总览]]

---

### 0.5 GRPO、DAPO、Dr.GRPO、GSPO 怎么区分？

**答案：** 这几类都围绕 GRPO 及其长推理训练问题展开，但改的不是同一个点。GRPO 是基线；DAPO 主要改采样和 clip；Dr.GRPO 主要改目标里的归一化偏置；GSPO 主要改概率比和裁剪粒度。

| 算法      | 要解决的问题                                   | 关键改动                                                           | 面试抓手           |
| ------- | ---------------------------------------- | -------------------------------------------------------------- | -------------- |
| GRPO    | PPO critic 成本高                           | 组内相对 reward 替代 value baseline                                  | 省 critic       |
| DAPO    | 长 CoT 中有效样本少、好样本被 clip、回答过长              | dynamic sampling、Clip-Higher、token-level loss、overlong shaping | 提高信号密度和长推理稳定性  |
| Dr.GRPO | 长度归一化和 std 归一化带来偏置                       | 去掉 per-response length normalization 和 group std normalization | 修正目标偏置         |
| GSPO    | sequence reward 搭配 token-level ratio 不匹配 | sequence-level length-normalized ratio 和 clipping              | 粒度对齐，MoE/长序列更稳 |

> **面试表达：** DAPO 偏工程训练策略，Dr.GRPO 偏目标函数无偏性，GSPO 偏 update 粒度一致性；它们不是简单替代关系，而是分别修 GRPO 的不同短板。

**相关：** [[后训练/RL 算法/GRPO/GRPO 算法原理|GRPO 算法原理]]、[[后训练/RL 算法/DAPO/DAPO 算法原理|DAPO 算法原理]]、[[后训练/RL 算法/Dr.GRPO/Dr.GRPO 算法原理|Dr.GRPO 算法原理]]、[[后训练/RL 算法/GSPO/GSPO 算法原理|GSPO 算法原理]]

---

### 0.6 面试里如何做算法选型？

**答案：** 先判断任务有没有高质量偏好对、有没有自动 verifier、是否需要在线探索、回答是否很长、模型是否 MoE 或训练/推理分离。选型可以按下面路线说：

| 场景 | 更优先考虑 | 理由 |
|---|---|---|
| 有大量高质量 chosen/rejected 偏好对 | DPO | 成本低、稳定、复现简单 |
| 开放式助手偏好，需要 RM 打分和在线探索 | PPO | 经典 RLHF 管线，reward 形式更通用 |
| 数学/代码/格式等可自动验证任务 | GRPO | 不需要 critic，规则 reward 直接可用 |
| 长 CoT reasoning，group 经常全对/全错 | DAPO | 动态采样提高有效训练信号 |
| 发现错误回答越训越长或 std 偏置明显 | Dr.GRPO | 修正长度归一和 group std 归一偏置 |
| MoE 或长序列下 token-level ratio 噪声大 | GSPO | sequence-level ratio 与 sequence reward 更匹配 |

> **面试表达：** 先看数据和奖励：偏好对充足选 DPO；可验证奖励选 GRPO；需要通用在线 RL 选 PPO。再看训练病灶：有效样本少看 DAPO，长度/std 偏置看 Dr.GRPO，token-level ratio 不稳看 GSPO。

---

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

---

## 11. PPO 里的 reward model 和 critic/value model 分别是怎么得到的？

**答案：** reward model 通常在 PPO 之前用人类偏好对单独训练：先用 SFT model 对同一个 prompt 采样多个回答，再标注 chosen/rejected，用 pairwise ranking loss 让 RM 学会给更优回答更高分。PPO 阶段 RM 一般冻结，只负责给完整 response 打 reward。critic/value model 则通常在 PPO 阶段训练：从 SFT/policy backbone 初始化，加 value head，用 rollout 得到的 return 作为监督信号拟合 $V(s_t)$，再用它计算 advantage、降低 policy gradient 方差。  
> **面试表达：** RM 是 PPO 前训练好的偏好打分器，critic 是 PPO 中训练出来的状态价值估计器；RM 评价完整回答，critic 估计当前前缀的平均回报基线。  

**相关：** [[后训练/RL 算法/PPO/PPO 算法原理|PPO 算法原理]]、[[后训练/RL 基础/RLHF 总览|RLHF 总览]]、[[后训练/RL 基础/Reward Return Value Advantage|Reward Return Value Advantage]]
