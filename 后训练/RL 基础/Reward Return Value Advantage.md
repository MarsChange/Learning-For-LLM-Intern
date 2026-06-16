# Reward Return Value Advantage

## 面试定位

理解 PPO/GRPO/DAPO 前，必须分清 reward、return、value、advantage。很多面试追问会卡在这些概念上。

一句话概括：

> Reward 是环境给当前轨迹/动作的反馈，Return 是未来累计奖励，Value 是状态平均能拿多少回报，Advantage 是当前动作比平均动作好多少。

## 四个概念

| 概念 | 符号 | 含义 |
|---|---|---|
| Reward | `r_t` | 当前时间步获得的奖励 |
| Return | `G_t` | 从当前时刻开始的累计折扣奖励 |
| Value | `V(s_t)` | 状态 `s_t` 下预期 return |
| Advantage | `A(s_t,a_t)` | 动作 `a_t` 比平均策略好多少 |

## Return

折扣累计回报：

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

其中 `γ` 是 discount factor。

LLM 后训练里常见情况：

- 中间 token 没有奖励。
- 完整回答结束后得到一个 sequence-level reward。
- 需要把这个奖励分配给 token-level policy gradient。

## Value

状态价值：

$$
V^\pi(s)=\mathbb{E}_{\pi}[G_t|s_t=s]
$$

直觉：在这个状态下，按当前策略继续生成，平均能拿多少回报。

PPO 里 critic/value model 的作用就是估计 `V(s)`，降低 policy gradient 方差。

## Advantage

动作价值：

$$
Q^\pi(s,a)=\mathbb{E}_{\pi}[G_t|s_t=s,a_t=a]
$$

Advantage：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

直觉：

- `A > 0`：这个动作比平均水平好，应该提高概率。
- `A < 0`：这个动作比平均水平差，应该降低概率。

## LLM 中的对应关系

| RL | LLM |
|---|---|
| state | prompt + 已生成 token |
| action | 下一个 token / 工具动作 |
| trajectory | 完整 response / Agent 轨迹 |
| reward | RM 分数、规则验证、人工偏好 |
| value | 当前前缀继续生成的预期 reward |
| advantage | 某 token/动作相对平均策略的好坏 |

## GAE

PPO 常用 GAE（Generalized Advantage Estimation）：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
A_t^{GAE}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

`λ` 控制 bias-variance tradeoff：

- `λ` 大：更接近 Monte Carlo return，方差大、偏差小。
- `λ` 小：更依赖 value bootstrap，方差小、偏差大。

## GRPO 如何不用 Value

GRPO 不训练 value model，而是对同一个 prompt 采样多条回答：

$$
\hat{A}_i=\frac{r_i-\text{mean}(r_1,\ldots,r_G)}{\text{std}(r_1,\ldots,r_G)}
$$

这里组内均值相当于 baseline。

直觉：

- 同一题的多条回答难度相同。
- 比组内平均更好的回答 advantage 为正。
- 比组内平均更差的回答 advantage 为负。

## 面试高频问题

1. **Reward 和 Return 区别？**  
   Reward 是单步反馈，Return 是未来累计奖励。

2. **Value 和 Advantage 区别？**  
   Value 评估状态平均好坏，Advantage 评估某动作相对平均策略的增益。

3. **为什么 policy gradient 要用 advantage？**  
   它降低方差，并告诉策略应该提高还是降低某动作概率。

4. **GRPO 为什么可以去掉 critic？**  
   它用同一 prompt 下多个回答的组内平均 reward 作为 baseline。

## 参考资料

- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
