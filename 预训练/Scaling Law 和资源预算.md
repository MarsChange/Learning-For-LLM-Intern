# Scaling Law 和资源预算

## 面试定位

Scaling Law 解释模型规模、数据规模、计算量和 loss 之间的经验关系。它对实习面试的价值在于：你能说清“为什么不是参数越大越好”，以及训练/推理成本如何估算。

一句话概括：

> Scaling Law 关注在固定 compute budget 下如何分配参数量和训练 token；资源预算关注模型能不能训练、能不能推理、成本是否可接受。

## 三个核心变量

| 变量 | 含义 |
|---|---|
| `N` | 模型参数量 |
| `D` | 训练 token 数 |
| `C` | 训练计算量 |

粗略估算 Dense Transformer 训练 FLOPs：

$$
C \approx 6ND
$$

这是常用经验估算，真实训练还受 batch、sequence length、激活重算、并行通信影响。

## Kaplan vs Chinchilla

早期 scaling law 发现增大模型、数据、算力都能降低 loss。Chinchilla 进一步强调：很多大模型在给定 compute 下参数过多、训练 token 不足。

核心结论：

> Compute-optimal 训练需要同时增加模型参数和训练 token，而不是只堆参数。

直觉：

- 参数太大、数据太少：模型没吃够数据，欠训练。
- 参数太小、数据太多：模型容量不足。
- 二者平衡才能在固定算力下更优。

## 资源预算表

| 资源 | 主要受什么影响 | 常见优化 |
|---|---|---|
| 权重显存 | 参数量、精度 | 量化、张量并行 |
| 梯度显存 | 可训练参数量 | ZeRO/FSDP、LoRA |
| 优化器状态 | 参数量、优化器 | ZeRO、8-bit optimizer |
| 激活显存 | batch、seq_len、层数 | activation checkpointing |
| KV Cache | 并发、seq_len、KV heads | GQA/MQA/MLA、PagedAttention |
| 训练 FLOPs | 参数量、token 数 | 数据/模型规模规划 |

## 推理成本粗估

推理分 prefill 和 decode：

- Prefill：处理 prompt，全序列 attention，长上下文贵。
- Decode：每次生成一个 token，受 KV Cache 读写和 batch 调度影响大。

权重显存：

$$
\text{weight memory} \approx N \times \text{bytes per param}
$$

KV Cache 显存：

$$
2 \times L \times B \times T \times H_{kv} \times d_{head} \times \text{bytes}
$$

这解释了为什么长上下文服务往往不是权重显存先爆，而是 KV Cache 先吃满。

## Scaling Law 对应用算法的意义

- 小模型微调时，不要期待 SFT 弥补所有基座能力缺口。
- RAG 更适合补事实知识，预训练/继续预训练更适合补分布能力。
- 数据质量、任务覆盖和评估闭环比单纯增加 LoRA rank 更重要。
- 推理成本决定线上模型选型，不能只看 benchmark。

## 面试高频问题

1. **为什么不是模型越大越好？**  
   固定算力下，参数和数据要平衡；参数过大但训练 token 不足会欠训练。

2. **训练 FLOPs 粗略怎么估？**  
   Dense Transformer 常用 `6 * 参数量 * 训练 token 数` 作为粗估。

3. **为什么长上下文服务成本高？**  
   prefill attention 计算贵，decode 还要维护大量 KV Cache，影响并发。

4. **Scaling Law 和微调有什么关系？**  
   微调主要改变行为和任务适配，不能凭空创造基座模型没有学到的基础能力。

## 参考资料

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
