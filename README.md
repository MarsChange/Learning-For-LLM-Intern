# Learning-For-LLM-Intern

面向大模型算法、后训练、推理系统和 Agent 实习面试的个人学习笔记库。这个仓库不是论文资料合集，而是一套可以反复复习、面试复述、手写验证和继续扩展的 Obsidian Markdown 笔记。

## 这份笔记适合怎么用

如果你是第一次打开，建议先读 [LLM 学习地图](<00-首页与索引/LLM 学习地图.md>)。它给出了从输入表示、Transformer、预训练、后训练、推理优化、训练系统到 Agent 的完整主线。

如果你已经在准备面试，不建议从头到尾顺序背完。更高效的方式是按岗位问题反推阅读路线：

1. 算法岗先抓 `模型架构`、`后训练`、`常考手撕`。
2. 推理/系统岗先抓 `大模型推理`、`训练系统工程`、`模型架构/MoE`。
3. Agent/RL 岗先抓 `Agent 框架`、`后训练/RL 基础`、`后训练/RL 算法`、`训练系统工程/veRL Ray 后训练系统`。
4. 临面前一两天直接刷五份八股文件和 `常考手撕`，用它们做口头复述和代码默写。

## 当前目录怎么读

| 目录 | 主要内容 | 食用方法 |
|---|---|---|
| [00-首页与索引](<00-首页与索引>) | 全局学习地图 | 先用它建立主线，不要一上来陷入单篇细节 |
| [输入表示](<输入表示>) | Tokenizer、BPE、Chat Template、Tool Calling | 面试常用于解释“用户请求如何变成模型输入”，也是 Agent 工具调用的前置知识 |
| [模型架构](<模型架构>) | Transformer、Attention、FFN、Norm、MoE、长上下文 | 每篇都要能说清楚模块解决的问题、核心公式、复杂度和工程取舍 |
| [预训练](<预训练>) | Causal LM、数据工程、Scaling Law、Muon | 用来回答“基座模型能力怎么来、训练资源怎么估、训练为什么稳定或不稳定” |
| [后训练](<后训练>) | SFT、LoRA、RLHF、PPO、DPO、GRPO、DAPO、GSPO、Agentic RL | 这是当前仓库最重要的主线之一，建议按“基础概念 -> 算法对比 -> 系统落地”读 |
| [大模型推理](<大模型推理>) | KV Cache、Flash Attention、vLLM、SGLang、Speculative Decoding、Ray、并行推理 | 用来回答吞吐、延迟、显存、调度、KV 管理和服务化问题 |
| [训练系统工程](<训练系统工程>) | FSDP、ZeRO、并行训练、资源估算、veRL/Ray、Infra 八股 | 适合和后训练 RL 一起读，重点理解 rollout、reward、reference、trainer 的分布式闭环 |
| [Agent 框架](<Agent 框架>) | ReAct、Plan-and-Execute、Agent 上下文和工具调用八股 | 适合准备 Agent 工程、工具调用、上下文管理和 MCP/Skill 类问题 |
| [常考手撕](<常考手撕>) | Softmax、CrossEntropy、Attention、GQA/MQA/MLA、Kmeans、LSTM、Norm、Conv2D | 不只是看代码，要按 ACM 输入输出格式自己默写并跑样例 |
| [多模态](<多模态>) | Qwen2.5-VL 模态融合机制 | 用来回答视觉 token、projector/adapter、多模态输入如何进入 LLM |

## 四条推荐阅读路线

### 1. 大模型算法基础路线

适合刚开始系统复习，目标是把“模型为什么这么设计”讲顺。

1. [Tokenizer 和 BPE](<输入表示/Tokenizer 和 BPE.md>)
2. [Chat Template 和 Tool Calling](<输入表示/Chat Template 和 Tool Calling.md>)
3. [Transformer 算法概述](<模型架构/Transformer/Transformer 算法概述.md>)
4. [注意力机制（MHA、MQA、GQA、MLA）](<模型架构/Transformer/注意力机制（MHA、MQA、GQA、MLA）.md>)
5. [位置编码](<模型架构/Transformer/位置编码.md>)
6. [FFN](<模型架构/Transformer/FFN.md>)
7. [Layer Norm 和 RMS Norm](<模型架构/Transformer/Layer Norm 和 RMS Norm.md>)
8. [MoE 和 Expert Parallelism](<模型架构/MoE/MoE 和 Expert Parallelism.md>)

读完这一条线后，至少要能回答：

- Transformer 一层里 attention、FFN、norm 的顺序和作用是什么？
- MHA、MQA、GQA、MLA 分别在优化什么？
- KV Cache 为什么能加速 decode，它和 MQA/GQA 有什么关系？
- MoE 为什么省计算但增加路由、通信和负载均衡问题？

### 2. 后训练和 RL 算法路线

适合准备后训练、RLHF、推理模型训练、Agentic RL 方向。

1. [大模型训练八股](<后训练/大模型训练八股.md>)
2. [RLHF 总览](<后训练/RL 基础/RLHF 总览.md>)
3. [Reward Return Value Advantage](<后训练/RL 基础/Reward Return Value Advantage.md>)
4. [Rollout 和采样](<后训练/RL 基础/Rollout 和采样.md>)
5. [PPO 算法原理](<后训练/RL 算法/PPO/PPO 算法原理.md>)
6. [DPO 算法原理](<后训练/RL 算法/DPO/DPO 算法原理.md>)
7. [GRPO 算法原理](<后训练/RL 算法/GRPO/GRPO 算法原理.md>)
8. [DAPO 算法原理](<后训练/RL 算法/DAPO/DAPO 算法原理.md>)
9. [Dr.GRPO 算法原理](<后训练/RL 算法/Dr.GRPO/Dr.GRPO 算法原理.md>)
10. [GSPO 算法原理](<后训练/RL 算法/GSPO/GSPO 算法原理.md>)
11. [GiGPO 算法原理](<后训练/RL 算法/GiGPO/GiGPO 算法原理.md>)
12. [RL 算法八股](<后训练/RL 算法/RL 算法八股.md>)

这条线不要只背缩写。每个算法都按同一套模板复述：

- 它要解决 PPO/GRPO/DPO 的哪个痛点？
- reward、advantage、old policy、reference model 分别在哪里出现？
- 是 on-policy、near on-policy 还是离线偏好优化？
- 对长 CoT、MoE、异步 rollout、训练/推理分离有什么影响？
- 失败模式是什么，应该监控哪些指标？

### 3. 推理系统和训练系统路线

适合准备 infra、推理加速、分布式训练和 RL 系统岗位。

1. [KV Cache](<大模型推理/KV Cache.md>)
2. [Flash Attention](<大模型推理/Flash Attention.md>)
3. [Speculative Decoding 和 EAGLE](<大模型推理/Speculative Decoding 和 EAGLE.md>)
4. [vLLM SGLang](<大模型推理/vLLM SGLang.md>)
5. [数据并行（DP、DDP、FSDP）](<大模型推理/数据并行（DP、DDP、FSDP）.md>)
6. [Ray](<大模型推理/Ray.md>)
7. [FSDP ZeRO 和并行训练](<训练系统工程/FSDP ZeRO 和并行训练.md>)
8. [大模型资源估算](<训练系统工程/大模型资源估算.md>)
9. [veRL Ray 后训练系统](<训练系统工程/veRL Ray 后训练系统.md>)
10. [Infra 八股](<训练系统工程/Infra 八股.md>)

这条线要抓住四个关键词：显存、吞吐、延迟、通信。面试回答时不要停留在“用了某框架”，而要说清楚瓶颈在哪里：

- KV Cache 省的是重复计算，但吃显存。
- Flash Attention 优化的是 attention IO 和显存访问。
- vLLM/SGLang 解决的是高吞吐 serving、调度、KV 管理和复杂 LLM program 执行。
- FSDP/ZeRO/TP/PP/EP 解决的是模型和优化器状态放不下、通信太贵、流水线气泡等问题。
- veRL/Ray 这类系统把 rollout、reward、reference logprob、actor update、权重同步拆成分布式 worker。

### 4. Agent 和应用算法路线

适合准备 Agent、工具调用、业务 workflow、Agentic RL。

1. [Chat Template 和 Tool Calling](<输入表示/Chat Template 和 Tool Calling.md>)
2. [ReAct 和 Plan-and-Excute](<Agent 框架/ReAct 和 Plan-and-Excute.md>)
3. [Agent 框架八股](<Agent 框架/Agent 框架八股.md>)
4. [Agentic RL 八股](<后训练/RL 基础/Agentic RL 八股.md>)
5. [Rollout 和采样](<后训练/RL 基础/Rollout 和采样.md>)
6. [veRL Ray 后训练系统](<训练系统工程/veRL Ray 后训练系统.md>)
7. [Infra 八股](<训练系统工程/Infra 八股.md>)

这条线的关键不是“Agent 会调用工具”这么简单，而是能讲清楚完整闭环：

- 模型如何输出 action 或 function call？
- runtime 如何解析、校验、执行工具？
- observation 如何回填上下文？
- 上下文溢出时保留什么、压缩什么、丢弃什么？
- 多轮 trajectory 里哪些 token 参与 loss，哪些要 mask？
- rollout 很长、工具很慢、policy lag 明显时怎么处理？

## 五份八股文件怎么刷

当前面试速查主要集中在这几份：

1. [大模型训练八股](<后训练/大模型训练八股.md>)
2. [RL 算法八股](<后训练/RL 算法/RL 算法八股.md>)
3. [Agentic RL 八股](<后训练/RL 基础/Agentic RL 八股.md>)
4. [Infra 八股](<训练系统工程/Infra 八股.md>)
5. [Agent 框架八股](<Agent 框架/Agent 框架八股.md>)

推荐刷法：

1. 第一遍只看问题，先自己说 30 秒，再对照答案补缺口。
2. 第二遍只看 `面试表达`，练成短回答，保证能在追问前先给出清晰主结论。
3. 第三遍回到相关原理笔记，把公式、流程图、失败模式补上。
4. 第四遍做横向比较，比如 PPO vs GRPO vs DPO vs DAPO，vLLM vs SGLang，ReAct vs CoT，SFT vs RL。
5. 临面前只复述“问题 -> 一句话答案 -> 关键追问 -> 失败模式/指标”。

八股文件的定位是“面试入口”，不是唯一答案。遇到不确定或新算法，应该回到对应算法原理笔记，必要时查原论文或官方文档。

## 常考手撕怎么刷

[常考手撕](<常考手撕>) 目录目前包含：

- [SoftMax](<常考手撕/SoftMax.md>)
- [交叉熵损失-CrossEntropy](<常考手撕/交叉熵损失-CrossEntropy.md>)
- [Self-Attention](<常考手撕/Self-Attention.md>)
- [MHA](<常考手撕/MHA.md>)
- [MQA](<常考手撕/MQA.md>)
- [GQA](<常考手撕/GQA.md>)
- [MLA](<常考手撕/MLA.md>)
- [Kmeans](<常考手撕/Kmeans.md>)
- [BatchNorm](<常考手撕/BatchNorm.md>)
- [LayerNorm](<常考手撕/LayerNorm.md>)
- [LSTM](<常考手撕/LSTM.md>)
- [Conv2D](<常考手撕/Conv2D.md>)

手撕题建议按“三步法”练：

1. 先说清楚输入输出 shape，例如 NCHW、OIHW、batch、seq_len、hidden_size、num_heads。
2. 再写朴素但正确的 NumPy/PyTorch 代码，不追求一开始就最优。
3. 最后跑测试用例，检查边界：维度、padding、mask、数值稳定、空簇、`-0.000000`、广播方向。

当前手撕笔记的标准结构是：

```text
面试定位 -> 输入输出格式 -> 可运行代码 -> 测试用例 -> 易错点
```

新增手撕笔记也建议保持这个结构，并至少给 3 条测试用例。能用 NumPy 解决的题优先写 NumPy 版本；注意力相关题可以保留 PyTorch 版本，因为它更贴近真实张量实现。

## 单篇笔记的阅读方法

每篇笔记不要平均用力。建议按下面顺序读：

1. 先看标题和 `面试定位`，判断它解决什么问题。
2. 找“一句话概括”或核心结论，把它压缩成自己的 20 秒回答。
3. 看公式和流程图，明确输入、输出、中间变量、复杂度。
4. 看对比表，理解它相对前一个方法改了什么。
5. 看失败模式和监控指标，这是面试追问里最容易拉开差距的部分。
6. 有代码的笔记必须跑样例，不要只读。

一个主题真正掌握的标准不是“看懂了”，而是能完成这三件事：

- 对非专业同学讲清楚它解决什么问题。
- 对面试官讲清楚公式、流程和 trade-off。
- 对代码题写出可运行版本，并能解释每个 shape。

## 面试输出模板

回答算法或系统问题时，可以固定用这套结构：

```text
1. 这个问题的背景/瓶颈是什么？
2. 这个方法的核心思想是什么？
3. 关键公式或流程是什么？
4. 它相比上一个方法改进在哪里？
5. 代价、失败模式和监控指标是什么？
6. 如果落到工程实现，要注意哪些输入输出、mask、并发、显存或版本一致性问题？
```

例如回答 GRPO/DAPO，不要只说“去掉 critic”或“动态采样”。更完整的回答应该覆盖 group reward、advantage、old logprob、ratio clipping、reference KL、长 CoT、全对/全错 group、off-policy 风险和监控指标。

## 如何继续维护

新增笔记时优先遵守现有风格：

1. 文件名用中文主题名，必要时保留英文缩写，例如 `KV Cache.md`、`LoRA 和 QLoRA.md`。
2. 概念类笔记先写“面试定位”和“一句话概括”，再展开原理。
3. 八股类笔记用“问题 -> 答案 -> 面试表达 -> 相关链接”的结构。
4. 手撕类笔记保持 ACM 输入输出、可运行代码、测试用例、易错点。
5. 涉及新论文、新框架、新版本能力时，优先核对原论文、官方文档或项目 README。
6. 链接尽量指向仓库内已有笔记，让 Obsidian 和 GitHub 都能顺着读。

## 当前复习重点

如果时间很紧，建议按下面优先级：

1. 先刷五份八股文件，建立面试回答框架。
2. 再刷 `RL 算法` 的 PPO、DPO、GRPO、DAPO、Dr.GRPO、GSPO。
3. 同时补 `vLLM SGLang`、`KV Cache`、`Flash Attention`、`FSDP ZeRO`、`veRL Ray 后训练系统`。
4. 每天默写 1 到 2 道 `常考手撕`，确保代码能跑。
5. 最后用 [LLM 学习地图](<00-首页与索引/LLM 学习地图.md>) 串回完整主线，避免知识点碎片化。
