# Agent 框架八股

## 1. ReAct 的基本原理是什么？相比 CoT 的优势是什么？

**答案：** CoT 主要让模型在文本里展开推理，仍然依赖模型内部知识；ReAct 把 Reasoning 和 Acting 交替起来，让模型在推理过程中选择工具、执行动作、读取 observation，再根据真实反馈继续推理。优势是能接入外部信息和环境，减少纯记忆幻觉，轨迹更可审计，也更适合动态任务。  
> **面试表达：** CoT 是“只想”，ReAct 是“边想边查、边做边改”。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]

---

## 2. ReAct loop 如何识别 action、执行工具、接收 observation、继续推理？

**答案：** 模型输出会被约束成可解析格式，比如 function call JSON 或 Action/Action Input。runtime 解析 action，校验工具名和参数，执行对应工具，把结果包装成 observation 追加到上下文，再调用模型生成下一步。这个循环持续到模型输出 final answer、工具失败不可恢复或达到预算。  
> **面试表达：** ReAct loop 的关键是解析、校验、执行、回填 observation，再继续让模型决策。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[输入表示/Chat Template 和 Tool Calling|Chat Template 和 Tool Calling]]

---

## 3. Agent 遇到上下文窗口溢出时怎么办？

**答案：** 先区分哪些信息必须保留：用户目标、约束、当前计划、关键 observation、工具 schema 和未完成状态。可丢弃或压缩低价值历史，比如重复推理、长日志、已完成步骤细节；把完整原始结果存到外部存储，只在上下文里保留引用和摘要；必要时做 checkpoint、检索式记忆或重新规划。  
> **面试表达：** 上下文管理不是简单截断，而是把状态、证据和可恢复引用留下来。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[模型架构/Transformer/稀疏注意力和长上下文|稀疏注意力和长上下文]]

---

## 4. 工具返回结果很长时，应该裁剪、摘要、丢弃还是压缩？

**答案：** 要看结果用途。若只需要少数字段，应该结构化抽取；若需要整体语义，做摘要并保留原文引用；若后续可能复查，原始结果放外部存储，上下文里只放 ID、路径或短摘要；明显无关或重复内容可以丢弃。不能盲目压缩，因为压缩可能丢掉证据、数字、错误信息和权限状态。  
> **面试表达：** 长 observation 要从“后续决策需要什么信息”反推处理方式。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[模型架构/Transformer/稀疏注意力和长上下文|稀疏注意力和长上下文]]

---

## 5. 哪些上下文不能轻易压缩或删除？

**答案：** 不能轻易删的是会改变任务语义或安全边界的信息：用户原始目标和硬约束、权限和安全规则、工具 schema、当前计划和未完成状态、关键 observation、错误原因、外部资源 ID、引用来源、已经执行过的不可逆动作，以及最终答案必须依赖的证据。  
> **面试表达：** 可以压缩过程噪声，但不能压缩掉目标、约束、状态和证据。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[输入表示/Chat Template 和 Tool Calling|Chat Template 和 Tool Calling]]

---

## 6. Codex / Claude Code 类系统的上下文管理大概怎么做？

**答案：** 这类 coding agent 系统一般会把对话目标、计划、最近关键操作、文件路径、工具输出摘要和阻塞点保留在短上下文里；把完整文件内容、命令输出、历史日志按需读取，而不是全部常驻上下文；长任务中通过 checkpoint/summary 压缩会话状态；对代码位置用路径和行号引用，对外部工具结果保留可复查证据。  
> **面试表达：** Coding agent 的上下文管理核心是“状态常驻，原文按需读取，关键证据可追溯”。  

**相关：** [[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]、[[模型架构/Transformer/稀疏注意力和长上下文|稀疏注意力和长上下文]]

---

## 7. MCP 和 Skill 分别是什么，功能定位有什么区别？

**答案：** MCP 更像模型客户端和外部能力之间的连接协议，用标准方式暴露 tools、resources、prompts，以及数据库、文件系统或远程服务等上下文来源；Skill 更像本地可复用的任务说明书或工作流规范，告诉 agent 面对某类任务时应该按什么步骤、约束和检查清单执行。简单说，MCP 扩展“能连接和调用什么”，Skill 约束“应该怎么做”。  
> **面试表达：** MCP 是能力接口，Skill 是行为方法论。  

**相关：** [[输入表示/Chat Template 和 Tool Calling|Chat Template 和 Tool Calling]]、[[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]

---

## 8. Function call 和 Agent 工具调用的关系是什么？

**答案：** Function call 是模型输出结构化工具调用的一种协议，解决“如何让模型生成可解析、可校验的函数名和参数”。Agent 工具调用是在此基础上的完整运行循环：选择工具、执行工具、接收 observation、更新上下文、处理错误、决定是否继续调用或输出 final answer。Function call 是一次动作格式，Agent loop 是多步决策系统。  
> **面试表达：** function call 是工具调用的语法，Agent 是围绕工具调用构建的控制循环。  

**相关：** [[输入表示/Chat Template 和 Tool Calling|Chat Template 和 Tool Calling]]、[[Agent 框架/ReAct 和 Plan-and-Excute|ReAct 和 Plan-and-Execute]]
