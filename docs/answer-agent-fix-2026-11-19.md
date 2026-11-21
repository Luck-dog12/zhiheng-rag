# 回答 Agent、上下文和截断恢复验证

> 历史记录：本报告描述 2026-11-19 的非思考 Agent。2026-11-20 已升级为思考模式、显式证据评估及最多 4 次自适应检索，当前行为见 [新版验证报告](adaptive-agent-thinking-2026-11-20.md)。下文的默认值与测量结果保留为当时基线。

日期：2026-11-19。检索仍为**方案 A：Qwen 稠密 + Milvus 内置 BM25**，远程嵌入为 `qwen3.7-text-embedding`，重排序为 `qwen3.7-text-rerank`。控制器及回答均使用现有 `.env` 的 **DeepSeek / deepseek-flash**。未重新入库，复用已有 8 份阿里云文档、746 块及外部 PostgreSQL、Milvus；未部署本地模型。

## 已确认的根因

实际旧记录中，“我应该怎么调用模型？指导我”以 `generation_incomplete` 失败。数据库只保留原问题，未保存明确后的任务、草稿和证据；当时的上下文只接纳成功回答。“继续”因此接到更早的 OpenAI 接口介绍，误判需要澄清。

旧的额度递增仅覆盖指代消解、元数据等结构化 JSON，流式回答仍是一次生成，且没有回答阶段的工具决策。这些是应用编排的缺口，不是用户没有重复问题，也不是换用了其他回答模型。

`max_tokens` 是生成额度，不能仅凭 `finish_reason=length` 推断整个上下文窗口已满。提高额度本身不依赖 Agent；本次同时补齐输出恢复和实际的 Agent 工具调用。[DeepSeek 请求参数](https://api-docs.deepseek.com/api/create-chat-completion/)、[工具调用](https://api-docs.deepseek.com/guides/tool_calls/)。

## 现在的行为

- DeepSeek 通过 function calling 选择检索、复用历史证据、回答或澄清。默认最多 6 个决策步骤、2 次检索；每次检索保留混合召回、RRF、重排序及最多 5 个片段。模型不能指定授权范围或调用任意外部地址。
- 最近 6 轮可访问上下文同时传给决策与回答阶段。可恢复的失败任务保留问题，但未验证正文不充当事实。“继续”承接已完成回答，或按原任务完整重答未完成回答；兼容旧版只有问题、没有恢复状态的失败记录。
- 控制器明确使用非思考模式及 `tool_choice=required`。回答用独立开关 `ANSWER_DEEPSEEK_THINKING=disabled`，不会再默认把回答额度花在隐藏推理中。指代消解仍保持原有非思考设置。
- 流式输出遇到长度限制，默认按 **8,192 → 16,384 → 32,768** token 重试；网络中断等可恢复终止也有有界重试。每次重新生成前发出 SSE `reset`，替换旧草稿，不重复拼接正文与引用。
- 达到额度、次数或总时限后，保存明确的问题、证据 ID、工具轨迹和未验证草稿，页面显示“重新完成”。成功回答显示“继续展开”。权限撤销或来源版本变化后，历史草稿和恢复线索同样被服务端隐藏。
- 正式答案须通过结束标记、引用编号、实际来源 URL、权限及有效版本校验。内容过滤不自动重试。失败草稿与正式答案分开存储。

## 实际外部服务结果

下面的 token 额度是**回答阶段实际请求值**。为稳定触发截断，测试使用独立 API 进程降低回答额度，未修改 `.env` 或主实例默认配置。

| 场景 | 实际结果 |
|---|---|
| “opemai兼容接口是那个？” → “我应该怎么调用模型？指导我” → “继续” | 三轮完成；最后一轮 `reuse_evidence → answer → generate_answer`，mode=`continue`，展开流式调用、LangChain 等，2,416 字符、4 个引用 |
| 真实旧会话兼容 | 复制原失败记录和其后的澄清到测试会话，不修改用户原会话；“继续”恢复“如何调用模型”任务，mode=`restart`，3,700 字符、4 个引用 |
| 自动加大额度 | 真实 DeepSeek 连续返回两次 `length`，按 **512 → 1,024 → 2,048** 重试，收到两次 `reset`，最终完成 2,747 字符、5 个引用；页面累计文本与已保存答案一致 |
| 故意耗尽额度后跨进程恢复 | **64 → 128 → 256** 后按预期失败，604 字符草稿和恢复状态可重新读取；改向正常 API 发送“继续”，以 8,192 额度恢复成 2,841 字符、5 个引用 |
| Edge 页面恢复与续写 | 另一个真实截断测试会话刷新后显示草稿及“重新完成”；点击后完成 4 个引用的答案，再点击“继续展开”完成后续回答；两个动作的数据库状态分别为 `restart`、`continue` |
| 原文追溯 | Edge 打开引用面板，显示当前版本的原文段落；脚本逐条核对引用与分块版本 |

共执行 10 个真实生成轮次，其中 2 个按设计耗尽测试额度，其余 8 个完成；旧记录复制不计为真实生成。本轮不是新的 20 路并发或全量质量评测，不把少数成功样本推广为任意问题都不会截断。

自动额度场景用时 19.08 秒；耗尽额度后单独恢复用时 9.13 秒；普通三轮分别为 13.67 / 8.07 / 8.17 秒。这些是单次专项样本，升级前报告的并发 P50/P95 不作为 Agent 新基线。

证据文件（本地忽略目录，无模型密钥）：

- `output/answer-agent-live/context.json`：三轮会话、答案、实际工具轨迹。
- `output/answer-agent-live/legacy-context.json`：旧失败记录恢复。
- `output/answer-agent-live/automatic.json`：自动重试的额度、结束原因及 SSE 事件。
- `output/answer-agent-live/exhausted.json`：耗尽额度、已保存草稿及跨进程恢复。
- `output/answer-agent-live/browser-results.json`：浏览器恢复及续写的数据库记录。
- `output/playwright/answer-agent-recovery-before.png`、`answer-agent-recovery-after.png`、`answer-agent-citation.png`：本机 Edge 页面证据。

## 自动检查与迁移

- 后端 **123 项测试通过**，其中 3 项使用外部 PostgreSQL 独立临时 schema；无跳过，Ruff 通过。
- 前端 **4 项测试通过**，TypeScript 与生产构建通过；仍有既存的 Ant Design 大包提示。
- 专项覆盖：跨轮上下文、失败问题恢复、纯推理耗尽且正文为空、流末段保留、SSE 重置、受限工具参数、二次检索上限、权限撤销、版本失效、草稿不进入事实历史、工具协议非法响应、内容过滤及旧数据迁移。
- Alembic `0002` 已应用到外部 PostgreSQL，为旧轮次填入空恢复状态，旧问题和回答不变；临时库迁移兼容、升降级和 schema check 均通过。
- Edge 登录后复测控制台 0 错误、0 警告。全程使用已安装 Microsoft Edge，没有下载 Chrome 或 Chrome CLI。

## 启动与限制

本次修复版已运行于 **http://127.0.0.1:8004**。旧 `8001` 进程未替换，不要用旧入口验证本次修复。手动启动命令：

```powershell
uv run alembic upgrade head
$env:APP_ORIGIN='http://127.0.0.1:8004'
uv run uvicorn rag.api:create_app --factory --host 127.0.0.1 --port 8004
```

已有进程运行时无需重复启动。新参数见 `.env.example`：`ANSWER_INITIAL_TOKENS`、`ANSWER_MAX_TOKENS`、`ANSWER_MAX_ATTEMPTS`、`ANSWER_DEEPSEEK_THINKING`、`AGENT_MAX_STEPS`、`AGENT_MAX_SEARCHES`、`CHAT_TIMEOUT_SECONDS`。真实脚本为 `scripts/verify_answer_agent_live.py`，`context` 使用 8004，`exhausted` 和 `automatic` 分别另需低额度测试实例 8002 / 8003。

恢复采用从原任务完整重答，未承诺从上次半句逐字续写。旧版没有持久化的草稿无法补回，但原问题可重新生成。服务商上下文上限、长答案、网络与总时限仍可能使请求失败；达到上限时明确失败并保留可恢复状态。硬退出的进行中请求经过 10 分钟失活窗口后可重新领取。还需对新版 Agent 补做 20 路并发、持续压测和更广业务问题的质量评测。
