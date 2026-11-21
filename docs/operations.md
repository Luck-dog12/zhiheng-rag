# 配置、处理策略与运维说明

项目概览与首次启动见 [README](../README.md)。以下说明对应当前实现；历史验证结果按各报告日期解读。

## 处理链路

```text
TXT / Markdown / DOCX / 文本 PDF
  → 结构化解析 → 保守清洗 → DeepSeek 指代消解（默认关闭思考）
  → 自然分块 → DeepSeek 元数据卡片
  → PostgreSQL 正文 / 元数据 + S3 原文 / 中间结果
  → qwen3.7-text-embedding 稠密 / 模型稀疏向量 + Milvus 原生 BM25 索引

问题 + 最近 6 轮可访问对话（含可恢复的未完成任务） → DeepSeek 思考与工具决策
  → 选择查询及 hybrid / dense / sparse，或复用并核验历史证据
  → 知识库权限过滤 → 所选路线召回（每路默认 50）
  → RRF(k=60) → 当前版本回查 → qwen3.7-text-rerank
  → 累积证据并评估完整原问题 → 不足时改写 / 拆分 / 切换路线（最多 4 次检索）
  → 最多 5 个相关片段 → DeepSeek 思考后归纳回答 / 截断恢复 / 校验引用
```

正文和元数据通过 `chunk_id` 关联；向量化的是处理后的正文。指代消解保留歧义，不直接覆盖原文。时间地点必须有逐字原文证据，日期归一化由代码根据明确日期完成。

指代消解默认共用 DeepSeek 凭证与模型，`COREFERENCE_DEEPSEEK_THINKING=disabled` 仅控制这一用途；元数据、追问改写和回答不受此开关影响。调用独立指代模型时不发送 DeepSeek 特有参数。此设置纳入处理指纹，切换后重新上传会生成新版本。结构化输出初始额度为 4,096 token，仅在返回 `finish_reason=length` 时将两次修复的额度依次提高至 8,192 / 16,384；所有结果仍须通过完整性、结构及证据校验。

回答引用编号由服务端绑定实际分块和当前版本。需要展示接口地址时，仅允许逐字复制所引用片段中的 HTTP(S) 地址；不存在的来源编号和自行拼接的链接会使该轮失败，不进入后续有效历史。

回答阶段使用同一份 `DEEPSEEK_*` 配置：控制器通过真实 function calling 选择 `search_knowledge_base`、`reuse_evidence`、`assess_evidence`、`answer` 或 `ask_clarification`。默认最多 12 个决策步骤、4 次检索；每次取证后必须评估相关性、要点覆盖和冲突，证据不足且仍有预算时继续查证。各轮相关证据可以组合，充分时提前结束；预算耗尽后只回答有依据部分并说明缺口，无相关证据时明确资料不足。模型不能指定授权范围或调用任意外部工具。

控制器 `AGENT_DEEPSEEK_THINKING=enabled` 和回答 `ANSWER_DEEPSEEK_THINKING=enabled` 默认均开启思考，推理强度默认为 `high`。思考工具调用使用 `tool_choice=auto`，在同一次工具循环内回传 `reasoning_content`；服务端仍校验工具状态与参数，不把模型直接输出的未核验文本作为答案。隐藏推理不进入页面或会话数据库。指代消解仍使用自己的非思考开关，不受影响。

最终答案默认直接回答问题，用自己的语言概括，说明类问题优先少量要点；完整原文通过引用面板查看。用户要求详细解释或代码时允许展开，不用硬截字数损害完整性。简洁性来自回答指令与证据筛选，开启思考本身不保证答案更短或更准确，也会增加生成用量和耗时。新版 Agent 尚未重跑 20 路并发，旧报告不能作为当前延迟基线。

片段数量分三层：默认两路各召回 50 条，融合去重与权限/版本过滤后最多重排 100 个候选；每轮取前 `ANSWER_TOP_K` 个片段（默认 5，可配置 1–5）交给 Agent 评估；多轮证据累积后，最终回答仍最多使用 `ANSWER_TOP_K` 个相关片段。检索调试页分别展示“选中片段”和“其余候选”，不是把全部重排候选都交给最终回答模型。

“继续”会关联最近的实质任务：已完成回答承接展开，失败回答从原问题重新组织完整答案。失败的问题、明确后的任务、证据 ID、生成状态及未验证草稿存入 PostgreSQL；草稿仅供页面展示，不作为后续回答事实。恢复与复用证据时重新检查权限和当前文档版本。

回答遇到 `finish_reason=length`，默认按 8,192 → 16,384 → 32,768 token 最多生成三次。重试发送 SSE `reset`，前端替换旧草稿；完成引用校验后才保存为正式答案。网络中断可在同一额度内重试；达到尝试数、额度或总时限后保留可恢复状态，页面提供“重新完成”。这不是无限输出承诺，服务商上下文上限及网络可用性仍然有效。相关参数见 `.env.example`，新增状态字段需执行 `uv run alembic upgrade head`。

`CLEAN_NORMALIZE_WHITESPACE` 和 `CLEAN_REMOVE_FILLERS` 分别控制无效空白处理与明确语气噪声清理，默认开启；代码、引用和表格始终受保护。指代消解按章节读取前文，长段落拆成最多 4,000 字符的处理窗口，并把替换位置映射回完整原段落。规则变更会影响处理配置指纹，重传会生成新版本。

## 本地开发

需要 Python 3.12、uv、Node.js 20.19+，以及可访问的外部服务。Windows/PowerShell 可直接使用以下命令。

```powershell
uv sync --python 3.12
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# 编辑 .env，填写自己的外部服务配置
uv run alembic upgrade head
uv run rag create-admin
uv run rag init-index
uv run rag check-services --probe-models
```

`create-admin` 交互输入用户名和至少 12 位密码，没有生产默认账号。`init-index` 明确创建远程集合，启动 API 不自动修改数据库。`--probe-models` 会发送少量真实模型请求，产生服务商费用。

分别打开终端启动 API、Worker、前端：

```powershell
uv run uvicorn rag.api:create_app --factory --host 127.0.0.1 --port 8000
uv run rag worker
```

```powershell
Set-Location frontend
npm ci
npm run dev
```

浏览器打开 `http://localhost:5173`，与默认 `APP_ORIGIN` 保持一致。通过本机 Edge 验证，不需要安装 Chrome。接口文档位于 `http://127.0.0.1:8000/docs`。

## 外部服务配置

完整变量见 `.env.example`。密钥只保存在服务端配置中，不进入前端或日志。

| 服务 | 必填配置与契约 |
|---|---|
| PostgreSQL | `DATABASE_URL=postgresql+asyncpg://...`；建议专用账号、数据库及连接池 |
| 对象存储 | `S3_ENDPOINT_URL/REGION/BUCKET/ACCESS_KEY/SECRET_KEY`；已有私有 bucket，需要读写删除权限 |
| Milvus | `MILVUS_URI/TOKEN/DATABASE/COLLECTION`；远程 HTTP(S) URI，支持 BM25、中文分析器和稀疏向量，客户端锁定于 2.6 系列 |
| DeepSeek | `DEEPSEEK_BASE_URL/API_KEY/MODEL`；默认 `deepseek-flash`，元数据、改写和回答共用同一配置 |
| 指代消解 | 默认 `COREFERENCE_USE_DEEPSEEK=true`，共用 DeepSeek 配置；如需独立模型，设为 `false` 并填写 `COREFERENCE_BASE_URL/API_KEY/MODEL` |
| 嵌入 | `EMBEDDING_PROTOCOL/ENDPOINT/API_KEY/MODEL/REVISION/DIMENSION`；默认 qwen3.7-text-embedding，1024 维；模板使用 DashScope 原生接口 |
| 重排序 | `RERANK_PROTOCOL/ENDPOINT/API_KEY/MODEL`；完整 HTTP URL，默认 qwen3.7-text-rerank |

腾讯 COS 使用地域地址 `https://cos.<region>.myqcloud.com`（地址不含桶名），并设置 `S3_SIGNATURE_VERSION=s3`、`S3_ADDRESSING_STYLE=virtual`；其他 S3 服务默认 `s3v4` / `auto`，需要路径寻址时设置 `path`。应用对批量删除请求补充序列化 XML 的 Content-MD5，以兼容 COS。

真实依赖预检：`uv run python scripts/preflight_live.py`。该命令使用 `.env`，执行少量计费模型请求和临时对象写入、读取、删除，并分别检查 PostgreSQL、Milvus，结果保存到 `output/rag-live-preflight.json`。失败返回非零退出码；报告不包含密钥。单项模型探针通过不代表完整 RAG 入库或问答验收通过。

对话地址在 base URL 后追加 `/chat/completions`。`EMBEDDING_PROTOCOL=openai` 时使用 `EMBEDDING_BASE_URL` 追加 `/embeddings`；`dashscope` 时使用完整 `EMBEDDING_ENDPOINT`，请求为 `input.texts` 和 `parameters`，支持 `output_type=dense&sparse`、`text_type=query/document`、查询侧 `instruct`。批次最多 20 条，自动拆分请求；按响应索引还原顺序并校验稠密维度及稀疏词表索引。

`RERANK_PROTOCOL=openai` 请求为 `{model, query, documents, top_n}`，响应为 `results: [{index, relevance_score}]`；`dashscope` 使用 `input` / `parameters` 以及 `output.results`。必须返回所有候选，可以乱序；索引不得缺失或重复。百炼兼容重排序完整路径是 `/compatible-api/v1/reranks`，不是嵌入使用的 `/compatible-mode/v1` 基础路径。

模板设置 `EMBEDDING_STORE_SPARSE=true`：一次远程调用取得正文稠密和模型稀疏向量，Milvus 的 `sparse` 字段由内置 BM25 函数生成，额外的 `model_sparse` 字段存储模型向量并用 IP 索引。`RETRIEVAL_SPARSE_MODE=bm25` 是默认方案；改成 `model` 并重启 API 可切换方案，已完整存储两类向量时不需要重建。两个方案都保留相同的权限、有效版本校验和重排序流程。

存储 Qwen 稀疏向量不表示当前检索使用方案 B；稀疏来源由 API 启动时加载的 `RETRIEVAL_SPARSE_MODE` 决定。当前使用 `bm25`，即方案 A 的索引配置。Agent 的 `hybrid` 同时检索 Qwen 稠密与 BM25，`dense` 只检索稠密，`sparse` 只检索 BM25；这是单次检索路线选择，不是自动切换到方案 B。检索调试接口仍默认混合检索。

调整模型、协议、维度或是否存储模型稀疏向量时，需要新集合并重新入库；嵌入/索引/处理指纹会阻止混用旧向量。旧集合不会被自动覆盖或删除。上线前通过 `check-services --probe-models` 和真实评测验证接口。双方案对比命令与适用边界见 [评测说明](../evals/README.md#两种混合检索方案对比)。

## 管理与使用

1. 管理员创建知识库和普通成员，并在“成员授权”中分配知识库。
2. 选择知识库，上传不超过 50 MB 的支持格式文件，查看后台处理阶段和错误。
3. 打开文档查看分块、摘要、关键词、时间地点、原文以及修改记录。
4. 在“检索调试”查看两路排名、RRF 与重排序得分，区分本次选中的片段和其余调试候选。
5. 在“知识问答”选择检索范围、发送问题、追问并展开原文引用。
6. 更新文档时保持旧版本在线，待新版本完整入库后切换；删除立即禁止访问，后台任务清理对象和索引。
7. 在最近对话列表右侧点击删除图标，确认后删除该会话及全部问答记录；窄屏下先选择会话，再点击“删除当前对话”。删除当前会话后回到空白对话。知识库文档不受影响。

单企业、管理员/成员角色、知识库级授权。无公开注册。会话使用 HttpOnly Cookie 和 CSRF header。成员只能访问自己的会话；引用文档变化或权限撤销后，历史问答会隐藏，并从后续上下文排除。

`DELETE /api/v1/conversations/{conversation_id}` 仅允许会话本人调用（管理员也不能删除他人会话），成功返回 204，记录删除审计；其他人的会话与不存在的会话均返回 404。正在生成回答时返回 409，先停止生成或等待完成后再删除。删除操作和开始问答共用会话行锁，避免并发创建遗留消息。此功能无需新增数据库迁移。

## 可靠性与运维

- Worker 用 PostgreSQL `FOR UPDATE SKIP LOCKED`、租约 token 和心跳恢复任务；发布时再次验证租约及文档目标版本，阻止旧任务覆盖新版本。
- 解析/指代消解中间结果写入 S3；分块元数据逐条保存；重试复用已完成的检查点，向量 upsert 使用稳定 ID。
- PostgreSQL 是发布和授权的权威状态，不声称它与 Milvus 有跨库事务。候选无效时扩大召回到每路最多 200，再取最多 100 个有效候选重排序。
- 网络错误、429 和可重试 5xx 最多尝试三次。回答编排另按 `ANSWER_MAX_ATTEMPTS` 恢复不完整生成；总时限默认 480 秒。未验证的正文不进入事实历史，原问题保留用于恢复。
- `/api/v1/health/live` 只检测进程；`/api/v1/health/ready` 检查数据库、对象存储、Milvus 和模型配置，不执行付费模型调用。
- `/api/v1/metrics` 管理员可访问 API 的 Prometheus 指标。Worker 默认在 `127.0.0.1:9001` 暴露处理阶段、模型用量和队列指标，通过 `rag worker --metrics-host/--metrics-port` 调整；容器只在内部网络暴露该端口。多实例需分别采集再聚合。
- `/api/v1/audit` 提供最近操作记录。错误返回 request trace ID，不透传供应商响应原文或密钥。
- 默认每用户每分钟 30 次问答、60 次检索；每 IP 每五分钟 20 次登录尝试。全局问答并发限制是每个 API 进程 20；扩容时需同步服务商配额和数据库连接预算。

生产设置 `APP_ENV=production`、`APP_ORIGIN=https://实际域名`。通过反向代理终止 HTTPS，限制请求体 50 MB，关闭 SSE 缓冲并配置流式请求超时。构建前端后，FastAPI 可同源托管静态文件：

```powershell
Set-Location frontend
npm run build
Set-Location ..
docker compose build
docker compose run --rm api /app/.venv/bin/alembic upgrade head
docker compose up -d
```

容器只包含应用 API/Worker，不包含模型、Milvus、PostgreSQL 或 S3 服务。发布前备份外部 PostgreSQL 和对象存储；应用回滚保持数据库兼容，破坏性迁移需另行安排。

## 测试与评测

```powershell
uv run ruff check backend tests scripts
uv run pytest -q
Set-Location frontend
npm test
npm run build
```

测试使用 HTTP 协议模拟、内存对象/向量存储和 SQLite 测试数据库，生产程序不支持降级到这些测试替身。测试覆盖结构处理、模型返回校验、授权、版本发布、任务恢复、引用、50 问评测及 20 路并发。

另提供容量测试的断点恢复工具和外部 PostgreSQL 专项测试。未配置外部测试数据库时，三项 PostgreSQL 测试会明确跳过；运行方式见 `evals/README.md`。`scripts/capacity_live.py --dry-run` 可在无任何外部服务时预检万份文档生成。

合成语料及真实服务评测命令见 `evals/README.md`。模拟报告 `output/evaluation-simulated.json` 只证明应用处理和评测程序能运行；它不是实际模型质量、Milvus 召回率、PostgreSQL 并发锁或生产延迟的验收结果。

本轮验证结果、Edge 操作记录及待联调项目见 [验证记录](verification.md)。

独立的界面测试环境：

```powershell
# 先构建前端
uv run python -m tests.preview
```

打开 `http://127.0.0.1:8000`，使用 `preview / preview-password-2026`。这是仅监听回环地址的合成数据测试环境，使用 `tests/` 中的模拟服务；它不连接真实模型，不能用于生产。

## 首期边界

扫描 PDF/图片不执行 OCR；有文本与扫描页混合的 PDF 会明确列出缺失页。PDF 保留页码和基础表格，不承诺复杂多栏排版完全还原。暂不支持 Excel/PPT、多租户、SSO、文档级 ACL 或无限循环检索；回答 Agent 的多轮检索受明确步骤、次数和总时限限制。

方案 A 已完成真实供应商接入、PostgreSQL 多 Worker 锁与故障恢复、Milvus 索引一致性及 20 路并发验证。尚待完成方案 B 的独立应用端到端及并发验收、真实万份文档容量与持续压测、容器实际运行和业务专家质量验收；检索专项或模拟测试不替代这些项目。
