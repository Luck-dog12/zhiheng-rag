# 知衡 · 企业级 Agent RAG 工作台

面向企业内部知识问答的全栈应用：管理文档与知识库权限，让 DeepSeek 根据问题选择检索策略、评估证据并给出可追溯的回答。提供中文管理面板、流式问答、检索调试及文档入库 Worker。

**所有模型和基础存储均通过外部服务连接。** 应用不下载模型权重，也不启动本地 Milvus、PostgreSQL 或对象存储。当前默认使用 **Qwen 稠密向量 + Milvus 内置 BM25**；同时支持 Qwen 稀疏向量方案进行对比。

[快速启动](#快速启动) · [项目结构](#项目结构) · [配置与运维](docs/operations.md) · [测试与评测](#测试与评测) · [验证记录](docs/verification.md)

## 已有能力

- **文档管理**：TXT、Markdown、DOCX、文本型 PDF；保守清洗、指代消解、自然分块、元数据卡片、版本更新与删除。
- **Agent 问答**：DeepSeek 思考模式与工具调用；按证据缺口改写问题、拆分检索或切换检索路线，最多 4 次检索，最终归纳回答。
- **混合检索**：稠密与稀疏召回、RRF 融合、Qwen 重排序；调试页展示各阶段排名、得分与最终选中片段。
- **来源追溯**：引用关联真实分块、文档版本、页码或段落及原文；查看清洗、指代替换记录。
- **多轮会话**：最近 6 轮有效历史、追问与“继续”、未完成回答恢复、会话删除、SSE 状态与正文流。
- **权限与可靠性**：知识库授权、服务端会话、CSRF、审计；任务租约和心跳、检查点恢复、完整入库后发布新版本。

## 架构

```text
文档上传
  → S3 保存原文件 → 解析 / 清洗 / 指代消解 → 自然分块
  → DeepSeek 元数据卡片 → PostgreSQL 保存正文与元数据
  → Qwen 嵌入 → Milvus 稠密 / BM25 / 可选模型稀疏索引

问题 + 可访问的会话历史
  → DeepSeek Agent 规划查询
  → 权限过滤 → hybrid / dense / sparse 检索
  → RRF 融合 → 有效版本回查 → Qwen 重排序
  → 评估证据：不足则调整并再次检索，充分则结束取证
  → 最多 5 个片段 → DeepSeek 归纳回答 → 引用校验 → SSE 输出
```

向量由**分块正文**生成；摘要、关键词、时间与地点等元数据独立保存，通过同一 `chunk_id` 关联。PostgreSQL 是文档有效版本和授权的最终依据，检索不能仅凭 Milvus 的记录判断可见性。

| 层级 | 技术 / 默认模型 |
|---|---|
| API / Worker | Python 3.12、FastAPI、Pydantic、SQLAlchemy、Alembic |
| 管理界面 | React 19、TypeScript、Vite、Ant Design |
| 业务数据 / 文件 / 索引 | 外部 PostgreSQL / S3 兼容存储 / Milvus 2.6 功能接口 |
| 元数据、指代消解、Agent 与回答 | 共用 DeepSeek 配置，默认 `deepseek-flash` |
| 嵌入 | `qwen3.7-text-embedding`，默认 1,024 维 |
| 重排序 | `qwen3.7-text-rerank` |

模型名为项目当前配置，可按供应商可用型号调整。指代消解默认**关闭思考**；Agent 决策和最终回答默认**开启思考**。输出截断按额度递增进行有界重试，未通过引用校验的草稿不会成为后续对话事实。

## 快速启动

### 1. 准备环境与外部服务

需要 Git、Python 3.12、uv、Node.js 20.19+，以及已开通的 PostgreSQL、S3 兼容私有桶、Milvus 和模型 API。Milvus 服务须支持原生 BM25、中文分析器和稀疏向量。以下命令以 Windows PowerShell 为例。

```powershell
git clone https://github.com/Luck-dog12/zhiheng-rag.git
Set-Location zhiheng-rag
uv sync --frozen --python 3.12
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，填写自己的服务地址和凭证。已有 `.env` 时请保留实际配置，不要用模板覆盖。

| 配置组 | 主要变量 |
|---|---|
| PostgreSQL | `DATABASE_URL`，使用 `postgresql+asyncpg://...` |
| S3 | `S3_ENDPOINT_URL`、`S3_REGION`、`S3_BUCKET`、`S3_ACCESS_KEY`、`S3_SECRET_KEY` |
| Milvus | `MILVUS_URI`、`MILVUS_TOKEN`、`MILVUS_DATABASE`、`MILVUS_COLLECTION` |
| DeepSeek | `DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` |
| 嵌入 | `EMBEDDING_PROTOCOL`、`EMBEDDING_ENDPOINT`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_DIMENSION` |
| 重排序 | `RERANK_PROTOCOL`、`RERANK_ENDPOINT`、`RERANK_API_KEY`、`RERANK_MODEL` |
| 浏览器来源 | 开发环境 `APP_ORIGIN=http://localhost:5173`，须与访问地址一致 |

完整配置见 [.env.example](.env.example)。模板包含占位地址，不能直接连接服务。协议差异、COS 配置、模型指纹和切换注意事项见 [外部服务配置](docs/operations.md#外部服务配置)。密钥只进入服务端配置，不放入前端或 Git。

### 2. 初始化数据库、账号和索引

```powershell
uv run alembic upgrade head
uv run rag create-admin
uv run rag init-index
uv run rag check-services --probe-models
```

管理员账号通过交互创建，密码至少 12 位，没有生产默认账号。`init-index` 在外部 Milvus 创建应用集合；`--probe-models` 发送少量真实模型请求，会产生供应商用量。API 启动不会自动执行迁移或创建索引。

### 3. 启动应用

分别在三个终端执行：

```powershell
# 终端 1：API，在项目根目录
uv run uvicorn rag.api:create_app --factory --host 127.0.0.1 --port 8000
```

```powershell
# 终端 2：文档处理 Worker，在项目根目录
uv run rag worker
```

```powershell
# 终端 3：管理界面
Set-Location frontend
npm ci
npm run dev
```

- 管理面板：[http://localhost:5173](http://localhost:5173)
- OpenAPI：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 就绪检查：`GET /api/v1/health/ready`

使用本机 Edge 即可，不需要下载 Chrome。Vite 将 `/api` 代理到 `127.0.0.1:8000`；更改 API 端口时同步修改代理。创建知识库 → 授权成员 → 上传文档 → 等待入库完成 → 检索调试 / 知识问答。Worker 必须运行，上传任务才会被处理。

### 4. 同源运行或容器运行

在 `frontend` 目录执行 `npm ci` 和 `npm run build` 构建 `frontend/dist`，回到项目根目录启动 FastAPI 即可托管界面；直接访问 API 地址，并将 `APP_ORIGIN` 设为相同来源。修改 `.env` 后重启对应进程。

仓库也提供 [Dockerfile](Dockerfile) 与 [compose.yaml](compose.yaml)，容器构建自动编译前端；在项目根目录执行：

```powershell
docker compose build
docker compose run --rm api /app/.venv/bin/alembic upgrade head
docker compose run --rm api /app/.venv/bin/rag create-admin
docker compose run --rm api /app/.venv/bin/rag init-index
docker compose up -d
```

仅启动 API 和 Worker，所有基础服务仍使用外部地址。生产部署使用 HTTPS、`APP_ENV=production` 与实际 `APP_ORIGIN`，反向代理关闭 SSE 缓冲。详细说明见 [可靠性与运维](docs/operations.md#可靠性与运维)。

## 检索方案与片段数量

| 方案 | 配置 | 真实验证范围 |
|---|---|---|
| **A，默认**：Qwen 稠密 + Milvus 内置 BM25 | `RETRIEVAL_SPARSE_MODE=bm25` | 检索 A/B；早期完整应用端到端及 20 路并发；新版 Agent 专项验证 |
| **B**：Qwen 稠密 + Qwen 模型稀疏向量 | `RETRIEVAL_SPARSE_MODE=model` | 检索与重排序 A/B；尚未单独完成完整应用端到端和 20 路并发 |

`EMBEDDING_STORE_SPARSE=true` 表示同时保存模型稀疏向量，**不代表当前正在使用方案 B**。已完整保存两类向量时，可切换稀疏模式并重启 API；更换嵌入模型、维度或索引结构需要新集合及重新入库。

默认每路召回 50 条，融合与有效性检查后最多重排 100 个候选；每轮按**重排序分数降序取前 5 个**供 Agent 评估，最终回答仍最多使用 5 个片段。调试页中的“稠密第 N 名”“BM25 第 N 名”是各自初始召回排名，因此不一定连续为 1–5；最终排序应看“重排序第 N 名”。

## 项目结构

```text
backend/
  rag/
    api.py / security.py     HTTP 接口、登录、授权、审计
    config.py / services.py  配置与依赖装配
    models.py / schemas.py   数据表与类型契约
    processing.py / jobs.py  文档处理、任务租约、检查点和版本发布
    providers.py            外部模型协议、重试与响应校验
    vector.py / storage.py  Milvus 与 S3 适配器
    retrieval.py            召回、RRF、版本过滤、重排序
    agent.py / chat.py      工具决策、证据评估、会话、SSE 与引用
    cli.py / telemetry.py   管理命令与指标
  migrations/              Alembic 数据库迁移
frontend/src/              中文管理面板和问答界面
tests/                     模拟服务回归及可选 PostgreSQL 集成测试
scripts/                   依赖预检、文档下载、真实评测与容量测试
evals/                     合成语料、证据标注与评测说明
test-documents/aliyun/     官方测试文档来源目录；快照在本地生成
docs/                      运维说明与按日期归档的验证报告
.env.example               不含密钥的配置模板
```

## 测试与评测

本地回归使用模拟模型、内存存储与测试数据库，不消耗真实模型额度：

```powershell
uv run ruff check backend tests scripts
uv run pytest -q
Set-Location frontend
npm test
npm run build
```

外部 PostgreSQL 专项测试需要显式开启，默认跳过 3 项。真实评测、50 问数据集、20 路并发与万份文档容量工具的运行方法见 [评测说明](evals/README.md)。

阿里云测试文档下载到本地，快照与原始运行报告不随仓库上传：

```powershell
uv run --with markdownify python scripts/download_aliyun_docs.py
```

来源清单见 [官方文档测试集](test-documents/aliyun/README.md)。官方文档可能更新，复跑时须重新校验逐字证据，历史报告中的指标不保证在新快照上复现。

| 文档 | 用途 |
|---|---|
| [配置、处理策略与运维](docs/operations.md) | 模型协议、指代处理、Agent 预算、故障恢复、权限与部署 |
| [验证总览](docs/verification.md) | 自动测试与真实服务验收的范围及边界 |
| [双方案检索对比](docs/rag-live-test-2026-11-18.md) | 方案 A / B 在官方文档上的检索与重排序结果 |
| [方案 A 端到端](docs/rag-e2e-live-2026-11-19.md) | 入库、问答、权限、版本、Worker 恢复及早期并发基线 |
| [思考模式 Agent](docs/adaptive-agent-thinking-2026-11-20.md) | 多次检索、错配恢复、证据评估和简洁回答 |
| [会话与检索展示](docs/conversation-management-2026-11-20.md) | 会话删除、片段数量与排名显示 |

## 当前边界

当前为单企业、知识库级授权；不包含 OCR、Excel/PPT、多租户、SSO 或文档级 ACL。扫描 PDF 页面会标记为需要 OCR。Agent 检索受次数、步骤和总时限限制。

已有真实验证采用 8 份官方文档、746 个分块。**一万份文档是容量设计目标，尚不是已验证的容量结论。** 2026-11-19 的 20 路并发属于升级前回答流程，不能作为当前思考模式 Agent 的延迟基线；新版 Agent 的完整并发重测、方案 B 全链路验收、容器实际运行及业务专家质量验收仍待完成。
