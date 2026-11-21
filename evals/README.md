# 检索质量基准

## 已完成测试的方案归属

| 日期与测试 | 方案 A：Qwen 稠密 + Milvus 内置 BM25 | 方案 B：Qwen 稠密 + Qwen 稀疏 |
|---|---|---|
| 2026-11-18，8 份官方文档的检索与重排序 A/B | 已完成 | 已完成 |
| 2026-11-19，应用完整入库、问答、权限、版本、删除与 Edge 流程 | 已完成 | 尚未单独验收 |
| 2026-11-19，完整语料下 20 路并发问答 | 已完成 | 尚未单独验收 |

详见 [A/B 检索专项报告](../docs/rag-live-test-2026-11-18.md) 和 [方案 A 端到端报告](../docs/rag-e2e-live-2026-11-19.md)。方案由 API 启动时的 `RETRIEVAL_SPARSE_MODE` 决定：`bm25` 为 A，`model` 为 B。`EMBEDDING_STORE_SPARSE=true` 只表示入库时存储模型稀疏向量，不表示当前查询使用 B。

## 合成语料与应用评测

`corpus.json` 是 10 份专门编写的合成企业文档，`questions.json` 包含 50 条问题、相关文档和逐字证据；`unanswerable.json` 另含 5 条资料不足问题。它们不是实际企业数据。

`uv run pytest tests/test_evaluation.py -q` 验证评分流程和 20 路并发下的应用逻辑，将报告保存到 `output/evaluation-simulated.json`。该模式使用测试替身，不衡量 Qwen、DeepSeek、Milvus 或 PostgreSQL 的真实质量与性能。

真实评测：配置外部服务、启动 API 与 Worker 后，设置 `EVAL_BASE_URL`、`EVAL_USERNAME`、`EVAL_PASSWORD`，执行：

```powershell
uv run python scripts/evaluate_live.py --ingest --answers --concurrency 20
```

此命令会在专门的“RAG 合成评测”知识库中上传合成文档、等待入库并调用真实模型，会产生模型费用。返回问题级召回结果、证据覆盖、nDCG@5、引用出处命中、问答成功率和 P50/P95。额外执行 5 条资料不足问题，并记录是否出现拒答措辞。

应用评测通过当前 API 执行，不自动切换 A/B。测试另一方案时应修改配置并重启 API，另存测试状态与结果；不得把前一方案的断点恢复结果混入另一方案。下方 `compare_retrieval.py` 则会明确分别执行 A、B 两种检索路径。

引用文档命中和拒答措辞只是自动检查指标，不代表事实正确或回答确实未编造。脚本保存预期答案、逐字证据、实际答案和引用，需人工复核引用是否支持每个结论、资料冲突处理以及拒答质量。加入真实业务样本后才适合确定生产阈值。

## 两种混合检索方案对比

已适配 `qwen3.7-text-embedding` 和 `qwen3.7-text-rerank`。对比方案 A：Qwen 稠密 + Milvus 原生 BM25；方案 B：同一 Qwen 稠密 + Qwen 模型稀疏向量（Milvus IP 检索）。两者均使用每路 50、RRF k=60、最多 100 个候选和相同的 Qwen 重排序模型。

官方文档：[嵌入稠密/稀疏输出](https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api)、[重排序接口](https://help.aliyun.com/en/model-studio/text-rerank-api)、[Milvus 内置 BM25](https://milvus.io/docs/full-text-search.md)、[Milvus 模型稀疏向量 IP 检索](https://milvus.io/docs/sparse_vector.md)。嵌入稀疏输出需要 DashScope 原生接口；不把模型稀疏向量写入 BM25 自动生成字段。

配置 `.env` 的嵌入、重排序和 Milvus 后运行：

```powershell
uv run python scripts/compare_retrieval.py
```

脚本先检查并创建随机命名的独立 `rag_ab_<uuid>` 集合，使用现有 10 份合成文档、50 条带原文证据的问题。不会修改生产集合，也不需要 PostgreSQL、S3、指代消解、元数据或回答模型；这是检索专项实验，不能替代完整入库与问答验收。该集合保留用于复核。

两种方案使用完全相同的解析、清洗及分块结果，正文和问题各调用一次 `dense&sparse` 以共享稠密向量；逐题交替执行 A/B 顺序。报告 `output/retrieval-comparison.json` 记录数据哈希、模型与参数、各路原始排名、RRF 顺序、重排序来源、证据级召回、nDCG@5、MRR@5、阶段 P50/P95 以及逐题胜负。失败时保存阶段及脱敏错误，不输出虚构分数。

默认小语料的分块数低于每路召回数，候选召回率容易饱和，应重点查看融合排序差异；重排序也可能消除差异。报告中的总耗时包含共享的稠密+稀疏嵌入调用，不能当作独立稠密模式的延迟。一次顺序测试也不能作为 20 路容量基线。可用 `--corpus path/to/corpus.json --questions path/to/questions.json --output output/custom-comparison.json` 接入更大业务标注集，格式沿用现有评测文件。

## 万份文档容量测试

先在本地生成预检，不连接任何外部服务：

```powershell
uv run python scripts/capacity_live.py --dry-run --documents 10000 --run-id acceptance-10k
```

准备外部服务、管理员测试账号以及独立 Worker 后，设置 `EVAL_BASE_URL`、`EVAL_USERNAME`、`EVAL_PASSWORD`，再执行：

```powershell
uv run python scripts/capacity_live.py --documents 10000 --characters 2400 --run-id acceptance-10k --chat --wait-minutes 120
```

此命令实际上传 10,000 份合成文件，每份约 2,400 字符；会产生入库模型、存储和索引费用。只使用独立的 `RAG 容量测试 acceptance-10k` 知识库。成功入库后，从整个测试集均匀抽取最多 20 份文档并发提问，报告成功率、P50/P95 和引用文档命中。小样本仅用于验证工具时，可设置 `--documents 20`。

状态保存在 `output/capacity/<run-id>.jsonl`，报告保存在同目录的 `<run-id>.report.json`。中断后使用相同 run-id、文档数、字符数重跑，会跳过已记录上传的文件；单次上传响应丢失时由服务端文件哈希去重。已失败的入库任务会列在报告中，不自动反复重试。修复失败任务后应使用新 run-id 重做验收；测试资料保留用于复核，可通过文档管理页或删除文档 API 清理。

等待时间到期时，未完成任务保留为 pending，进程返回非零状态，不会记为成功。进度日志中的入库耗时包含状态轮询延迟。报告指标需要结合实际服务配额和服务器资源判断，不能据合成文档直接声称达到实际业务容量要求。

## 外部 PostgreSQL 专项测试

这些测试只验证数据库事务和任务领取，模型、S3、Milvus仍使用替身，不产生模型调用。测试连接必须指向专用外部测试数据库，账号需要创建 schema 的权限：

```powershell
$env:RAG_RUN_POSTGRES_TESTS = '1'
$env:TEST_POSTGRES_URL = 'postgresql+asyncpg://test-user:password@external-host:5432/rag_test'
uv run pytest tests/integration/test_postgres_worker.py -q
```

每项测试创建随机命名的 `rag_acceptance_<uuid>` schema，结束后只删除本次创建的 schema，不操作已有业务表。覆盖锁定任务被 `SKIP LOCKED` 跳过、20 个领取请求不重复、过期租约被恢复且旧 Worker 无法发布。默认 `pytest` 未显式启用时显示三项 skipped，不能将 skipped 当成已通过的真实数据库验收。
