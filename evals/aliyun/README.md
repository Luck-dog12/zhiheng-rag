# 阿里云真实文档检索评测

**本文命令执行方案 A + 方案 B 的检索与重排序专项对比。** 2026-11-18 已完成两方案测试，结果见 [A/B 报告](../../docs/rag-live-test-2026-11-18.md)。后续 2026-11-19 的完整应用问答与 20 路并发验收使用方案 A，见 [方案 A 端到端报告](../../docs/rag-e2e-live-2026-11-19.md)；方案 B 尚未单独完成同等全链路验收。

语料为 `test-documents/aliyun/documents/` 的 8 篇官方文档快照，包含自然语言、参数表、代码和错误码。`facts.json` 人工标注 25 条事实，每条包含两个问法，共 50 问；包含预期答案和逐字原文证据。两种问法相关，不应当作 50 条独立事实进行统计推断。

首次克隆需先下载本地快照；已有历史快照时不要为复核旧报告而重新下载。当前标注针对 2026-11-17 的快照，新版官方内容可能变化，须先通过哈希与逐字证据校验，必要时更新标注后重新建立基线。

准备及执行：

```powershell
# 首次准备；会覆盖已有下载快照
uv run --with markdownify python scripts/download_aliyun_docs.py
uv run python scripts/prepare_aliyun_eval.py
uv run python scripts/compare_retrieval.py --corpus output/aliyun-live/corpus.json --questions output/aliyun-live/questions.json --output output/aliyun-live/retrieval-comparison.json
```

准备步骤只读本地快照，校验原文件哈希及证据能否保留在分块中。实际比较调用 `.env` 配置的 Qwen 嵌入、Qwen 重排序和远程 Milvus，会产生供应商费用。结果及每条查询的各路候选、融合排序、最终命中均保存在指定报告内。随机命名的 `rag_ab_<uuid>` 集合保留用于复核，不覆盖应用集合。

- A：Qwen 稠密向量 + Milvus 内置中文 BM25。
- B：相同稠密向量 + Qwen 模型稀疏向量（IP）。
- 两路各取 50，RRF k=60，最多 100 个候选，重排序取前 5。
- 一次查询嵌入调用返回 dense&sparse，供 A/B 共用；每个问题交替执行 A/B 顺序。
- 相关性采用严格的“指定文档 + 逐字证据”匹配。同义证据或其他文档的等价段落不计分，需人工复核假阴性。
- 同一标注事实在多个块出现时仅按最先命中的位置得分，nDCG@5 使用单条标注事实的倒数对数排名。

这是检索专项测试，不执行文档指代消解、元数据生成、PostgreSQL 业务持久化、权限校验或答案生成，不能代替完整应用端到端验收。顺序查询耗时也不是 20 路并发的性能基线。文档内容以下载快照为准，后续网站更新不自动改变本次证据。
