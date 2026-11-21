# 阿里云官方文档测试集

用于本项目文档解析、清洗、分块和检索测试。仓库保存来源清单与下载脚本；`documents/`、`raw/`、`manifest.json`、`validation.json` 是仅在本地保存的快照或报告，不随 Git 上传。首次克隆后先执行下方下载命令。

将 `documents/` 中的 Markdown 文件上传到知识库。`raw/` 保存原始网页；`manifest.json` 保存来源、下载时间、官方更新时间、文件大小及 SHA-256。

正文来自官方网页，未通过模型改写；去掉网页导航，转换为 Markdown。包含代码或过长表头的布局表格按原顺序展开为正文，普通数据表的合并单元格重复填充。保留代码示例，未执行示例中的代码或下载其引用资源。版权归原权利人。

| 文档 | Markdown 文件 | 官方来源 |
|---|---|---|
| 文本与多模态向量化 | `embedding.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/embedding) |
| 同步接口API详情 | `text-embedding-synchronous-api.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api) |
| 排序模型（Rerank） | `text-rerank-api.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/text-rerank-api) |
| 错误码 | `error-code.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/error-code) |
| 获取与配置 API Key | `get-api-key.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/get-api-key) |
| Base URL总览 | `base-url.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/base-url) |
| 生成临时API Key | `generate-temporary-api-key.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/generate-temporary-api-key) |
| OpenAI兼容-Chat | `compatibility-of-openai-with-dashscope.md` | [阿里云原文](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope) |

重新下载：`uv run --with markdownify python scripts/download_aliyun_docs.py`。会更新这些生成的快照和清单。

下载脚本仅准备测试文档，不上传知识库或调用模型。不同文档可能重复介绍相同参数，可用于检索去重和交叉引用测试。应用真实入库与问答的历史结果见 [方案 A 端到端报告](../../docs/rag-e2e-live-2026-11-19.md)。

2026-11-17 本地验证：8 篇全部通过项目的解析、清洗和分块检查，生成 746 个分块，每块不超过 2,000 字符；103 个代码块均被识别。文件哈希核验全部通过，详细结果保存在本地 `validation.json`（未随仓库上传）。该验证未调用指代消解、摘要或嵌入模型。

官方内容会更新，重新下载会覆盖生成的快照与清单。需要复核历史指标时应保留原快照；新快照须重新校验评测的逐字证据。
