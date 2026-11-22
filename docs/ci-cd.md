# CI 与镜像交付

仓库使用 GitHub Actions。自动检查在云端执行，不依赖开发者电脑上的 `.env`、运行进程或下载快照；不会调用 DeepSeek、Qwen、业务 PostgreSQL、Milvus 或 S3。

## 工作流

| 工作流 | 触发方式 | 作用 |
|---|---|---|
| [CI](../.github/workflows/ci.yml) | 推送 `main`、面向 `main` 的 PR、手动运行、被交付流程调用 | 后端、前端、PostgreSQL、容器检查与统一结果 |
| [Publish image](../.github/workflows/publish-image.yml) | 在 Actions 页面手动运行，选择 `main` | 完整检查成功后发布提交对应的 GHCR 镜像 |

[查看 Actions](https://github.com/Luck-dog12/zhiheng-rag/actions)。私有仓库的运行日志、构建产物和徽章需要相应访问权限。

## 已验证的运行

2026-11-22 复核：首次 CI 与镜像交付已在 GitHub runner 上实际完成，源码提交为 `ebbdf82b4756f7d930bdd6a6201b8a7056d1b4f9`。

- [CI 运行记录](https://github.com/Luck-dog12/zhiheng-rag/actions/runs/44890292734)：Backend、Frontend、PostgreSQL、Container 和 `CI required` 全部成功。
- [镜像交付记录](https://github.com/Luck-dog12/zhiheng-rag/actions/runs/44890532187)：重新检查同一提交后，GHCR 推送成功。
- 已发布镜像：`ghcr.io/luck-dog12/zhiheng-rag:sha-ebbdf82b4756f7d930bdd6a6201b8a7056d1b4f9`。
- 不可变引用：`ghcr.io/luck-dog12/zhiheng-rag@sha256:3744fa0d2f79820ffda97e9e74deadeb5523b137cec656d979a5053628b6107b`。
- 匿名拉取授权检查返回 HTTP 401；需要有权限的登录凭证才能拉取。没有向实际应用服务器部署。

上述结果只对应列明的提交；后续提交以各自 Actions 状态为准。

## CI 检查内容

- **Backend**：Python 3.12、与 Dockerfile 一致的 uv 0.9.26；`uv sync --locked` 校验依赖锁，Ruff，134 项不依赖外部服务的后端测试（数量随代码变化）。
- **Frontend**：Node.js 24，与 Dockerfile 一致；`npm ci`、Vitest、TypeScript 检查、Vite 生产构建。
- **PostgreSQL**：GitHub runner 临时 PostgreSQL 16 服务；验证 Alembic 升级、模型差异检查、降级、再次升级，运行 3 项真实数据库锁、并发领取和租约恢复测试。数据库凭证只用于当次可销毁测试容器，不是生产凭证。
- **Container**：多阶段 Docker 构建；以非 root 用户启动镜像，禁用容器外部网络；检查 `/api/v1/health/live`、首页和编译后的 JS/CSS、Worker CLI，以及镜像内没有本地 `.env`、交接文档或运行产物。
- **CI required**：所有检查都成功才通过，失败、取消或跳过均不能被当作成功。适合作为分支保护要求的检查。

后端、前端和 PostgreSQL 的 JUnit 报告以及前端构建产物保存 7 天。只上传这些明确指定的文件，不上传整个 `output/`。执行超时与新提交取消旧 CI 用于限制无效运行。

临时数据库只用于 CI；应用生产配置仍只连接外部服务。容器冒烟仅验证镜像可运行及界面打包，**不等于真实模型问答、依赖就绪、20 路并发或生产验收**。CI 不安装浏览器；本机浏览器验证继续使用 Edge。

## 持续交付：发布容器镜像

1. 打开仓库 **Actions → Publish image → Run workflow**，分支选择 `main`。
2. 工作流为选定的同一个提交重新执行 CI。失败时发布任务不会运行。
3. 成功后，使用 GitHub 自动提供的短期 `GITHUB_TOKEN` 发布镜像，不需要上传个人 PAT 或本地服务密钥。
4. 工作流摘要给出部署引用，形式为 `ghcr.io/luck-dog12/zhiheng-rag@sha256:...`。同时保留 `sha-<完整提交 SHA>` 标签用于定位源码。实际部署与回滚优先使用不可变的 digest。

镜像标签与提交关联，不推送浮动的 `latest`。发布不会运行生产数据库迁移、重启服务或修改外部索引。仓库与包的访问策略由 GitHub 控制，不在工作流中改成公开。

默认工作流只有 `contents: read`；只有发布镜像的任务拥有 `packages: write`。所有第三方 Action 固定到完整 commit SHA，升级时核对上游版本并更新 SHA。checkout 不保留 Git 凭证；不使用 `pull_request_target` 执行 PR 代码。

## 分支保护与部署接入

当前工作流会报告检查结果；要**阻止检查失败的代码合并**，仓库管理员还需在 GitHub Rulesets / Branch protection 中将 `CI required` 设为必需检查，并要求通过 PR 合并。此规则是否可用取决于账号方案和私有仓库权限。本次不自动改变协作规则。

还未确定实际部署环境，所以当前完成到容器交付，不声称已自动部署。后续接入时需要明确：

- 目标服务器、云平台或 Kubernetes，以及访问方式。
- 测试与生产域名、HTTPS/SSE 代理配置、服务端密钥注入位置。
- 迁移执行方式、发布前备份、`/api/v1/health/ready` 检查、失败回滚和版本保留。

现有 [compose.yaml](../compose.yaml) 用于在目标机器从源码构建。使用 GHCR 镜像时，应改为固定 digest 的 `image` 部署配置；生产 `.env` 留在目标环境或密钥管理系统，不作为 Actions artifact 或镜像内容。

## 本地复核

```powershell
uv sync --locked
uv run --no-sync ruff check backend tests scripts
uv run --no-sync pytest -q -m "not postgres"
Set-Location frontend
npm ci
npm test
npm run build
```

容器检查依赖 Docker，在没有 Docker 的开发电脑上由 GitHub runner 执行。本地有 Docker 时可以参照 `ci.yml` 的 `container` 任务；`scripts/smoke_container.py` 设计为在 Linux 应用容器内部运行，不直接在 Windows 主机执行。

GitHub 官方参考：[PostgreSQL 服务容器](https://docs.github.com/en/actions/tutorials/use-containerized-services/create-postgresql-service-containers)、[容器镜像发布](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)；uv 官方参考：[GitHub Actions 集成](https://docs.astral.sh/uv/guides/integration/github/)。
