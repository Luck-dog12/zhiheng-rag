"""Download public Alibaba Cloud documentation snapshots for local RAG tests.

Run with: uv run --with markdownify python scripts/download_aliyun_docs.py
No model, database, credentials, or browser is used.
"""
import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "test-documents/aliyun"
BASE = "https://help.aliyun.com/zh/model-studio/"
SOURCES = [
    "embedding", "text-embedding-synchronous-api", "text-rerank-api", "error-code",
    "get-api-key", "base-url", "generate-temporary-api-key", "compatibility-of-openai-with-dashscope",
]
CHINA_TIME = timezone(timedelta(hours=8))


def linearize_layout_tables(soup):
    """Keep long reference text and fenced examples out of a single Markdown table cell."""
    converted = 0
    for table in reversed(list(soup.find_all("table"))):
        rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
        if not rows or (not table.find("pre") and len(rows[0].get_text()) <= 800):
            continue
        replacement = soup.new_tag("div")
        note = soup.new_tag("p")
        note.string = "以下内容按原网页布局表格逐行展开，保留参数说明及代码示例。"
        replacement.append(note)
        for row in rows:
            for cell in row.find_all(["td", "th"], recursive=False):
                section = soup.new_tag("div")
                for child in list(cell.contents):
                    section.append(child.extract())
                replacement.append(section)
        table.replace_with(replacement)
        converted += 1
    return converted


def expand_merged_tables(soup):
    """Repeat merged-cell contents so each Markdown table row remains self-contained."""
    for table in list(soup.find_all("table")):
        if not table.find(attrs={"rowspan": True}) and not table.find(attrs={"colspan": True}):
            continue
        rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
        pending, expanded = {}, []
        for row in rows:
            grid = {column: cell for column, (cell, remaining) in pending.items()}
            pending = {column: (cell, remaining - 1) for column, (cell, remaining) in pending.items() if remaining > 1}
            column = 0
            for cell in row.find_all(["th", "td"], recursive=False):
                while column in grid:
                    column += 1
                width, height = int(cell.get("colspan", 1)), int(cell.get("rowspan", 1))
                if not 1 <= width <= 100 or not 1 <= height <= 1000:
                    raise ValueError("Unsupported table span")
                value = cell.decode_contents()
                for offset in range(width):
                    grid[column + offset] = value
                    if height > 1:
                        pending[column + offset] = (value, height - 1)
                column += width
            expanded.append(grid)
        columns = max((max(row, default=-1) + 1 for row in expanded), default=0)
        replacement = soup.new_tag("table")
        for number, row in enumerate(expanded):
            tr = soup.new_tag("tr")
            for column in range(columns):
                cell = soup.new_tag("th" if number == 0 else "td")
                fragment = BeautifulSoup(row.get(column, ""), "html.parser")
                for child in list(fragment.contents):
                    cell.append(child.extract())
                tr.append(cell)
            replacement.append(tr)
        table.replace_with(replacement)


def extract_document(page, source_url):
    # Parse the site's JSON data as data; never execute embedded JavaScript.
    marker = "window.__ICE_PAGE_PROPS__="
    offset = page.index(marker) + len(marker)
    props, _ = json.JSONDecoder().raw_decode(page[offset:].lstrip())
    document = props["docDetailData"]["storeData"]["data"]
    title, content = document["title"], document["content"]
    soup = BeautifulSoup(content, "html.parser")
    for node in soup.find_all(["script", "style", "meta", "link", "input", "button", "iframe"]):
        node.decompose()
    for node in soup.find_all(href=True):
        node["href"] = urljoin(source_url, node["href"])
    for node in soup.find_all(src=True):
        node["src"] = urljoin(source_url, node["src"])
    counts = {tag: len(soup.find_all(tag)) for tag in ("h2", "h3", "h4", "table", "pre")}
    counts["layout_tables_linearized"] = linearize_layout_tables(soup)
    expand_merged_tables(soup)

    def code_language(element):
        for name in element.get("class", []):
            if name.startswith("language-"):
                return name[len("language-"):]
        return ""

    body = MarkdownConverter(heading_style="ATX", code_language_callback=code_language,
                             table_infer_header=True, bullets="-").convert_soup(soup).strip()
    if len(body) < 500 or "\ufffd" in body:
        raise ValueError("Missing or incorrectly decoded document body")
    if len(re.findall(r"(?m)^```", body)) % 2:
        raise ValueError("Unbalanced code fences")
    modified = document.get("lastModifiedTime")
    updated = datetime.fromtimestamp(modified / 1000, CHINA_TIME).isoformat() if modified else None
    return title, body, updated, counts


async def download(client, slug, semaphore):
    url = BASE + slug
    async with semaphore:
        response = await client.get(url)
        response.raise_for_status()
    if response.url.host != "help.aliyun.com":
        raise ValueError("Unexpected download destination")
    title, body, updated, counts = extract_document(response.content.decode("utf-8"), url)
    fetched = datetime.now(CHINA_TIME).isoformat()
    metadata = (f"# {title}\n\n来源：阿里云帮助中心\n\n原文链接：{url}\n\n"
                f"下载时间：{fetched}\n\n官方更新时间：{updated or '页面未提供'}\n\n---\n\n")
    markdown = (metadata + body + "\n").encode("utf-8")
    raw_path, doc_path = DESTINATION / "raw" / (slug + ".html"), DESTINATION / "documents" / (slug + ".md")
    raw_path.write_bytes(response.content)
    doc_path.write_bytes(markdown)
    return {"id": slug, "title": title, "source_url": url, "resolved_url": str(response.url),
            "downloaded_at": fetched, "source_updated_at": updated, "http_status": response.status_code,
            "raw_file": str(raw_path.relative_to(DESTINATION)).replace("\\", "/"),
            "markdown_file": str(doc_path.relative_to(DESTINATION)).replace("\\", "/"),
            "raw_bytes": len(response.content), "markdown_bytes": len(markdown), "body_characters": len(body),
            "raw_sha256": hashlib.sha256(response.content).hexdigest(),
            "markdown_sha256": hashlib.sha256(markdown).hexdigest(), "source_structure": counts}


async def main():
    for directory in ("raw", "documents"):
        (DESTINATION / directory).mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=45, follow_redirects=True, headers={"User-Agent": "RAG-test-documents/1.0"}) as client:
        semaphore = asyncio.Semaphore(3)
        results = await asyncio.gather(*(download(client, slug, semaphore) for slug in SOURCES), return_exceptions=True)
    documents, failures = [], []
    for slug, result in zip(SOURCES, results):
        if isinstance(result, Exception):
            failures.append({"id": slug, "error_type": type(result).__name__})
        else:
            documents.append(result)
    manifest = {"purpose": "Public official documentation snapshots for local RAG testing", "documents": documents,
                "failures": failures, "conversion": "Body extracted from official page data; site navigation omitted; long layout tables linearized; merged data-table cells repeated; code blocks and source links preserved."}
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阿里云官方文档测试集", "", "用于本项目后续文档解析、清洗、分块和检索测试。", "",
             "将 `documents/` 中的 Markdown 文件上传到知识库。`raw/` 保存原始网页；`manifest.json` 保存来源、下载时间、官方更新时间、文件大小及 SHA-256。", "",
             "正文来自官方网页，未通过模型改写；去掉网页导航，转换为 Markdown。包含代码或过长表头的布局表格按原顺序展开为正文，普通数据表的合并单元格重复填充。保留代码示例，未执行示例中的代码或下载其引用资源。版权归原权利人。", "",
             "| 文档 | Markdown 文件 | 官方来源 |", "|---|---|---|"]
    for doc in documents:
        lines.append(f"| {doc['title']} | [{doc['id']}.md]({doc['markdown_file']}) | [阿里云原文]({doc['source_url']}) |")
    lines += ["", "重新下载：`uv run --with markdownify python scripts/download_aliyun_docs.py`。会更新这些生成的快照和清单。", "",
              "本次仅准备测试文档，尚未上传知识库或调用模型。不同文档可能重复介绍相同参数，可用于后续检索去重和交叉引用测试。", ""]
    (DESTINATION / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"downloaded": len(documents), "markdown_bytes": sum(d["markdown_bytes"] for d in documents),
                      "destination": str(DESTINATION), "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
