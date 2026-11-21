import io

import pytest
from docx import Document
from rag.errors import AppError
from rag.processing import (
    Block,
    apply_replacements,
    clean_block,
    make_chunks,
    parse_document,
    resolve_document,
    split_text,
    validate_metadata,
)
from rag.schemas import MetadataCard, ReplacementOutput

from tests.fakes import FakeModels


def test_conservative_cleaning_preserves_facts_and_meaningful_ellipsis():
    block = clean_block(Block("b1", "嗯，不能服用吗啡。价格 3.50 元，重量 20 kg。\n\n\n他说：‘也许……’。"))
    assert block.text == "不能服用吗啡。价格 3.50 元，重量 20 kg。\n\n他说：‘也许……’。"
    assert block.original.startswith("嗯，")
    assert block.edits[0]["before"] == block.original


@pytest.mark.parametrize("kind", ["code", "quote", "table"])
def test_protected_regions_not_aggressively_cleaned(kind):
    text = "嗯，  它……\n\n\n  价格 10.50"
    assert clean_block(Block("b", text, kind)).text == text


async def test_cross_paragraph_coreference_has_original_evidence():
    parsed = parse_document("cola.md", "# 产品\n\n可乐是一种饮料。\n\n它的容量是 330 ml。".encode())
    await resolve_document(parsed, FakeModels())
    assert parsed.blocks[-1].text == "可乐的容量是 330 ml。"
    assert parsed.blocks[-1].original == "它的容量是 330 ml。"
    assert parsed.blocks[-1].edits[-1]["evidence"] == "可乐是一种饮料"


@pytest.mark.parametrize("change", [{"unambiguous": False}, {"target": "雪碧"}, {"evidence": "不存在的原文"}, {"start": 2}, {"block_id": "other"}])
def test_invalid_replacements_leave_original(change):
    prior = clean_block(Block("a", "可乐是一种饮料。"))
    block = clean_block(Block("b", "它很好喝。"))
    patch = {"block_id": "b", "start": 0, "end": 1, "original": "它", "target": "可乐",
             "evidence_block_id": "a", "evidence": "可乐是一种饮料", "unambiguous": True, **change}
    apply_replacements(block, ReplacementOutput.model_validate({"replacements": [patch]}), [prior, block])
    assert block.text == "它很好喝。"


def test_chunk_heading_boundaries_and_provenance():
    parsed = parse_document("test.md", "# A\n\n第一部分正文。\n\n# B\n\n第二部分正文。".encode())
    pieces = make_chunks([clean_block(b) for b in parsed.blocks], 1200, 2000, 150)
    assert len(pieces) == 2
    assert pieces[0].heading_path == ["A"] and pieces[1].heading_path == ["B"]
    assert pieces[0].sources[-1]["original"] == "第一部分正文。"


def test_sentence_overlap_is_bounded_and_no_gaps():
    text = "句子甲。句子乙。句子丙。句子丁。句子戊。"
    spans = split_text(text, 12, 4)
    assert all(b - a <= 12 for a, b in spans)
    assert set(range(len(text))) == {i for a, b in spans for i in range(a, b)}
    assert all(spans[i][1] - spans[i + 1][0] <= 4 for i in range(len(spans) - 1))


def test_very_long_sentence_has_hard_bound():
    spans = split_text("汉" * 5100, 2000, 150)
    assert spans == [(0, 2000), (2000, 4000), (4000, 5100)]


def test_many_sentence_boundaries_keep_coverage_and_overlap():
    text = "设备检查完成。" * 50000
    spans = split_text(text, 2000, 150)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(0 < end - start <= 2000 and text[end - 1] == "。" for start, end in spans)
    assert all(0 <= spans[i][1] - spans[i + 1][0] <= 150 for i in range(len(spans) - 1))


def test_long_table_repeats_header():
    text = "| 产品 | 价格 |\n| --- | --- |\n" + "\n".join(f"| 产品{i} | 100元 |" for i in range(30))
    pieces = make_chunks([clean_block(Block("t", text, "table"))], 80, 120, 10)
    assert len(pieces) > 1
    assert all(p.body.startswith("| 产品 | 价格 |\n| --- | --- |") and len(p.body) <= 120 for p in pieces)


def test_oversized_table_separator_fails_without_looping():
    text = "| 产品 |\n|" + "-" * 150 + "|\n| 可乐 |"
    with pytest.raises(AppError) as error:
        make_chunks([clean_block(Block("t", text, "table"))], 80, 120, 10)
    assert error.value.code == "table_header_too_large"


@pytest.mark.parametrize("maximum,overlap", [(0, 0), (-1, 0), (2, 2)])
def test_split_rejects_nonprogressing_parameters(maximum, overlap):
    with pytest.raises(ValueError):
        split_text("文本", maximum, overlap)


def test_metadata_rejects_invented_evidence():
    card = MetadataCard(summary="摘要", keywords=[], locations=[{"raw": "北京", "evidence": "北京总部"}])
    with pytest.raises(AppError, match="原文证据"):
        validate_metadata(card, "上海总部")


def test_relative_date_never_inherits_today():
    card = MetadataCard(summary="摘要", keywords=["复盘"], time_mentions=[
        {"raw": "明年", "evidence": "明年复盘", "normalized": "2027-01-01"}])
    assert validate_metadata(card, "明年复盘").time_mentions[0].normalized is None


def test_docx_retains_table_in_reading_order():
    doc = Document()
    doc.add_heading("产品", 1)
    doc.add_paragraph("第一段。")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "产品", "价格"
    table.cell(1, 0).text, table.cell(1, 1).text = "可乐", "5元"
    doc.add_paragraph("最后一段。")
    file = io.BytesIO()
    doc.save(file)
    parsed = parse_document("product.docx", file.getvalue())
    assert [b.kind for b in parsed.blocks] == ["heading", "paragraph", "table", "paragraph"]
    assert "可乐" in parsed.blocks[2].original


def test_binary_or_unsupported_file_rejected():
    with pytest.raises(AppError):
        parse_document("bad.pdf", b"not a pdf")
    with pytest.raises(AppError):
        parse_document("workbook.xlsx", b"no")


def test_cleaning_rules_can_be_disabled():
    text = "嗯，可乐  \n\n\n价格 5 元。"
    assert clean_block(Block("b", text), normalize_whitespace=False, remove_fillers=False).text == text


async def test_long_coreference_windows_rebase_offsets_and_stay_bounded():
    text = "前" * 3990 + "。可乐是一种饮料。它的价格是5元。" + "后" * 4500
    parsed = parse_document("long.txt", text.encode())
    models = FakeModels()
    await resolve_document(parsed, models)
    assert "可乐的价格是5元" in parsed.blocks[0].text
    assert all(len(c["current_block"]["text"]) <= 4000 for p, c in models.calls if p == "coreference")
    assert parsed.blocks[0].edits[-1]["start"] == text.index("它")
    for purpose, content in models.calls:
        if purpose == "coreference":
            window = content["current_block"]["text"]
            assert content["mention_candidates"]
            assert all(window[m["start"]:m["end"]] == m["original"] for m in content["mention_candidates"])


def make_pdf(streams: list[bytes]) -> bytes:
    """Small valid PDF fixture: one page per content stream, no extra dependency."""
    font_id = 3 + 2 * len(streams)
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(streams)))
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {len(streams)} >>".encode()]
    for i, stream in enumerate(streams):
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + i * 2} 0 R >>".encode())
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output, offsets = b"%PDF-1.4\n", [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return output + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF".encode()


def test_mixed_pdf_preserves_page_and_warns_about_missing_text():
    parsed = parse_document("mixed.pdf", make_pdf([b"BT /F1 14 Tf 50 700 Td (Cola costs 5 yuan.) Tj ET", b""]))
    assert parsed.blocks[0].page == 1 and "Cola costs 5 yuan" in parsed.blocks[0].original
    assert len(parsed.warnings) == 1 and "第 2 页" in parsed.warnings[0]
    with pytest.raises(AppError) as error:
        parse_document("scan.pdf", make_pdf([b""]))
    assert error.value.code == "requires_ocr"
