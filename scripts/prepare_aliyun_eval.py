"""Build and validate the real-document retrieval fixture without network or model calls."""
import hashlib
import json
from pathlib import Path

from compare_retrieval import prepare_corpus
from rag.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "test-documents/aliyun"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    corpus = []
    for document in manifest["documents"]:
        raw = (source / document["markdown_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != document["markdown_sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {document['id']}")
        corpus.append({"id": document["id"], "text": raw.decode("utf-8")})
    facts = json.loads((ROOT / "evals/aliyun/facts.json").read_text(encoding="utf-8"))
    questions = [{"id": f"aliyun-{i + 1:02d}-{j + 1}", "document_id": fact["document_id"],
                  "question": question, "evidence": fact["evidence"], "answer": fact["answer"]}
                 for i, fact in enumerate(facts) for j, question in enumerate(fact["questions"])]
    rows, _, _ = prepare_corpus(corpus, questions, Settings())
    output = ROOT / "output/aliyun-live"
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("corpus", corpus), ("questions", questions)):
        (output / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"documents": len(corpus), "chunks": len(rows), "questions": len(questions),
                      "distinct_evidence_facts": len(facts), "hashes_and_evidence_validated": True}))


if __name__ == "__main__":
    main()
