"""Explicit test doubles; never imported by the production package."""
import asyncio
import hashlib
import json
import re

from rag.errors import AppError
from rag.providers import EmbeddingBatch
from rag.schemas import MetadataCard, ReplacementOutput, RewriteOutput


class FakeStorage:
    def __init__(self):
        self.objects = {}

    def configured(self):
        pass

    async def put(self, key, data, content_type="application/octet-stream"):
        self.objects[key] = data

    async def get(self, key):
        return self.objects[key]

    async def delete(self, keys):
        for key in keys:
            self.objects.pop(key, None)

    async def check(self):
        pass

    async def close(self):
        pass


def tokens(text):
    return set(re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower()))


class FakeModels:
    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.calls = []
        self.embedding_failure = False
        self.answer_text = None
        self.delay = 0

    def configured(self):
        pass

    async def structured(self, purpose, system, content, schema, validator=None):
        self.calls.append((purpose, content))
        if schema is ReplacementOutput:
            block = content["current_block"]
            text = block["text"]
            evidence = next((b for b in content["context"] if "可乐是一种饮料" in b["text"]), None)
            if evidence and "它" in text:
                start = text.index("它")
                return ReplacementOutput.model_validate({"replacements": [{"block_id": block["block_id"],
                    "start": start, "end": start + 1, "original": "它", "target": "可乐",
                    "evidence_block_id": evidence["block_id"], "evidence": "可乐是一种饮料", "unambiguous": True}]})
            return ReplacementOutput(replacements=[])
        if schema is MetadataCard:
            value = MetadataCard(summary=content["body"][:200], keywords=["测试资料"], time_mentions=[], locations=[])
            return validator(value) if validator else value
        if schema is RewriteOutput:
            return RewriteOutput(question=content["question"])
        raise AssertionError(schema)

    async def embed(self, texts, query=False):
        if self.embedding_failure:
            raise AppError("provider_unavailable", "模拟嵌入服务故障", 502)
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in tokens(text):
                index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimension
                vector[index] += 1
            vectors.append(vector)
        return vectors

    async def agent_step(self, messages, tools):
        from rag.agent import is_continuation
        data = json.loads(messages[1]["content"])
        self.calls.append(("agent", data))
        available = {t["function"]["name"] for t in tools}
        target = next((t for t in data["history"] if t["turn_id"] == data["continuation_target_id"]), None)
        question = data["question"]
        if target:
            question = target.get("resolved_question") or target["question"]
        elif data["history"] and len(question) < 30:
            question = (data["history"][-1].get("resolved_question") or data["history"][-1]["question"]) + " " + question
        if "answer" in available:
            name, args = "answer", {"question": question, "mode": "continue" if target else "answer",
                                    "source_turn_id": target["turn_id"] if target else None}
        elif "assess_evidence" in available:
            result = next(json.loads(m["content"]) for m in reversed(messages)
                          if m["role"] == "tool" and "evidence" in json.loads(m["content"]))
            ids = [h["chunk_id"] for h in result["evidence"]]
            name, args = "assess_evidence", {"question": question, "verdict": "sufficient" if ids else "irrelevant",
                "relevant_chunk_ids": ids, "missing_aspects": [] if ids else ["回答所需资料"]}
        elif target and messages[-1]["role"] != "tool":
            name, args = "reuse_evidence", {"turn_id": target["turn_id"]}
        elif is_continuation(question) and not data["history"]:
            name, args = "ask_clarification", {"question": "请说明要继续的主题。"}
        else:
            searches = sum(m["role"] == "assistant" and m["tool_calls"][0]["function"]["name"] == "search_knowledge_base"
                           for m in messages)
            name, args = "search_knowledge_base", {"query": question + (" 补充资料" if searches >= 3 else ""),
                "strategy": ["hybrid", "dense", "sparse"][searches % 3]}
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "test-call-" + str(len(messages)),
                "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}

    async def rerank(self, query, documents):
        query_tokens = tokens(query)
        return sorted([(i, len(query_tokens & tokens(doc)) / max(1, len(query_tokens)))
                       for i, doc in enumerate(documents)], key=lambda x: (-x[1], x[0]))

    async def embed_vectors(self, texts, *, query=False, include_sparse=False):
        dense = await self.embed(texts, query=query)
        return EmbeddingBatch(dense, [{i: value for i, value in enumerate(vector) if value}
                                     for vector in dense] if include_sparse else [])

    async def stream_answer(self, messages, *, max_tokens=None):
        data = json.loads(messages[-1]["content"])
        self.calls.append(("answer", {**data, "max_tokens": max_tokens}))
        answer = self.answer_text or f"根据资料，{data['evidence'][0]['body']} [1]"
        for part in [answer[i:i + 12] for i in range(0, len(answer), 12)]:
            await asyncio.sleep(self.delay)
            yield part

    async def close(self):
        pass


class FakeIndex:
    def __init__(self):
        self.rows = {}
        self.searches = []
        self.fail_verify = False

    def configured(self):
        pass

    async def initialize(self):
        pass

    async def check(self):
        pass

    async def upsert(self, rows):
        for row in rows:
            self.rows[row["chunk_id"]] = row.copy()

    async def verify(self, ids):
        if self.fail_verify or not set(ids) <= self.rows.keys():
            raise AppError("index_incomplete", "模拟索引校验失败", 502)

    async def search(self, question, vector, kb_ids, limit, *, model_sparse=None, sparse_mode=None, strategy="hybrid"):
        mode = sparse_mode or ("model" if model_sparse is not None else "bm25")
        self.searches.append({"kb_ids": kb_ids, "limit": limit, "sparse_mode": mode, "strategy": strategy})
        docs = [r for r in self.rows.values() if r["kb_id"] in kb_ids]
        dense = sorted([{"chunk_id": r["chunk_id"], "score": sum(a * b for a, b in zip(vector, r["dense"]))}
                        for r in docs], key=lambda r: -r["score"])[:limit]
        sparse = sorted([{"chunk_id": r["chunk_id"], "score": float(len(tokens(question) & tokens(r["body"])))}
                         for r in docs], key=lambda r: -r["score"])[:limit]
        if mode == "model" and strategy != "dense":
            sparse = sorted([{"chunk_id": r["chunk_id"], "score": sum(value * r["model_sparse"].get(i, 0)
                            for i, value in model_sparse.items())} for r in docs], key=lambda r: -r["score"])[:limit]
        paths = {"dense": dense, "sparse": sparse}
        return paths if strategy == "hybrid" else {strategy: paths[strategy]}

    async def delete_document(self, document_id):
        self.rows = {k: v for k, v in self.rows.items() if v["document_id"] != document_id}

    async def delete_versions(self, version_ids):
        self.rows = {k: v for k, v in self.rows.items() if v["version_id"] not in version_ids}

    async def close(self):
        pass
