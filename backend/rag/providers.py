import asyncio
import json
import math
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from rag.config import Settings
from rag.errors import AppError, GenerationInterrupted
from rag.telemetry import MODEL_ERRORS, STAGES, record_usage

T = TypeVar("T", bound=BaseModel)
QUERY_INSTRUCTION = "Given a question about enterprise documents, retrieve passages that answer the question."


@dataclass
class EmbeddingBatch:
    dense: list[list[float]]
    sparse: list[dict[int, float]]


class RemoteModels:
    """HTTP-only adapters. Never instantiate a local inference runtime."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds, connect=10),
            limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
            follow_redirects=False,
        )
        self.semaphore = asyncio.Semaphore(settings.model_concurrency)

    @asynccontextmanager
    async def _slot(self, purpose):
        async with self.semaphore:
            with STAGES.labels(f"model_{purpose}").time():
                yield

    def configured(self):
        required = ["deepseek_base_url", "deepseek_api_key", "deepseek_model",
                    "embedding_endpoint" if self.settings.embedding_protocol == "dashscope" else "embedding_base_url",
                    "embedding_api_key", "embedding_model",
                    "rerank_endpoint", "rerank_api_key", "rerank_model"]
        if not self.settings.coreference_use_deepseek:
            required.extend(["coreference_base_url", "coreference_api_key", "coreference_model"])
        missing = []
        for name in required:
            value = getattr(self.settings, name)
            if not (value.get_secret_value() if hasattr(value, "get_secret_value") else value):
                missing.append(name.upper())
        if missing:
            raise AppError("missing_configuration", "缺少配置：" + ", ".join(missing), 503)

    def chat_config(self, purpose: str):
        prefix = "coreference" if purpose == "coreference" and not self.settings.coreference_use_deepseek else "deepseek"
        url = getattr(self.settings, prefix + "_base_url").rstrip("/")
        key = getattr(self.settings, prefix + "_api_key").get_secret_value()
        model = getattr(self.settings, prefix + "_model")
        if not url or not key or not model:
            raise AppError("missing_configuration", f"缺少 {prefix.upper()} 模型配置", 503)
        return url + "/chat/completions", key, model

    async def _post(self, url: str, key: str, body: dict, purpose: str) -> dict:
        if not url or not key:
            raise AppError("missing_configuration", f"缺少 {purpose} 接口配置", 503)
        async with self._slot(purpose):
            for attempt in range(3):
                try:
                    response = await self.client.post(url, headers={"Authorization": f"Bearer {key}"}, json=body)
                    if response.status_code in (429, 500, 502, 503, 504):
                        if attempt < 2:
                            await asyncio.sleep(min(2 ** attempt + random.random() / 4, 5))
                            continue
                    self._check_status(response, purpose)
                    result = response.json()
                    if not isinstance(result, dict):
                        raise ValueError("Expected object")
                    record_usage(purpose, result.get("usage") or {})
                    return result
                except (httpx.TimeoutException, httpx.NetworkError):
                    if attempt == 2:
                        MODEL_ERRORS.labels(purpose, "network").inc()
                        raise AppError("provider_unavailable", f"{purpose} 服务超时或网络不可用", 502) from None
                    await asyncio.sleep(2 ** attempt)
                except (ValueError, TypeError):
                    raise AppError("provider_protocol", f"{purpose} 返回了无效 JSON", 502) from None
        raise AppError("provider_unavailable", "外部服务不可用", 502)

    @staticmethod
    def _check_status(response: httpx.Response, purpose: str):
        if response.is_error or response.is_redirect:
            MODEL_ERRORS.labels(purpose, str(response.status_code)).inc()
            code = "provider_auth" if response.status_code in (401, 403) else "provider_unavailable"
            # Never propagate provider response bodies: they can contain prompts or credentials.
            raise AppError(code, f"{purpose} 服务返回 HTTP {response.status_code}", 502)

    async def structured(self, purpose: str, system: str, content: dict, schema: type[T],
                         validator: Callable[[T], T] | None = None) -> T:
        url, key, model = self.chat_config(purpose)
        messages = [
            {"role": "system", "content": system + "\n只输出 json 对象，严格匹配这个 JSON Schema：" +
             json.dumps(schema.model_json_schema(), ensure_ascii=False)},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ]
        output_budget = 4096
        for attempt in range(3):
            result = await self._post(url, key, {
                "model": model, "messages": messages, "response_format": {"type": "json_object"},
                "temperature": 0, "max_tokens": output_budget,
                **({"thinking": {"type": self.settings.coreference_deepseek_thinking}}
                   if purpose == "coreference" and self.settings.coreference_use_deepseek else {}),
            }, purpose)
            try:
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    # Reasoning tokens share the output budget. Repeating the same cap
                    # can exhaust all repairs without ever producing the JSON payload.
                    output_budget = min(output_budget * 2, 16384)
                if choice.get("finish_reason") not in ("stop", None):
                    raise ValueError("Incomplete generation")
                raw = choice["message"]["content"]
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("Empty generation")
                parsed = schema.model_validate_json(raw)
                return validator(parsed) if validator else parsed
            except (KeyError, IndexError, TypeError, ValueError, ValidationError, AppError):
                if attempt == 2:
                    raise AppError("invalid_model_output", f"{purpose} 输出在两次修复后仍不符合结构要求", 502) from None
                messages.append({"role": "user", "content": "上次返回无效、为空或被截断。请重新生成完整且符合 schema 的 json。"})
        raise AssertionError("unreachable")

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        return (await self.embed_vectors(texts, query=query)).dense

    async def embed_vectors(self, texts: list[str], *, query: bool = False,
                            include_sparse: bool = False) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch([], [])
        s = self.settings
        key = s.embedding_api_key.get_secret_value()
        url = s.embedding_endpoint if s.embedding_protocol == "dashscope" else s.embedding_base_url.rstrip("/") + "/embeddings"
        if not key or not (s.embedding_endpoint if s.embedding_protocol == "dashscope" else s.embedding_base_url):
            raise AppError("missing_configuration", "缺少 EMBEDDING 接口配置", 503)
        if include_sparse and s.embedding_protocol != "dashscope":
            raise AppError("sparse_not_supported", "模型稀疏向量需要 DashScope 原生接口", 503)
        combined = EmbeddingBatch([], [])
        for start in range(0, len(texts), s.embedding_batch_size):
            batch = texts[start:start + s.embedding_batch_size]
            if s.embedding_protocol == "dashscope":
                params = {"dimension": s.embedding_dimension, "text_type": "query" if query else "document",
                          "output_type": "dense&sparse" if include_sparse else "dense"}
                if query:
                    params["instruct"] = QUERY_INSTRUCTION
                body = {"model": s.embedding_model, "input": {"texts": batch}, "parameters": params}
            else:
                inputs = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in batch] if query else batch
                body = {"model": s.embedding_model, "input": inputs, "encoding_format": "float",
                        "dimensions": s.embedding_dimension}
            result = await self._post(url, key, body, "embedding")
            parsed = self._parse_embeddings(result, len(batch), include_sparse)
            combined.dense.extend(parsed.dense)
            combined.sparse.extend(parsed.sparse)
        return combined

    def _parse_embeddings(self, result: dict, count: int, include_sparse: bool) -> EmbeddingBatch:
        try:
            rows = result["output"]["embeddings"] if self.settings.embedding_protocol == "dashscope" else result["data"]
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError("Missing embeddings")
            index_key = "text_index" if rows and "text_index" in rows[0] else "index"
            if (any(type(x[index_key]) is not int for x in rows)
                    or sorted(x[index_key] for x in rows) != list(range(count))):
                raise ValueError("Missing or duplicated embedding indices")
            ordered = sorted(rows, key=lambda x: x[index_key])
            vectors = [row["embedding"] for row in ordered]
            for vector in vectors:
                if not isinstance(vector, list) or len(vector) != self.settings.embedding_dimension or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector
                ) or not any(v != 0 for v in vector):
                    raise ValueError("Invalid dimension or vector values")
            sparse = []
            if include_sparse:
                for row in ordered:
                    values = row["sparse_embedding"]
                    if not isinstance(values, list) or not values:
                        raise ValueError("Missing sparse vector")
                    converted = {}
                    for entry in values:
                        index, value = entry["index"], entry["value"]
                        if (type(index) is not int or not 0 <= index < 2 ** 32 - 1 or index in converted
                                or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                            raise ValueError("Invalid sparse entries")
                        converted[index] = float(value)
                    nonzero = {i: value for i, value in converted.items() if value > 0}
                    if not nonzero:
                        raise ValueError("Empty sparse vector")
                    sparse.append(nonzero)
            return EmbeddingBatch(vectors, sparse)
        except (KeyError, TypeError, ValueError):
            raise AppError("embedding_mismatch", "嵌入数量、索引、维度或稠密/稀疏数值与配置不一致", 502) from None

    async def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        s = self.settings
        body = {
            "model": s.rerank_model, "query": query, "documents": documents, "top_n": len(documents),
        }
        if s.rerank_protocol == "dashscope":
            body = {"model": s.rerank_model, "input": {"query": query, "documents": documents},
                    "parameters": {"top_n": len(documents)}}
        result = await self._post(s.rerank_endpoint, s.rerank_api_key.get_secret_value(), body, "rerank")
        try:
            payload = result["output"] if s.rerank_protocol == "dashscope" else result
            rows = [(row["index"], row["relevance_score"]) for row in payload["results"]]
            if len(rows) != len(documents) or sorted(i for i, _ in rows) != list(range(len(documents))):
                raise ValueError("Missing or duplicated ranking indices")
            if any(type(i) is not int or type(v) not in (float, int) or not math.isfinite(v) for i, v in rows):
                raise ValueError("Invalid scores")
            return sorted(rows, key=lambda pair: (-pair[1], pair[0]))
        except (KeyError, TypeError, ValueError):
            raise AppError("rerank_mismatch", "重排序返回的索引或分数无效", 502) from None

    async def agent_step(self, messages: list[dict], tools: list[dict]) -> dict:
        """Ask the shared DeepSeek model to select one validated server-side tool."""
        url, key, model = self.chat_config("agent")
        allowed = {t["function"]["name"] for t in tools}
        attempts = list(messages)
        thinking = self.settings.agent_deepseek_thinking
        for attempt in range(3):
            result = await self._post(url, key, {"model": model, "messages": attempts, "tools": tools,
                "tool_choice": "auto" if thinking == "enabled" else "required", "thinking": {"type": thinking},
                **({"reasoning_effort": self.settings.agent_reasoning_effort} if thinking == "enabled" else {}),
                "max_tokens": min(self.settings.agent_initial_tokens * (2 ** attempt), 32768)}, "agent")
            try:
                choice = result["choices"][0]
                message = choice["message"]
                preserved = {"role": "assistant", "content": message.get("content")}
                if thinking == "enabled":
                    if not isinstance(message.get("reasoning_content"), str):
                        raise ValueError("Missing reasoning state")
                    # Required by DeepSeek within the tool loop. Never log or expose this field.
                    preserved["reasoning_content"] = message["reasoning_content"]
                if choice["finish_reason"] == "stop" and not message.get("tool_calls"):
                    attempts.append(preserved)
                    raise ValueError("Use a validated terminal tool instead of ungrounded prose")
                calls = message["tool_calls"]
                if choice["finish_reason"] != "tool_calls" or len(calls) != 1:
                    raise ValueError("Expected exactly one completed tool call")
                call = calls[0]
                if (call["type"] != "function" or call["function"]["name"] not in allowed
                        or not isinstance(call["id"], str) or not call["id"]):
                    raise ValueError("Unknown tool")
                if not isinstance(json.loads(call["function"]["arguments"]), dict):
                    raise ValueError("Invalid arguments")
                return {**preserved, "tool_calls": calls}
            except (KeyError, IndexError, TypeError, ValueError):
                if attempt == 2:
                    raise AppError("agent_protocol", "对话决策未返回有效工具调用，请重试", 502) from None
                attempts.append({"role": "user", "content": "请仅调用一个已提供工具，输出完整合法的参数。"})
        raise AssertionError("unreachable")

    async def stream_answer(self, messages: list[dict], *, max_tokens: int | None = None) -> AsyncIterator[str]:
        url, key, model = self.chat_config("answer")
        async with self._slot("answer"):
            for attempt in range(3):
                started = False
                try:
                    async with self.client.stream("POST", url, headers={"Authorization": f"Bearer {key}"}, json={
                        "model": model, "messages": messages, "stream": True, "temperature": 0.2,
                        "max_tokens": max_tokens or self.settings.answer_initial_tokens,
                        "thinking": {"type": self.settings.answer_deepseek_thinking},
                        **({"reasoning_effort": self.settings.answer_reasoning_effort}
                           if self.settings.answer_deepseek_thinking == "enabled" else {}),
                        "stream_options": {"include_usage": True},
                    }) as response:
                        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        self._check_status(response, "answer")
                        stopped = False
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            item = json.loads(payload)
                            record_usage("answer", item.get("usage") or {})
                            for choice in item.get("choices", []):
                                reason = choice.get("finish_reason")
                                content = choice.get("delta", {}).get("content")
                                if content:
                                    started = True
                                    yield content
                                if reason and reason != "stop":
                                    raise GenerationInterrupted(reason)
                                stopped |= reason == "stop"
                        if not stopped:
                            raise GenerationInterrupted("stream_closed")
                        return
                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                    if started or attempt == 2:
                        raise GenerationInterrupted("network") from None
                    await asyncio.sleep(2 ** attempt)
                except (ValueError, TypeError):
                    raise AppError("provider_protocol", "模型流格式不符合协议", 502) from None

    async def close(self):
        await self.client.aclose()
