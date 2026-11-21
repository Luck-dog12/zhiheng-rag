import json

import httpx
import pytest
from rag.config import Settings
from rag.errors import AppError
from rag.providers import QUERY_INSTRUCTION, RemoteModels
from rag.schemas import MetadataCard, ReplacementOutput


def model_client(handler, **kwargs):
    settings = Settings(_env_file=None, embedding_base_url="https://models.example/v1", embedding_api_key="test",
                        deepseek_api_key="test", rerank_endpoint="https://models.example/rerank", rerank_api_key="test",
                        embedding_dimension=3, **kwargs)
    return RemoteModels(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize("shared", [True, False])
async def test_coreference_routes_to_selected_remote_model(shared):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"replacements":[]}'},
                                                      "finish_reason": "stop"}]})
    models = model_client(handler, coreference_use_deepseek=shared,
                          coreference_base_url="https://small.example/v1", coreference_api_key="small-key",
                          coreference_model="dedicated-small")
    try:
        models.configured()
        await models.structured("coreference", "仅返回指代建议", {}, ReplacementOutput)
        request = requests[0]
        assert str(request.url) == ("https://api.deepseek.com/chat/completions" if shared else
                                    "https://small.example/v1/chat/completions")
        assert json.loads(request.content)["model"] == ("deepseek-flash" if shared else "dedicated-small")
        assert request.headers["Authorization"] == ("Bearer test" if shared else "Bearer small-key")
        assert json.loads(request.content).get("thinking") == ({"type": "disabled"} if shared else None)
    finally:
        await models.close()


async def test_shared_coreference_needs_no_separate_credentials_but_dedicated_mode_does():
    models = model_client(lambda r: httpx.Response(200))
    try:
        models.configured()
        assert models.chat_config("coreference") == models.chat_config("metadata")
        models.settings.coreference_use_deepseek = False
        with pytest.raises(AppError, match="COREFERENCE"):
            models.configured()
        with pytest.raises(AppError, match="COREFERENCE"):
            models.chat_config("coreference")
    finally:
        await models.close()


def test_processing_fingerprint_tracks_effective_coreference_model():
    shared = Settings(_env_file=None)
    unused_override = Settings(_env_file=None, coreference_model="unused-model")
    dedicated = Settings(_env_file=None, coreference_use_deepseek=False, coreference_model="another-model")
    assert shared.processing_fingerprint == unused_override.processing_fingerprint
    assert shared.processing_fingerprint != dedicated.processing_fingerprint
    thinking = Settings(_env_file=None, coreference_deepseek_thinking="enabled")
    assert thinking.processing_fingerprint != shared.processing_fingerprint
    assert thinking.embedding_fingerprint == shared.embedding_fingerprint
    dedicated_thinking = dedicated.model_copy(update={"coreference_deepseek_thinking": "enabled"})
    assert dedicated_thinking.processing_fingerprint == dedicated.processing_fingerprint


@pytest.mark.parametrize("purpose", ["metadata", "rewrite"])
async def test_coreference_thinking_does_not_affect_other_structured_purposes(purpose):
    bodies = []
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"replacements":[]}'},
                                                     "finish_reason": "stop"}]})
    models = model_client(handle)
    try:
        await models.structured(purpose, "检查用途隔离", {}, ReplacementOutput)
        assert "thinking" not in bodies[0]
        models.settings.coreference_deepseek_thinking = "enabled"
        await models.structured("coreference", "检查显式配置", {}, ReplacementOutput)
        assert bodies[1]["thinking"] == {"type": "enabled"}
    finally:
        await models.close()


async def test_embedding_reorders_indices_and_query_only_instruction():
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 1, 0]}, {"index": 0, "embedding": [1, 0, 0]}]})
    models = model_client(handle)
    assert await models.embed(["甲", "乙"]) == [[1, 0, 0], [0, 1, 0]]
    await models.embed(["甲", "乙"], query=True)
    assert requests[0]["input"] == ["甲", "乙"]
    assert QUERY_INSTRUCTION in requests[1]["input"][0]
    assert requests[0]["model"] == requests[1]["model"]
    await models.close()


@pytest.mark.parametrize("data", [
    [{"index": 0, "embedding": [1, 2]}],
    [{"index": 1, "embedding": [1, 2, 3]}],
    [{"index": 0, "embedding": [0, 0, 0]}],
])
async def test_invalid_embedding_shape_rejected(data):
    models = model_client(lambda r: httpx.Response(200, json={"data": data}))
    with pytest.raises(AppError) as error:
        await models.embed(["文本"])
    assert error.value.code == "embedding_mismatch"
    await models.close()


async def test_rerank_uses_returned_indices_not_response_position():
    models = model_client(lambda r: httpx.Response(200, json={"results": [
        {"index": 2, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.5}]}))
    assert await models.rerank("问题", ["a", "b", "c"]) == [(1, 0.9), (0, 0.5), (2, 0.1)]
    await models.close()


async def test_rerank_duplicate_index_rejected():
    models = model_client(lambda r: httpx.Response(200, json={"results": [
        {"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}))
    with pytest.raises(AppError) as error:
        await models.rerank("问题", ["a", "b"])
    assert error.value.code == "rerank_mismatch"
    await models.close()


async def test_auth_error_not_retried_or_leaked():
    calls = []
    def handle(request):
        calls.append(1)
        return httpx.Response(401, json={"secret": "private-token"})
    models = model_client(handle)
    with pytest.raises(AppError) as error:
        await models.embed(["测试"])
    assert len(calls) == 1 and "private-token" not in str(error.value)
    await models.close()


async def test_429_retries_three_times(monkeypatch):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("rag.providers.asyncio.sleep", no_sleep)
    calls = []
    def handle(request):
        calls.append(1)
        return httpx.Response(429)
    models = model_client(handle)
    with pytest.raises(AppError):
        await models.embed(["测试"])
    assert len(calls) == 3
    await models.close()


async def test_json_output_two_repairs_then_failure():
    calls = []
    def handle(request):
        calls.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": " "}, "finish_reason": "stop"}]})
    models = model_client(handle)
    with pytest.raises(AppError) as error:
        await models.structured("metadata", "摘要", {}, MetadataCard)
    assert len(calls) == 3 and error.value.code == "invalid_model_output"
    await models.close()


async def test_truncated_stream_is_not_success():
    wire = 'data: {"choices":[{"delta":{"content":"片段"}}]}\n\ndata: [DONE]\n\n'
    models = model_client(lambda r: httpx.Response(200, text=wire))
    with pytest.raises(AppError) as error:
        _ = [part async for part in models.stream_answer([])]
    assert error.value.code == "generation_incomplete"
    await models.close()


async def test_answer_uses_shared_deepseek_explicit_budget_and_preserves_last_truncated_delta():
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"最后片段"},"finish_reason":"length"}]}\n\n')
    models = model_client(handler)
    parts = []
    try:
        with pytest.raises(AppError) as error:
            async for part in models.stream_answer([], max_tokens=16384):
                parts.append(part)
        assert parts == ["最后片段"] and error.value.reason == "length"
        assert bodies[0]["model"] == "deepseek-flash"
        assert bodies[0]["max_tokens"] == 16384 and bodies[0]["thinking"] == {"type": "enabled"}
    finally:
        await models.close()


async def test_agent_uses_function_calling_and_rejects_unknown_tools():
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]}}]})
    models = model_client(handler)
    try:
        with pytest.raises(AppError) as error:
            await models.agent_step([], [{"type": "function", "function": {"name": "search_knowledge_base"}}])
        assert error.value.code == "agent_protocol" and len(bodies) == 3
        assert all(b["model"] == "deepseek-flash" and b["tool_choice"] == "auto"
                   and b["thinking"] == {"type": "enabled"} for b in bodies)
    finally:
        await models.close()


async def test_thinking_agent_preserves_reasoning_across_tools_and_rejects_unvalidated_prose():
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "content": "不应直接发布的回答", "reasoning_content": "private-first"}}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": "", "reasoning_content": "private-tool", "tool_calls": [{"id": "call-next", "type": "function",
                "function": {"name": "search_knowledge_base", "arguments": '{"query":"模型调用"}'}}]}}]})
    models = model_client(handler)
    tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}]
    try:
        message = await models.agent_step([{"role": "user", "content": "如何调用模型"}], tools)
        assert message["reasoning_content"] == "private-tool"
        assert bodies[1]["messages"][-2]["reasoning_content"] == "private-first"
        await models.agent_step([message, {"role": "tool", "tool_call_id": "call-next", "content": "资料"}], tools)
        assert bodies[-1]["messages"][0]["reasoning_content"] == "private-tool"
        assert all(b["thinking"]["type"] == "enabled" and b["tool_choice"] == "auto" for b in bodies)
    finally:
        await models.close()


async def test_thinking_stream_only_exposes_final_content():
    wire = ('data: {"choices":[{"delta":{"reasoning_content":"private-thought"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"简洁结论[1]"},"finish_reason":"stop"}]}\n\n')
    models = model_client(lambda r: httpx.Response(200, text=wire))
    try:
        assert "".join([part async for part in models.stream_answer([])]) == "简洁结论[1]"
    finally:
        await models.close()


@pytest.mark.parametrize("recovers", [True, False])
async def test_structured_truncation_increases_budget_without_accepting_partial_json(recovers):
    budgets = []
    def handle(request):
        budgets.append(json.loads(request.content)["max_tokens"])
        # Even valid JSON must be rejected if the provider reports truncation.
        complete = recovers and len(budgets) == 3
        return httpx.Response(200, json={"choices": [{
            "message": {"content": '{"summary":"完整摘要","keywords":[]}'},
            "finish_reason": "stop" if complete else "length"}]})
    models = model_client(handle)
    try:
        if recovers:
            card = await models.structured("metadata", "摘要", {}, MetadataCard)
            assert card.summary == "完整摘要"
        else:
            with pytest.raises(AppError) as error:
                await models.structured("metadata", "摘要", {}, MetadataCard)
            assert error.value.code == "invalid_model_output"
        assert budgets == [4096, 8192, 16384]
    finally:
        await models.close()


@pytest.mark.parametrize("card", [{"summary": "  ", "keywords": []},
                                  {"summary": "摘要", "keywords": [" "]},
                                  {"summary": "摘要", "keywords": [], "locations": [{"raw": " ", "evidence": " "}]}])
async def test_blank_metadata_fields_require_repair(card):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(card)}, "finish_reason": "stop"}]})
    models = model_client(handler)
    with pytest.raises(AppError) as error:
        await models.structured("metadata", "摘要", {}, MetadataCard)
    assert len(calls) == 3 and error.value.code == "invalid_model_output"
    await models.close()
