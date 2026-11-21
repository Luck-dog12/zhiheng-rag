from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("rag_requests_total", "HTTP responses", ["method", "route", "status"])
LATENCY = Histogram("rag_request_seconds", "HTTP request duration", ["route"])
STAGES = Histogram("rag_stage_seconds", "Pipeline and retrieval stage duration", ["stage"])
TOKENS = Counter("rag_model_tokens_total", "Provider-reported token usage", ["purpose", "kind"])
MODEL_ERRORS = Counter("rag_model_errors_total", "External model errors", ["purpose", "code"])
JOBS = Gauge("rag_jobs", "Job count by status (worker refresh)", ["status"])


def record_usage(purpose: str, usage: dict):
    for kind in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(kind)
        if isinstance(value, (int, float)) and value >= 0:
            TOKENS.labels(purpose, kind).inc(value)
