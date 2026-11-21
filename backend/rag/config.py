import hashlib
import json
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_origin: str = "http://localhost:5173"
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    session_hours: int = 12
    upload_limit_mb: int = 50
    s3_endpoint_url: str = ""
    s3_region: str = "us-east-1"
    s3_signature_version: Literal["s3v4", "s3"] = "s3v4"
    s3_addressing_style: Literal["auto", "virtual", "path"] = "auto"
    s3_bucket: str = ""
    s3_access_key: SecretStr = SecretStr("")
    s3_secret_key: SecretStr = SecretStr("")
    milvus_uri: str = ""
    milvus_token: SecretStr = SecretStr("")
    milvus_database: str = "default"
    milvus_collection: str = "rag_chunks_v1"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_model: str = "deepseek-flash"
    coreference_use_deepseek: bool = True
    coreference_deepseek_thinking: Literal["disabled", "enabled"] = "disabled"
    coreference_base_url: str = ""
    coreference_api_key: SecretStr = SecretStr("")
    coreference_model: str = ""
    embedding_base_url: str = ""
    embedding_protocol: Literal["openai", "dashscope"] = "openai"
    embedding_endpoint: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_revision: str = "1"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 20
    embedding_store_sparse: bool = False
    retrieval_sparse_mode: Literal["bm25", "model"] = "bm25"
    rerank_endpoint: str = ""
    rerank_protocol: Literal["openai", "dashscope"] = "openai"
    rerank_api_key: SecretStr = SecretStr("")
    rerank_model: str = "qwen3.7-text-rerank"
    http_timeout_seconds: float = 60
    model_concurrency: int = 8
    chat_concurrency: int = 20
    worker_concurrency: int = 2
    job_lease_seconds: int = 90
    job_heartbeat_seconds: int = 15
    chunk_target: int = 1200
    chunk_max: int = 2000
    chunk_overlap: int = 150
    clean_normalize_whitespace: bool = True
    clean_remove_fillers: bool = True
    retrieval_per_path: int = 50
    retrieval_max_per_path: int = 200
    retrieval_candidates: int = 100
    rrf_k: int = 60
    answer_top_k: int = 5
    answer_initial_tokens: int = 8192
    answer_max_tokens: int = 32768
    answer_max_attempts: int = 3
    answer_deepseek_thinking: Literal["disabled", "enabled"] = "enabled"
    answer_reasoning_effort: Literal["low", "high", "max"] = "high"
    agent_deepseek_thinking: Literal["disabled", "enabled"] = "enabled"
    agent_reasoning_effort: Literal["low", "high", "max"] = "high"
    agent_initial_tokens: int = 8192
    agent_max_steps: int = 12
    agent_max_searches: int = 4
    chat_timeout_seconds: int = 480
    pipeline_version: str = "1"

    @model_validator(mode="after")
    def validate_settings(self):
        if not 64 <= self.answer_initial_tokens <= self.answer_max_tokens <= 65536:
            raise ValueError("Require 64 <= answer_initial_tokens <= answer_max_tokens <= 65536")
        if not 1 <= self.answer_max_attempts <= 4 or not 4 <= self.agent_max_steps <= 20:
            raise ValueError("Invalid answer recovery or agent step limit")
        if (not 1 <= self.agent_max_searches <= 6 or self.agent_max_steps < 2 * self.agent_max_searches + 2
                or not 1024 <= self.agent_initial_tokens <= 16384 or not 30 <= self.chat_timeout_seconds <= 540):
            raise ValueError("Invalid agent search or total chat time limit")
        if not 0 <= self.chunk_overlap < self.chunk_target <= self.chunk_max <= 10000:
            raise ValueError("Require 0 <= overlap < target <= max <= 10000")
        if not 1 <= self.answer_top_k <= 5 or not 1 <= self.embedding_dimension <= 2560:
            raise ValueError("Invalid answer count or embedding dimension")
        if not 1 <= self.embedding_batch_size <= 20:
            raise ValueError("Embedding batch size must be between 1 and 20")
        if self.embedding_store_sparse and self.embedding_protocol != "dashscope":
            raise ValueError("Sparse embeddings require the DashScope protocol")
        if self.retrieval_sparse_mode == "model" and not self.embedding_store_sparse:
            raise ValueError("Model sparse retrieval requires EMBEDDING_STORE_SPARSE=true and a rebuilt index")
        if min(self.worker_concurrency, self.model_concurrency, self.chat_concurrency) < 1:
            raise ValueError("Concurrency must be positive")
        if min(self.session_hours, self.upload_limit_mb, self.http_timeout_seconds, self.job_heartbeat_seconds, self.rrf_k) <= 0:
            raise ValueError("Timeouts, limits and heartbeat intervals must be positive")
        if not 1 <= self.retrieval_per_path <= self.retrieval_max_per_path <= 1000:
            raise ValueError("Retrieval limits must satisfy 1 <= per_path <= max_per_path <= 1000")
        if not self.answer_top_k <= self.retrieval_candidates <= 1000:
            raise ValueError("Candidate limit must cover answer_top_k and be at most 1000")
        if self.job_lease_seconds < self.job_heartbeat_seconds * 3:
            raise ValueError("Job lease must cover at least three heartbeat intervals")
        if self.app_env != "test" and not self.database_url.startswith("postgresql+asyncpg://"):
            raise ValueError("Production and development require external PostgreSQL")
        if self.app_env == "production" and not self.app_origin.startswith("https://"):
            raise ValueError("Production requires an HTTPS origin")
        return self

    @property
    def processing_fingerprint(self) -> str:
        fields = {k: getattr(self, k) for k in (
            "chunk_target", "chunk_max", "chunk_overlap", "pipeline_version", "coreference_use_deepseek",
            "deepseek_model", "embedding_model", "embedding_revision", "embedding_dimension",
            "embedding_protocol", "embedding_store_sparse",
            "clean_normalize_whitespace", "clean_remove_fillers",
        )}
        fields["coreference_model"] = self.deepseek_model if self.coreference_use_deepseek else self.coreference_model
        fields["coreference_thinking"] = (self.coreference_deepseek_thinking if self.coreference_use_deepseek
                                          else "provider-default")
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:20]

    @property
    def embedding_fingerprint(self) -> str:
        return f"{self.embedding_model}@{self.embedding_revision}:{self.embedding_dimension}:{self.embedding_protocol}:query-v1"

    @property
    def index_fingerprint(self) -> str:
        return self.embedding_fingerprint + (":model-sparse-v1" if self.embedding_store_sparse else ":bm25-v1")


@lru_cache
def get_settings() -> Settings:
    return Settings()
