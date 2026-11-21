from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginInput(StrictModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class UserCreate(LoginInput):
    password: str = Field(min_length=12, max_length=256)
    display_name: str = Field(min_length=1, max_length=120)
    role: Literal["admin", "member"] = "member"


class UserUpdate(StrictModel):
    active: bool


class KBCreate(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=4000)


class GrantInput(StrictModel):
    user_ids: list[UUID] = Field(max_length=1000)


class SearchInput(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    kb_ids: list[UUID] = Field(min_length=1, max_length=100)


class ChatInput(SearchInput):
    conversation_id: UUID


class Mention(StrictModel):
    raw: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=1000)
    normalized: str | None = Field(default=None, max_length=100)

    @field_validator("raw", "evidence")
    @classmethod
    def has_text(cls, value):
        if not value.strip():
            raise ValueError("Mention evidence must contain text")
        return value


class MetadataCard(StrictModel):
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    keywords: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]] = Field(max_length=20)
    time_mentions: list[Mention] = Field(default_factory=list, max_length=30)
    locations: list[Mention] = Field(default_factory=list, max_length=30)


class Replacement(StrictModel):
    block_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    original: str
    target: str = Field(min_length=1, max_length=100)
    evidence_block_id: str
    evidence: str = Field(min_length=1, max_length=1000)
    unambiguous: bool


class ReplacementOutput(StrictModel):
    replacements: list[Replacement] = Field(default_factory=list, max_length=100)


class RewriteOutput(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    needs_clarification: bool = False
    clarification: str | None = Field(default=None, max_length=1000)


class AgentSearch(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    strategy: Literal["hybrid", "dense", "sparse"] = "hybrid"


class AgentAssessment(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    verdict: Literal["sufficient", "partial", "irrelevant", "conflicting"]
    relevant_chunk_ids: list[str] = Field(max_length=30)
    missing_aspects: list[str] = Field(default_factory=list, max_length=8)


class AgentReuse(StrictModel):
    turn_id: str = Field(min_length=1, max_length=36)


class AgentAnswer(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["answer", "continue", "restart"]
    source_turn_id: str | None = None
    chunk_ids: list[str] = Field(default_factory=list, max_length=5)


class AgentClarify(StrictModel):
    question: str = Field(min_length=1, max_length=1000)


class UserView(BaseModel):
    id: str
    username: str
    display_name: str
    role: Literal["admin", "member"]
    active: bool


class AuthView(BaseModel):
    user: UserView
    csrf_token: str


class KBView(BaseModel):
    id: str
    name: str
    description: str
    document_count: int = 0


class GrantView(BaseModel):
    user_ids: list[str]


class DocumentView(BaseModel):
    id: str
    kb_id: str
    title: str
    current_version_id: str | None
    latest_version_id: str | None
    status: str
    chunk_count: int
    warnings: list[str]
    created_at: datetime


class DocumentList(BaseModel):
    total: int
    items: list[DocumentView]


class UploadView(BaseModel):
    document_id: str
    version_id: str
    job_id: str | None
    deduplicated: bool


class JobView(BaseModel):
    id: str
    document_id: str
    version_id: str | None
    kind: str
    status: str
    stage: str
    attempts: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class VersionView(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    warnings: list[str]
    created_at: datetime
    current: bool


class SourceView(BaseModel):
    block_id: str
    page: int | None
    paragraph: int
    original: str
    cleaned: str | None = None
    resolved: str | None = None
    processed_start: int | None = None
    processed_end: int | None = None


class SavedMetadata(MetadataCard):
    model: str
    prompt_version: str


class ChunkView(BaseModel):
    chunk_id: str
    ordinal: int
    version_id: str
    document_id: str
    body: str
    heading_path: list[str]
    sources: list[SourceView]
    edits: list[dict]
    metadata: SavedMetadata


class ChunkList(BaseModel):
    total: int
    items: list[ChunkView]


class RetrievalPath(BaseModel):
    rank: int
    score: float


class HitView(BaseModel):
    chunk_id: str
    document_id: str
    version_id: str
    kb_id: str
    title: str
    body: str
    summary: str
    keywords: list[str]
    heading_path: list[str]
    time_mentions: list[Mention]
    locations: list[Mention]
    sources: list[SourceView]
    paths: dict[str, RetrievalPath]
    rrf_score: float
    rerank_score: float


class SearchView(BaseModel):
    query: str
    strategy: Literal["hybrid", "dense", "sparse"]
    sparse_mode: Literal["bm25", "model"]
    top_k: int = Field(ge=1, le=5)
    hits: list[HitView]
    candidates: list[HitView]
    discarded: int


class CitationView(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    version_id: str
    kb_id: str
    title: str
    heading_path: list[str]
    sources: list[SourceView]
    score: float


class TurnView(BaseModel):
    id: str
    question: str
    answer: str
    citations: list[CitationView]
    status: str
    error_code: str | None
    created_at: datetime
    draft_answer: str = ""
    can_resume: bool = False


class ConversationView(BaseModel):
    id: str
    title: str
    created_at: datetime | None = None


class ConversationDetail(ConversationView):
    turns: list[TurnView]
