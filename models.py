from pydantic import BaseModel


class STTResult(BaseModel):
    text: str
    language: str
    confidence: float
    latency_ms: float


class RetrievalHit(BaseModel):
    text: str
    score: float
    metadata: dict = {}


class RetrievalResult(BaseModel):
    hits: list[RetrievalHit]
    latency_ms: float


class GuardrailResult(BaseModel):
    passed: bool
    flags: dict = {}
    message: str = ""


class GenerationResult(BaseModel):
    answer: str
    latency_ms: float


class PipelineResponse(BaseModel):
    query_text: str
    answer: str
    retrieved_chunks: list[RetrievalHit]
    grounded: bool
    guardrail_flags: dict
    latencies: dict
    total_latency_ms: float
