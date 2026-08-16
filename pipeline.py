import time
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import (
    SARVAM_API_KEY, GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    QDRANT_COLLECTION, TOP_K, MAX_RETRIES, RETRY_BACKOFF,
)
from models import (
    STTResult, RetrievalHit, RetrievalResult, GenerationResult, PipelineResponse,
)
from guardrails import check_unsafe_input, check_off_topic, check_retrieval_confidence, check_groundedness
from analytics import LatencyTracker

_embed_model = None
_qdrant = None
_groq = None
_corpus_centroid = None
_bm25 = None
_bm25_docs = None
_tracker = LatencyTracker()


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path="./qdrant_data")
    return _qdrant


def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def get_corpus_centroid():
    global _corpus_centroid
    if _corpus_centroid is None:
        try:
            centroid_data = np.load("corpus_centroid.npy")
            _corpus_centroid = centroid_data
        except FileNotFoundError:
            _corpus_centroid = np.zeros(get_embed_model().get_sentence_embedding_dimension())
    return _corpus_centroid


_bm25_lookup = None

def get_bm25():
    global _bm25, _bm25_docs, _bm25_lookup
    if _bm25 is None:
        client = get_qdrant()
        try:
            count = client.count(QDRANT_COLLECTION).count
        except Exception:
            return None, [], {}
        records, _ = client.scroll(QDRANT_COLLECTION, limit=min(count, 50000), with_payload=True, with_vectors=False)
        _bm25_docs = [(r.id, r.payload.get("text", "")) for r in records]
        _bm25_lookup = {r.id: r.payload.get("text", "") for r in records}
        tokenized = [doc.lower().split() for _, doc in _bm25_docs]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _bm25_docs, _bm25_lookup


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=RETRY_BACKOFF))
def transcribe_audio(audio_path: str) -> STTResult:
    t0 = time.perf_counter()
    url = "https://api.sarvam.ai/speech-to-text"
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path, f, "audio/wav")}
        data = {"model": "saaras:v3", "mode": "translate", "with_timestamps": "false"}
        headers = {"api-subscription-key": SARVAM_API_KEY}
        resp = requests.post(url, files=files, data=data, headers=headers, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    latency = (time.perf_counter() - t0) * 1000
    return STTResult(
        text=result.get("transcript", ""),
        language=result.get("language_code", "unknown"),
        confidence=result.get("language_probability", 0.0),
        latency_ms=latency,
    )


def embed_query(text: str) -> np.ndarray:
    model = get_embed_model()
    return model.encode(f"query: {text}", normalize_embeddings=True)


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=RETRY_BACKOFF))
def retrieve(query_embedding: np.ndarray, query_text: str = "", top_k: int = TOP_K, bm25_weight: float = 0.3) -> RetrievalResult:
    t0 = time.perf_counter()
    client = get_qdrant()

    # Dense retrieval
    dense_results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding.tolist(),
        limit=top_k * 2,
    )

    # BM25 sparse retrieval
    bm25, bm25_docs, bm25_lookup = get_bm25()
    bm25_scores = {}
    if bm25 is not None and query_text:
        scores = bm25.get_scores(query_text.lower().split())
        top_indices = np.argsort(scores)[-top_k * 2:][::-1]
        max_bm25 = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0
        for idx in top_indices:
            if scores[idx] > 0:
                doc_id, text = bm25_docs[idx]
                bm25_scores[doc_id] = scores[idx] / max_bm25

    # Merge: combine dense + BM25 via weighted RRF
    combined = {}
    for r in dense_results:
        combined[r.id] = {
            "text": r.payload.get("text", ""),
            "payload": r.payload,
            "dense_score": r.score,
            "bm25_score": bm25_scores.get(r.id, 0.0),
        }
    for doc_id, bm25_s in bm25_scores.items():
        if doc_id not in combined:
            text = bm25_lookup.get(doc_id, "") if bm25_lookup else ""
            combined[doc_id] = {
                "text": text,
                "payload": {"text": text},
                "dense_score": 0.0,
                "bm25_score": bm25_s,
            }

    for v in combined.values():
        v["final_score"] = (1 - bm25_weight) * v["dense_score"] + bm25_weight * v["bm25_score"]

    ranked = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)[:top_k]
    hits = [
        RetrievalHit(text=r["text"], score=r["final_score"], metadata={**r["payload"], "dense_score": r["dense_score"], "bm25_score": r["bm25_score"]})
        for r in ranked
    ]
    latency = (time.perf_counter() - t0) * 1000
    return RetrievalResult(hits=hits, latency_ms=latency)


def generate_answer_stream(query: str, context_chunks: list[str]):
    """Yields (partial_text, is_final, latency_ms). Use for UI streaming."""
    t0 = time.perf_counter()
    context = "\n---\n".join(context_chunks)
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer the question using ONLY the context below. "
            "If the context doesn't contain enough information, say so. "
            "Be concise and factual. Respond in the same language as the question."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    stream = get_groq().chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=150, temperature=0.1, stream=True,
    )
    accumulated = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        accumulated += delta
        yield accumulated, False, 0.0
    yield accumulated, True, (time.perf_counter() - t0) * 1000


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=RETRY_BACKOFF))
def generate_answer(query: str, context_chunks: list[str]) -> GenerationResult:
    t0 = time.perf_counter()
    context = "\n---\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question using ONLY the context below. "
                "If the context doesn't contain enough information, say so. "
                "Be concise and factual. Respond in the same language as the question."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]
    client = get_groq()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=150,
        temperature=0.1,
    )
    answer = resp.choices[0].message.content.strip()
    latency = (time.perf_counter() - t0) * 1000
    return GenerationResult(answer=answer, latency_ms=latency)


def run_pipeline_stream(query_text: str):
    """Generator for Gradio streaming. Yields (answer, latency_md, chunks_md)."""
    latencies = {}

    guard = check_unsafe_input(query_text)
    if not guard.passed:
        yield guard.message, "", ""
        return

    t0 = time.perf_counter()
    query_emb = embed_query(query_text)
    latencies["embedding"] = (time.perf_counter() - t0) * 1000

    guard = check_off_topic(query_emb, get_corpus_centroid())
    if not guard.passed:
        yield guard.message, _fmt_latency(latencies), ""
        return

    retrieval = retrieve(query_emb, query_text=query_text)
    latencies["retrieval"] = retrieval.latency_ms
    chunks_md = _fmt_chunks(retrieval.hits)

    top_score = retrieval.hits[0].score if retrieval.hits else 0.0
    guard = check_retrieval_confidence(top_score)
    if not guard.passed:
        yield guard.message, _fmt_latency(latencies), chunks_md
        return

    chunk_texts = [h.text for h in retrieval.hits]
    for partial, is_final, gen_ms in generate_answer_stream(query_text, chunk_texts):
        if is_final:
            latencies["generation"] = gen_ms
            latencies["total"] = sum(latencies.values())
            _tracker.record(latencies)
            ground_check = check_groundedness(partial, chunk_texts)
            answer = partial if ground_check.passed else f"{partial}\n\n⚠️ {ground_check.message}"
            yield answer, _fmt_latency(latencies), chunks_md
        else:
            yield partial, _fmt_latency(latencies), chunks_md


def _fmt_latency(latencies: dict) -> str:
    rag_ms = latencies.get("embedding", 0) + latencies.get("retrieval", 0)
    gen_ms = latencies.get("generation", 0)
    total_ms = latencies.get("total", rag_ms + gen_ms)
    return f"**RAG pipeline:** {rag_ms:.0f}ms · **Generation:** {gen_ms:.0f}ms · **Total:** {total_ms:.0f}ms"


def _fmt_chunks(hits) -> str:
    if not hits:
        return ""
    parts = []
    for i, h in enumerate(hits[:3]):
        strategy = h.metadata.get("chunking_strategy", "?")
        parts.append(f"**[{i+1}]** `score={h.score:.3f}` `strategy={strategy}`\n\n{h.text[:300]}")
    return "\n\n---\n\n".join(parts)


def run_pipeline(query_text: str) -> PipelineResponse:
    latencies = {}

    # Guardrail: unsafe input
    guard = check_unsafe_input(query_text)
    if not guard.passed:
        return PipelineResponse(
            query_text=query_text, answer=guard.message,
            retrieved_chunks=[], grounded=False,
            guardrail_flags=guard.flags, latencies={}, total_latency_ms=0,
        )

    # Embed
    t0 = time.perf_counter()
    query_emb = embed_query(query_text)
    latencies["embedding"] = (time.perf_counter() - t0) * 1000

    # Guardrail: off-topic
    guard = check_off_topic(query_emb, get_corpus_centroid())
    if not guard.passed:
        return PipelineResponse(
            query_text=query_text, answer=guard.message,
            retrieved_chunks=[], grounded=False,
            guardrail_flags=guard.flags, latencies=latencies, total_latency_ms=sum(latencies.values()),
        )

    # Retrieve (hybrid: dense + BM25)
    retrieval = retrieve(query_emb, query_text=query_text)
    latencies["retrieval"] = retrieval.latency_ms

    # Guardrail: retrieval confidence
    top_score = retrieval.hits[0].score if retrieval.hits else 0.0
    guard = check_retrieval_confidence(top_score)
    if not guard.passed:
        return PipelineResponse(
            query_text=query_text, answer=guard.message,
            retrieved_chunks=retrieval.hits, grounded=False,
            guardrail_flags=guard.flags, latencies=latencies, total_latency_ms=sum(latencies.values()),
        )

    # Generate
    chunk_texts = [h.text for h in retrieval.hits]
    gen = generate_answer(query_text, chunk_texts)
    latencies["generation"] = gen.latency_ms

    # Guardrail: groundedness
    ground_check = check_groundedness(gen.answer, chunk_texts)
    grounded = ground_check.passed
    answer = gen.answer if grounded else f"{gen.answer}\n\n⚠️ {ground_check.message}"

    total = sum(latencies.values())
    latencies["total"] = total
    _tracker.record(latencies)

    return PipelineResponse(
        query_text=query_text, answer=answer,
        retrieved_chunks=retrieval.hits, grounded=grounded,
        guardrail_flags=ground_check.flags, latencies=latencies, total_latency_ms=total,
    )


def run_pipeline_voice(audio_path: str) -> PipelineResponse:
    stt = transcribe_audio(audio_path)
    result = run_pipeline(stt.text)
    result.latencies["stt"] = stt.latency_ms
    result.total_latency_ms += stt.latency_ms
    if _tracker.records:
        _tracker.records[-1]["stt"] = stt.latency_ms
    return result


def get_tracker() -> LatencyTracker:
    return _tracker


def warmup():
    import sys
    t0 = time.perf_counter()
    print("Warming up...", file=sys.stderr)
    get_embed_model()
    print(f"  Embedding model loaded ({(time.perf_counter()-t0)*1000:.0f}ms)", file=sys.stderr)
    get_qdrant()
    get_corpus_centroid()
    print(f"  Qdrant + centroid loaded ({(time.perf_counter()-t0)*1000:.0f}ms)", file=sys.stderr)
    get_bm25()
    print(f"  BM25 index built ({(time.perf_counter()-t0)*1000:.0f}ms)", file=sys.stderr)
    get_groq()
    print(f"  All warm ({(time.perf_counter()-t0)*1000:.0f}ms)", file=sys.stderr)
