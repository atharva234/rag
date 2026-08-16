import numpy as np
from models import GuardrailResult
from config import OFF_TOPIC_THRESHOLD, RELEVANCE_THRESHOLD, GROUNDEDNESS_THRESHOLD

_BLOCKED = [
    "ignore previous", "ignore above", "system prompt", "jailbreak",
    "pretend you", "act as", "you are now", "disregard",
]


def check_unsafe_input(query: str) -> GuardrailResult:
    q = query.lower()
    for pat in _BLOCKED:
        if pat in q:
            return GuardrailResult(
                passed=False, flags={"unsafe_input": True},
                message="Query blocked: potentially unsafe content detected.",
            )
    return GuardrailResult(passed=True)


def check_off_topic(query_emb: np.ndarray, corpus_centroid: np.ndarray) -> GuardrailResult:
    centroid_norm = float(np.linalg.norm(corpus_centroid))
    if centroid_norm < 1e-6:
        return GuardrailResult(passed=True, flags={"skipped": "no centroid"})
    sim = float(np.dot(query_emb, corpus_centroid) / (
        np.linalg.norm(query_emb) * centroid_norm + 1e-8
    ))
    if sim < OFF_TOPIC_THRESHOLD:
        return GuardrailResult(
            passed=False, flags={"off_topic": True, "similarity": sim},
            message="This question appears outside the scope of the knowledge base.",
        )
    return GuardrailResult(passed=True, flags={"similarity": sim})


def check_retrieval_confidence(top_score: float) -> GuardrailResult:
    if top_score < RELEVANCE_THRESHOLD:
        return GuardrailResult(
            passed=False, flags={"low_confidence": True, "top_score": top_score},
            message="Could not find sufficiently relevant information to answer.",
        )
    return GuardrailResult(passed=True, flags={"top_score": top_score})


def check_groundedness(answer: str, chunks: list[str]) -> GuardrailResult:
    context_words = set(" ".join(chunks).lower().split())
    answer_words = set(answer.lower().split())
    if not answer_words:
        return GuardrailResult(passed=True)
    overlap = len(answer_words & context_words) / len(answer_words)
    if overlap < GROUNDEDNESS_THRESHOLD:
        return GuardrailResult(
            passed=False,
            flags={"groundedness_score": overlap, "hallucination_risk": True},
            message="Answer may not be fully supported by retrieved context.",
        )
    return GuardrailResult(passed=True, flags={"groundedness_score": overlap})
