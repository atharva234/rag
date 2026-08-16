import re
import numpy as np
from config import CHUNK_SIZE, CHUNK_OVERLAP, SENTENCE_WINDOW


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?।])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def fixed_size_chunking(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def sentence_window_chunking(text: str, window: int = SENTENCE_WINDOW) -> list[dict]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    results = []
    for i, sent in enumerate(sentences):
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        results.append({
            "text": sent,
            "context": " ".join(sentences[start:end]),
            "sentence_index": i,
        })
    return results


def semantic_chunking(text: str, model, threshold: float = 0.5) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences
    embeddings = model.encode(sentences, normalize_embeddings=True)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i], embeddings[i - 1]))
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


def metadata_aware_chunking(passage: str, metadata: dict) -> dict:
    return {"text": passage, "metadata": metadata}
