"""Ingest MSMARCO-XI Hindi subset into Qdrant with multiple chunking strategies."""
import time
import numpy as np
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from chunking import fixed_size_chunking, sentence_window_chunking, semantic_chunking, metadata_aware_chunking
from config import EMBEDDING_MODEL, QDRANT_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP

BATCH_SIZE = 128
MAX_PASSAGES = 50000


def ingest():
    print("Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()

    print("Loading dataset...")
    ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)

    client = QdrantClient(path="./qdrant_data")
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    all_embeddings = []
    point_id = 0
    passage_count = 0

    t0 = time.perf_counter()
    for row in ds:
        passages = row.get("passages", {})
        translated = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        query = row.get("query", "")
        query_type = row.get("query_type", "")
        query_id = row.get("query_id", 0)

        for i, passage in enumerate(translated):
            if not passage or not passage.strip():
                continue

            selected = is_selected[i] if i < len(is_selected) else 0
            meta = {
                "query": query,
                "query_id": query_id,
                "query_type": query_type,
                "is_selected": selected,
                "chunking_strategy": "passage",
                "passage_index": i,
            }

            # Strategy 1: full passage as chunk
            full_chunk = metadata_aware_chunking(passage, meta)
            points.append((full_chunk["text"], {**full_chunk["metadata"], "text": full_chunk["text"]}))

            # Strategy 2: fixed-size sub-chunks for longer passages
            sub_chunks = fixed_size_chunking(passage, CHUNK_SIZE, CHUNK_OVERLAP)
            if len(sub_chunks) > 1:
                for j, sc in enumerate(sub_chunks):
                    m = {**meta, "chunking_strategy": "fixed_size", "sub_index": j, "text": sc}
                    points.append((sc, m))

            # Strategy 3: sentence-window chunks
            sw_chunks = sentence_window_chunking(passage)
            for sw in sw_chunks:
                m = {
                    **meta,
                    "chunking_strategy": "sentence_window",
                    "sentence_index": sw["sentence_index"],
                    "text": sw["context"],
                }
                points.append((sw["context"], m))

            # Strategy 4: semantic chunking (split on embedding discontinuities)
            # ponytail: only apply to selected passages to avoid 10x ingest slowdown
            # (encodes each sentence individually). Apply to all if ingest time is not a concern.
            if selected == 1:
                sem_chunks = semantic_chunking(passage, model, threshold=0.5)
                if len(sem_chunks) > 1:
                    for j, sc in enumerate(sem_chunks):
                        m = {**meta, "chunking_strategy": "semantic", "sub_index": j, "text": sc}
                        points.append((sc, m))

            passage_count += 1
            if passage_count >= MAX_PASSAGES:
                break

            # Batch embed + upsert
            if len(points) >= BATCH_SIZE:
                texts = [p[0] for p in points]
                metas = [p[1] for p in points]
                embs = model.encode(
                    [f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=64,
                )
                all_embeddings.extend(embs)
                batch_points = [
                    PointStruct(id=point_id + k, vector=embs[k].tolist(), payload=metas[k])
                    for k in range(len(texts))
                ]
                client.upsert(QDRANT_COLLECTION, batch_points)
                point_id += len(batch_points)
                points = []
                print(f"  Indexed {point_id} chunks ({passage_count} passages)...")

        if passage_count >= MAX_PASSAGES:
            break

    # Flush remaining
    if points:
        texts = [p[0] for p in points]
        metas = [p[1] for p in points]
        embs = model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=64)
        all_embeddings.extend(embs)
        batch_points = [
            PointStruct(id=point_id + k, vector=embs[k].tolist(), payload=metas[k])
            for k in range(len(texts))
        ]
        client.upsert(QDRANT_COLLECTION, batch_points)
        point_id += len(batch_points)

    elapsed = time.perf_counter() - t0

    # Save corpus centroid for off-topic detection
    centroid = np.mean(all_embeddings, axis=0)
    np.save("corpus_centroid.npy", centroid)

    print(f"\nDone. {point_id} chunks indexed from {passage_count} passages in {elapsed:.1f}s")
    print(f"Corpus centroid saved to corpus_centroid.npy")


if __name__ == "__main__":
    ingest()
