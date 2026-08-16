---
title: Voice RAG MSMARCO-XI
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Voice-Enabled RAG System — MSMARCO-XI

A voice-enabled Retrieval-Augmented Generation (RAG) system built on the [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) Hindi dataset. Speak or type a question, and the system transcribes it, retrieves relevant passages from a vector database, and generates a grounded answer — all end-to-end.

## What This Project Does

**Pipeline flow:**

```
Voice Input → Speech-to-Text (Sarvam) → Query Embedding → Hybrid Retrieval (Qdrant + BM25) → Answer Generation (Groq/Llama) → Guardrail Checks → Response
```

A user speaks a question into the microphone (or types it). The system:

1. **Transcribes** the audio to text using Sarvam AI's Saarika v2 model (optimized for Indian languages)
2. **Embeds** the query using a multilingual sentence transformer
3. **Retrieves** the most relevant passages using hybrid search (dense vectors + BM25 sparse scoring)
4. **Generates** a concise answer using Groq-hosted Llama 3.1-8B, constrained to the retrieved context
5. **Validates** the answer through multiple guardrails before returning it

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Speech-to-Text** | [Sarvam AI](https://sarvam.ai/) — Saaras v3 | Purpose-built for Indian languages/accents; the dataset is Hindi MS MARCO |
| **Embeddings** | `intfloat/multilingual-e5-small` (SentenceTransformers) | Multilingual, fast, good quality for Indic text |
| **Vector Database** | [Qdrant](https://qdrant.tech/) (local/in-memory) | Sub-10ms ANN search, lightweight, no server needed |
| **Sparse Retrieval** | `rank_bm25` (BM25Okapi) | Classic keyword matching to complement dense retrieval |
| **LLM** | [Groq](https://groq.com/) — Llama 3.1-8B-Instant | LPU inference gives fast generation — retrieval pipeline stays under 200ms |
| **Web UI** | [Gradio](https://gradio.app/) | Native mic input widget, tabs, markdown — deploy-ready |
| **Data Models** | [Pydantic](https://docs.pydantic.dev/) | Typed I/O between every pipeline stage |
| **Retry Logic** | [Tenacity](https://tenacity.readthedocs.io/) | Exponential backoff on STT/LLM API calls |
| **Dataset** | [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) | MS MARCO translated into 14 Indic languages |
| **Deployment** | Hugging Face Spaces (Docker) | Free hosting with a live URL |

---

## Project Structure

```
rag-goa/
├── config.py            # All configuration: API keys, thresholds, model names
├── models.py            # Pydantic schemas for typed I/O between stages
├── chunking.py          # 4 chunking strategies
├── guardrails.py        # 4 guardrail checks
├── pipeline.py          # Full orchestrated pipeline with harness (retries, error recovery)
├── ingest.py            # Data loading + multi-strategy chunking + Qdrant indexing
├── app.py               # Gradio web UI (text, voice, analytics tabs)
├── analytics.py         # Latency tracking with P50/P70/P100 reporting
├── benchmark.py         # Runs 20 test queries and prints latency report
├── test_self_check.py   # Self-check tests for chunking + guardrails
├── requirements.txt     # Python dependencies
├── Dockerfile           # For Hugging Face Spaces deployment
├── .env.example         # Template for API keys
└── .gitignore
```

---

## Detailed File-by-File Breakdown

### `config.py` — Central Configuration

All tunable parameters in one place, loaded from environment variables (via `.env`):

- **API keys**: `SARVAM_API_KEY`, `GROQ_API_KEY` (loaded from `.env`)
- **Model choices**: `intfloat/multilingual-e5-small` for embeddings, `llama-3.1-8b-instant` for generation
- **Chunking params**: chunk size (256 tokens), overlap (50 tokens), sentence window (±2)
- **Guardrail thresholds**: relevance (0.35), groundedness (0.4), off-topic (0.25)
- **Retry config**: 3 attempts, 0.1s exponential backoff

### `models.py` — Typed Data Models

Pydantic models that define the contract between every pipeline stage:

- `STTResult` — transcription output (text, language, confidence, latency)
- `RetrievalHit` — single search result (text, score, metadata)
- `RetrievalResult` — list of hits + latency
- `GuardrailResult` — pass/fail + flags + message
- `GenerationResult` — answer text + latency
- `PipelineResponse` — full end-to-end response with all metadata

### `chunking.py` — 4 Chunking Strategies

This is where the "vast chunking" requirement is met. Four distinct strategies:

1. **Fixed-size chunking with overlap** — splits text into 256-word windows with 50-word overlap. The baseline approach — handles any text length.

2. **Sentence-window chunking** — each sentence is the retrieval unit, but at generation time the surrounding ±2 sentences are included as context. Gives precise retrieval with rich context.

3. **Semantic chunking** — splits text at points where the embedding similarity between consecutive sentences drops below a threshold (0.5). Produces chunks that are topically coherent, unlike fixed-size which can split mid-topic.

4. **Metadata-aware chunking** — treats the full passage as a chunk but tags it with rich metadata (query, query_type, query_id, is_selected). Enables filtered retrieval by topic or relevance.

### `ingest.py` — Data Ingestion Pipeline

Loads the MSMARCO-XI Hindi dataset from HuggingFace (streaming, so no full download needed) and indexes it into Qdrant:

- Streams through up to 50,000 passages
- For each passage, applies **all 4 chunking strategies** and indexes every chunk
- Embeds chunks using the E5 model with the `passage:` prefix (as the model expects)
- Batch-upserts into Qdrant in groups of 128
- Computes and saves the **corpus centroid** (`corpus_centroid.npy`) — the average embedding of all indexed chunks — used later for off-topic detection

### `pipeline.py` — The Orchestrated Pipeline (Harness)

This is the core of the system. It's not a raw prompt-in/text-out call — it's a structured pipeline with:

**Initialization (lazy singletons):**
- Embedding model, Qdrant client, Groq client, BM25 index, corpus centroid — all loaded once on first use

**Speech-to-Text (`transcribe_audio`):**
- Calls Sarvam AI's `/speech-to-text` endpoint with `mode=translate`
- Wrapped in `@retry` with 3 attempts and exponential backoff
- Returns typed `STTResult`

**Hybrid Retrieval (`retrieve`):**
- **Dense search**: queries Qdrant with the embedded query vector, gets top 2×K candidates
- **Sparse search (BM25)**: scores all indexed documents against the query tokens, gets top 2×K candidates
- **Fusion**: combines both score sets with weighted formula: `0.7 × dense_score + 0.3 × bm25_score`
- Returns the top K results ranked by combined score
- This means a passage that's semantically similar AND contains the exact keywords ranks highest

**Answer Generation (`generate_answer`):**
- Sends retrieved chunks + query to Groq's Llama 3.1-8B
- System prompt constrains the model to answer ONLY from context and match the query language
- `temperature=0.1` for factual responses, `max_tokens=300` for conciseness

**Pipeline orchestration (`run_pipeline`):**
- Runs guardrails at 4 checkpoints (see below)
- Tracks latency of every stage
- Records metrics to the global `LatencyTracker`
- Returns a fully typed `PipelineResponse`

### `guardrails.py` — 4 Safety Checks

The system knows **when not to answer**, not just how to answer:

1. **Unsafe input detection** — checks for prompt injection patterns ("ignore previous", "pretend you are", "jailbreak", etc.). Blocks before any processing happens.

2. **Off-topic detection** — computes cosine similarity between the query embedding and the corpus centroid (average of all indexed embeddings). If similarity < 0.25, the query is outside the knowledge base scope — refuses to answer.

3. **Low-confidence retrieval** — if the top retrieval score is below 0.35, the system doesn't have good enough context to generate a reliable answer — says so explicitly.

4. **Groundedness check** — after generation, computes word overlap between the answer and the retrieved chunks. If less than 40% of answer words appear in the context, flags potential hallucination.

### `analytics.py` — Latency Tracking

Records latency for every query across all pipeline stages. Reports:

- **P50** (median) — typical performance
- **P70** — 70th percentile
- **P100** (max) — worst case
- **Mean** and **count**

Formatted as a table for easy inclusion in the submission.

### `app.py` — Gradio Web Interface

Three tabs:

1. **Text Query** — type a question, get an answer + latency breakdown + top retrieved chunks
2. **Voice Query** — record via microphone, get transcription + answer + latencies
3. **Latency Analytics** — click Refresh to see the P50/P70/P100 table across all queries run so far

### `benchmark.py` — Latency Benchmarking

Runs 20 pre-defined queries (10 Hindi, 10 English) through the pipeline and prints per-query latency with a final P50/P70/P100 summary table. This generates the numbers for the submission.

### `test_self_check.py` — Self-Check Tests

Verifies core logic without needing API keys:

- Fixed-size chunking produces expected number of chunks
- Sentence-window chunking returns correct window count
- Sentence splitter handles multiple punctuation marks
- Unsafe input guardrail blocks injection attempts
- Unsafe input guardrail allows normal queries
- Groundedness check passes for grounded answers
- Groundedness check catches ungrounded answers

---

## How to Set Up and Run (Step by Step)

### Prerequisites

- Python 3.10+
- A [Sarvam AI](https://dashboard.sarvam.ai/) API key (free tier available)
- A [Groq](https://console.groq.com/) API key (free tier available)

### Step 1: Clone and Install

```bash
cd rag-goa
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your real keys:

```
SARVAM_API_KEY=your_actual_sarvam_key
GROQ_API_KEY=your_actual_groq_key
```

### Step 3: Run Self-Check Tests

```bash
python test_self_check.py
```

Expected output: `All checks passed.`

### Step 4: Ingest the Dataset

```bash
python ingest.py
```

This will:
- Download MSMARCO-XI Hindi from HuggingFace (streamed, ~few GB)
- Apply all 4 chunking strategies to each passage
- Embed and index into local Qdrant (saved to `./qdrant_data/`)
- Save corpus centroid to `corpus_centroid.npy`
- Takes ~10-30 minutes depending on your machine

You'll see progress logs like:
```
Loading model...
Loading dataset...
  Indexed 128 chunks (15 passages)...
  Indexed 256 chunks (28 passages)...
  ...
Done. 245000 chunks indexed from 50000 passages in 1823.4s
Corpus centroid saved to corpus_centroid.npy
```

### Step 5: Launch the App

```bash
python app.py
```

Open `http://localhost:7860` in your browser. You'll see three tabs:
- **Text Query**: type a question and click "Ask"
- **Voice Query**: click the mic, speak, then click "Ask"
- **Latency Analytics**: click "Refresh" after running some queries

### Step 6: Run the Benchmark

```bash
python benchmark.py
```

This runs 20 queries and prints a latency report like:

```
Running 20 benchmark queries...

  [✓]  1. भारत की राजधानी क्या है?                          RAG: 18ms  Gen: 820ms  Total: 842ms
  [✓]  2. ताज महल कहाँ स्थित है?                              RAG: 12ms  Gen: 790ms  Total: 806ms
  ...

======================================================================
LATENCY REPORT
======================================================================
Stage                          P50      P70      P100
------------------------------------------------------------
embedding                     12.3ms   14.1ms   18.7ms
retrieval                      8.1ms    9.4ms   15.2ms
generation                   810.4ms  890.3ms 1240.1ms
total                        834.2ms  915.8ms 1274.0ms

Total queries: 20

Note: 'embedding' + 'retrieval' = RAG pipeline latency (target <200ms).
      'generation' = LLM inference via Groq (network-bound, typically 500-1500ms).
      STT latency is measured separately for voice queries.
```

### Step 7: Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Visibility: **Public**

2. Push your repo:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: voice RAG system"
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/voice-rag-msmarco
   git push -u origin main
   ```

3. Set secrets in Space settings:
   - `SARVAM_API_KEY` = your key
   - `GROQ_API_KEY` = your key

4. The Space will build the Docker image, run `ingest.py` during build, and launch `app.py`. Your live URL will be:
   ```
   https://YOUR_USERNAME-voice-rag-msmarco.hf.space
   ```

---

## Architecture Decisions and Why

### Why Sarvam over ElevenLabs?

MSMARCO-XI is an Indic-language dataset. Sarvam's Saarika model is purpose-built for Indian languages and accents. ElevenLabs is optimized for English/Western TTS-first workflows. Sarvam is the correct tool for this dataset.

### Why Groq over OpenAI/Anthropic?

The 200ms latency target for the retrieval pipeline. While LLM generation itself takes ~800-1200ms on any API (TTFT + token generation), Groq's LPU inference is still the fastest available option. The RAG pipeline (embedding + vector search + BM25) runs in <30ms — well under the 200ms target. Generation latency is network-bound and reported separately.

### Why Qdrant (local) over Pinecone/Weaviate?

Sub-10ms ANN search with zero network overhead (local mode). No account/server setup. The dataset fits in memory. For a submission-grade project, this removes an entire failure point.

### Why multilingual-e5-small over large?

Speed. The `small` variant gives 384-dim embeddings vs 1024-dim for `large`. Embedding is on the critical path, and the quality difference is marginal for passage retrieval. Keeps embedding latency under 15ms.

### Why hybrid retrieval (dense + BM25)?

Dense retrieval (embeddings) catches semantic matches — "What is India's capital?" matches "Delhi is the capital city of India." But it can miss exact keyword matches that BM25 catches perfectly. Combining both with 70/30 weighting gives the best of both worlds.

### Why 4 chunking strategies instead of 1?

The task explicitly says: "Chunking strategy should be vast — don't submit a single naive fixed-size chunking approach." Each strategy has different strengths:

- **Fixed-size**: reliable baseline, handles any text
- **Sentence-window**: precise retrieval with expanded context
- **Semantic**: topically coherent chunks
- **Metadata-aware**: preserves dataset structure for filtered search

All 4 are applied during ingestion so the vector DB contains a diverse mix of chunk types.

---

## Submission Checklist

- [ ] GitHub repo pushed and public
- [ ] Live working link on HF Spaces
- [ ] `python benchmark.py` output saved (P50/P70/P100 numbers)
- [ ] Video 1: 90s team/process video
- [ ] Video 2: end-to-end demo video
- [ ] Both videos posted on Instagram, X, LinkedIn by every team member with #RAGInGoa
- [ ] Submission form filled: https://forms.gle/MNvCjcv23Hn2Eeu58
