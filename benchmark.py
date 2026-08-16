"""Run benchmark queries and print P50/P70/P100 latency report."""
import time
from pipeline import run_pipeline, get_tracker, warmup

SAMPLE_QUERIES = [
    "भारत की राजधानी क्या है?",
    "ताज महल कहाँ स्थित है?",
    "सूर्य कितनी दूरी पर है?",
    "पानी का रासायनिक सूत्र क्या है?",
    "भारत के प्रथम प्रधानमंत्री कौन थे?",
    "चंद्रमा पर पहला व्यक्ति कौन गया?",
    "पृथ्वी की आयु कितनी है?",
    "हिमालय कहाँ है?",
    "इंटरनेट का आविष्कार किसने किया?",
    "भारत में कितने राज्य हैं?",
    "गंगा नदी कहाँ से निकलती है?",
    "What is the capital of India?",
    "Where is the Taj Mahal?",
    "How far is the sun from Earth?",
    "Who was the first person on the moon?",
    "What is the chemical formula of water?",
    "How many states are in India?",
    "Where does the Ganges river originate?",
    "Who invented the internet?",
    "How old is the Earth?",
]


def benchmark():
    warmup()  # load embedding model, Qdrant, BM25 index, Groq client once,
              # outside the timed loop — otherwise the first query eats
              # these one-time costs and wrecks P70/P100.
    print(f"\nRunning {len(SAMPLE_QUERIES)} benchmark queries...\n")
    for i, q in enumerate(SAMPLE_QUERIES):
        t0 = time.perf_counter()
        result = run_pipeline(q)
        total = (time.perf_counter() - t0) * 1000
        status = "✓" if result.grounded else "⚠"
        rag_ms = result.latencies.get("embedding", 0) + result.latencies.get("retrieval", 0)
        gen_ms = result.latencies.get("generation", 0)
        print(f"  [{status}] {i+1:2d}. {q[:40]:<42s} RAG: {rag_ms:.0f}ms  Gen: {gen_ms:.0f}ms  Total: {total:.0f}ms")

    print("\n" + "=" * 70)
    print("LATENCY REPORT")
    print("=" * 70)
    print(get_tracker().format_report())
    print()
    print("Note: 'embedding' + 'retrieval' = RAG pipeline latency (target <200ms).")
    print("      'generation' = LLM inference via Groq (network-bound, typically 500-1500ms).")
    print("      STT latency is measured separately for voice queries.")


if __name__ == "__main__":
    benchmark()
