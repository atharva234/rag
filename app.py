import gradio as gr
from pipeline import run_pipeline, run_pipeline_stream, run_pipeline_voice, get_tracker, warmup

EXAMPLES = [
    "भारत की राजधानी क्या है?",
    "ताज महल कहाँ स्थित है?",
    "गंगा नदी कहाँ से निकलती है?",
    "Who invented the internet?",
    "Where is the Himalaya?",
]

HEADER = """
# 🎙️ Voice-Enabled RAG — MSMARCO-XI
**Stack:** Sarvam Saaras v3 (STT) · Groq Llama 3.1-8B (LLM) · Qdrant (vector DB) · multilingual-E5 (embeddings) · BM25 (sparse) · 4-strategy chunking
"""

FOOTER = """
---
Built for HH Goa 2026 · Dataset: [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) · #RAGInGoa
"""


def format_chunks(hits):
    if not hits:
        return ""
    parts = []
    for i, h in enumerate(hits[:3]):
        strategy = h.metadata.get("chunking_strategy", "?")
        score = f"{h.score:.3f}"
        parts.append(f"**[{i+1}]** `score={score}` `strategy={strategy}`\n\n{h.text[:300]}")
    return "\n\n---\n\n".join(parts)


def handle_text(query: str):
    if not query.strip():
        yield "Please enter a question.", "", ""
        return
    for answer, latency_md, chunks_md in run_pipeline_stream(query):
        yield answer, latency_md, chunks_md


def handle_voice(audio):
    if audio is None:
        return "No audio recorded.", "", ""
    result = run_pipeline_voice(audio)
    lat = result.latencies
    rag_ms = lat.get("embedding", 0) + lat.get("retrieval", 0)
    latency_md = (
        f"**Transcription:** {result.query_text}\n\n"
        f"**STT:** {lat.get('stt', 0):.0f}ms · **RAG pipeline:** {rag_ms:.0f}ms · "
        f"**Generation:** {lat.get('generation', 0):.0f}ms · **Total:** {result.total_latency_ms:.0f}ms"
    )
    return result.answer, latency_md, format_chunks(result.retrieved_chunks)


def show_analytics():
    return get_tracker().format_report()


with gr.Blocks(theme=gr.themes.Soft(), title="Voice RAG — MSMARCO-XI") as app:
    gr.Markdown(HEADER)

    with gr.Tab("Text Query"):
        txt_in = gr.Textbox(label="Question", placeholder="Type your question in Hindi or English...")
        gr.Examples(examples=EXAMPLES, inputs=txt_in, label="Example queries")
        txt_btn = gr.Button("Ask", variant="primary")
        txt_answer = gr.Markdown(label="Answer")
        txt_latency = gr.Markdown(label="Latency")
        with gr.Accordion("Retrieved chunks", open=False):
            txt_chunks = gr.Markdown()
        txt_btn.click(handle_text, inputs=txt_in, outputs=[txt_answer, txt_latency, txt_chunks])

    with gr.Tab("Voice Query"):
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Record your question")
        voice_btn = gr.Button("Ask", variant="primary")
        voice_answer = gr.Markdown(label="Answer")
        voice_latency = gr.Markdown(label="Latency")
        with gr.Accordion("Retrieved chunks", open=False):
            voice_chunks = gr.Markdown()
        voice_btn.click(handle_voice, inputs=audio_in, outputs=[voice_answer, voice_latency, voice_chunks])

    with gr.Tab("Latency Analytics"):
        analytics_btn = gr.Button("Refresh")
        analytics_out = gr.Textbox(label="P50 / P70 / P100 per stage", lines=12, interactive=False)
        analytics_btn.click(show_analytics, outputs=analytics_out)

    gr.Markdown(FOOTER)


if __name__ == "__main__":
    warmup()
    app.launch(server_name="0.0.0.0", server_port=7860)
