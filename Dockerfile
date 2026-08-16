FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
# Install CPU-only PyTorch first — the default pip install pulls in the full
# CUDA/GPU build (multiple large nvidia-* packages), which wastes hundreds of
# MB of RAM on CPU-only instances and can cause OOM kills.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["sh", "-c", "[ -d qdrant_data ] || python ingest.py && python app.py"]