import os
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
LLM_MODEL = "llama-3.1-8b-instant"
QDRANT_COLLECTION = "msmarco_xi"

TOP_K = 5
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
SENTENCE_WINDOW = 2

RELEVANCE_THRESHOLD = 0.35
GROUNDEDNESS_THRESHOLD = 0.4
OFF_TOPIC_THRESHOLD = 0.25

MAX_RETRIES = 3
RETRY_BACKOFF = 0.1
