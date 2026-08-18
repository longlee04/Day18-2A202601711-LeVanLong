"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider (Groq mặc định) ---
# Groq expose OpenAI-compatible API → vẫn dùng SDK `openai`, chỉ đổi base_url.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if LLM_PROVIDER == "groq":
    LLM_API_KEY = GROQ_API_KEY
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    # [GIẢ ĐỊNH] gpt-oss-20b: llama-3.3-70b-versatile trả 404 trên account này;
    # gpt-oss-120b chất lượng cao hơn nhưng đã cạn TPD 200k/ngày của free tier;
    # qwen3.6-27b rò rỉ khối <think> vào content nên không dùng để sinh câu trả lời.
    # gpt-oss-20b: output sạch, nhanh (~101s/8 RAGAS job, 0 cell lỗi), còn nguyên quota.
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
else:
    LLM_API_KEY = OPENAI_API_KEY
    LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# Groq free tier: TPM thấp (8000 với gpt-oss-120b) → 429 rất thường xuyên.
# SDK tự backoff theo header retry-after nếu max_retries đủ lớn.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))

# [GIẢ ĐỊNH] Groq free tier giới hạn 200k token/ngày/model. Enrichment tốn ~79k
# (105 chunk × 1 call) — đủ để làm cạn quota trước khi RAGAS chạy xong. Đặt
# ENRICH_WITH_LLM=0 để M5 dùng fallback extractive offline, dành quota cho M4.
ENRICH_WITH_LLM = os.getenv("ENRICH_WITH_LLM", "1") not in ("0", "false", "False")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))


def get_llm_client():
    """OpenAI-compatible client trỏ tới provider đang chọn."""
    from openai import OpenAI

    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                  max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Device cho toàn bộ SentenceTransformer/CrossEncoder.
# [GIẢ ĐỊNH] Default "cpu": pipeline load 3 model cùng lúc (MiniLM + bge-m3 +
# bge-reranker-v2-m3) ≈ 4.6GB VRAM → OOM trên GPU 4GB. Set MODEL_DEVICE=cuda nếu đủ VRAM.
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
