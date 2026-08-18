"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider ---
# Mọi provider dưới đây đều expose API tương thích OpenAI → chỉ cần đổi
# base_url + key, KHÔNG phải sửa code gọi LLM ở M4/M5/pipeline.
#
# Đổi provider: đặt LLM_PROVIDER trong .env (groq | gemini | cerebras |
# openrouter | openai), rồi điền key tương ứng. Có thể override LLM_MODEL.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        # llama-3.3-70b-versatile trả 404 trên account này; qwen3.6-27b rò khối
        # <think> vào content. gpt-oss-20b output sạch nhưng free tier chỉ
        # 200k token/ngày/model — không đủ cho 1 lần main.py trọn vẹn (~400k).
        "model": "openai/gpt-oss-20b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "model": "gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
if LLM_PROVIDER not in PROVIDERS:
    raise ValueError(
        f"LLM_PROVIDER={LLM_PROVIDER!r} không hợp lệ. "
        f"Chọn một trong: {', '.join(PROVIDERS)}"
    )

_p = PROVIDERS[LLM_PROVIDER]
LLM_API_KEY = os.getenv(_p["key_env"], "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or _p["base_url"]
LLM_MODEL = os.getenv("LLM_MODEL", _p["model"])

# Giữ lại cho code cũ / tương thích ngược
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


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
