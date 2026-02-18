import time
import hashlib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Enable CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Configuration
# ---------------------------
MODEL_COST_PER_MILLION = 1.0
AVG_TOKENS_PER_REQUEST = 3000
TTL_SECONDS = 86400
MAX_CACHE_SIZE = 1500
SEMANTIC_THRESHOLD = 0.95

# ---------------------------
# Cache & Analytics
# ---------------------------
cache = OrderedDict()

analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0,
}

# ---------------------------
# Request Model
# ---------------------------
class QueryRequest(BaseModel):
    query: str
    application: str

# ---------------------------
# Helpers
# ---------------------------
def md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_embedding(text):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(384)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def summarize(text):
    # Simulate expensive LLM call
    start = time.time()

    # Artificial heavy computation
    for _ in range(5_000_000):
        pass

    return f"Summary: {text[:150]}..."


def remove_expired():
    now = time.time()
    expired = [
        key for key, value in cache.items()
        if now - value["timestamp"] > TTL_SECONDS
    ]
    for key in expired:
        del cache[key]

def evict_if_needed():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

# ---------------------------
# Health Check (IMPORTANT)
# ---------------------------
@app.get("/")
def health_check():
    return {"status": "AI Caching System Running"}

# ---------------------------
# Main Query Endpoint
# ---------------------------
@app.post("/")
def process_query(payload: QueryRequest):

    start_time = time.time()
    analytics["totalRequests"] += 1

    query = payload.query
    key = md5_hash(query)

    remove_expired()

    # Exact match
    if key in cache:
        analytics["cacheHits"] += 1
        cache.move_to_end(key)
        latency = max(1, int((time.time() - start_time) * 1000))

        return {
            "answer": cache[key]["response"],
            "cached": True,
            "latency": latency,
            "cacheKey": key
        }

    # Semantic match
    query_embedding = get_embedding(query)

    for cached_key, value in cache.items():
        similarity = cosine_similarity(query_embedding, value["embedding"])
        if similarity > SEMANTIC_THRESHOLD:
            analytics["cacheHits"] += 1
            cache.move_to_end(cached_key)
            latency = max(1, int((time.time() - start_time) * 1000))

            return {
                "answer": value["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": cached_key
            }

    # Cache miss
    analytics["cacheMisses"] += 1
    response = summarize(query)

    cache[key] = {
        "response": response,
        "embedding": query_embedding,
        "timestamp": time.time()
    }

    evict_if_needed()

    latency = max(1, int((time.time() - start_time) * 1000))

    return {
        "answer": response,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }

# ---------------------------
# Analytics Endpoint (REQUIRED)
# ---------------------------
@app.get("/analytics")
def get_analytics():

    total = analytics["totalRequests"]
    hits = analytics["cacheHits"]
    misses = analytics["cacheMisses"]

    hit_rate = hits / total if total > 0 else 0
    savings_percent = hit_rate * 100

    baseline_cost = (total * AVG_TOKENS_PER_REQUEST * MODEL_COST_PER_MILLION) / 1_000_000
    actual_cost = (misses * AVG_TOKENS_PER_REQUEST * MODEL_COST_PER_MILLION) / 1_000_000
    savings = baseline_cost - actual_cost

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "costSavings": round(savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": [
            "exact match caching",
            "semantic similarity caching",
            "LRU eviction",
            "TTL expiration"
        ]
    }


