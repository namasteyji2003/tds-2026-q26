import time
import hashlib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict

app = FastAPI()

# ---------------------------
# Configuration
# ---------------------------
MODEL_COST_PER_MILLION = 1.0
AVG_TOKENS_PER_REQUEST = 3000
TTL_SECONDS = 86400  # 24 hours
MAX_CACHE_SIZE = 1500
SEMANTIC_THRESHOLD = 0.95

# ---------------------------
# In-Memory Cache (LRU)
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
# Helper Functions
# ---------------------------
def md5_hash(text: str):
    return hashlib.md5(text.encode()).hexdigest()

def get_embedding(text: str):
    # Lightweight deterministic embedding (no ML model needed)
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(384)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def summarize(text: str):
    # Simulated LLM call
    time.sleep(1.5)  # simulate latency
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
        cache.popitem(last=False)  # Remove LRU

# ---------------------------
# Main Endpoint
# ---------------------------
@app.post("/")
def process_query(payload: QueryRequest):

    start_time = time.time()
    analytics["totalRequests"] += 1

    query = payload.query
    key = md5_hash(query)

    # Remove expired cache entries
    remove_expired()

    # 1️⃣ Exact Match Cache
    if key in cache:
        analytics["cacheHits"] += 1
        cache.move_to_end(key)
        latency = int((time.time() - start_time) * 1000)

        return {
            "answer": cache[key]["response"],
            "cached": True,
            "latency": latency,
            "cacheKey": key
        }

    # 2️⃣ Semantic Cache
    query_embedding = get_embedding(query)

    for cached_key, value in cache.items():
        similarity = cosine_similarity(query_embedding, value["embedding"])
        if similarity > SEMANTIC_THRESHOLD:
            analytics["cacheHits"] += 1
            cache.move_to_end(cached_key)
            latency = int((time.time() - start_time) * 1000)

            return {
                "answer": value["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": cached_key
            }

    # 3️⃣ Cache Miss → Simulated LLM Call
    analytics["cacheMisses"] += 1

    response = summarize(query)

    cache[key] = {
        "response": response,
        "embedding": query_embedding,
        "timestamp": time.time()
    }

    evict_if_needed()

    latency = int((time.time() - start_time) * 1000)

    return {
        "answer": response,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }

# ---------------------------
# Analytics Endpoint
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
