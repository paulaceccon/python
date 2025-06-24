import time
import asyncio
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Global variables to simulate production service state
products_df = None
embeddings_cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global products_df, embeddings_cache
    print("Loading data...")
    products_df = pd.read_parquet('data/products.parquet')
    with open('data/embeddings.pkl', 'rb') as f:
        embeddings_cache = pickle.load(f)
    print(f"Loaded {len(products_df)} products")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class SlowRecommendationService:
    """Demonstrates common performance anti-patterns"""

    @staticmethod
    def generate_query_embedding(query: str) -> np.ndarray:
        """Simulate embedding generation (e.g., from OpenAI, Sentence Transformers)"""
        # Simulate API call latency
        time.sleep(0.1)  # 100ms API call

        # Generate random embedding for demo
        np.random.seed(hash(query) % 2**32)
        embedding = np.random.normal(0, 1, 384)
        return embedding / np.linalg.norm(embedding)

    @staticmethod
    def slow_similarity_search(query: str, top_k: int = 10) -> list:
        """The main performance bottleneck - demonstrates multiple anti-patterns"""
        start_time = time.perf_counter()

        # Anti-pattern 1: Generate embedding in main thread (blocking)
        step1_start = time.perf_counter()
        query_embedding = SlowRecommendationService.generate_query_embedding(query)
        step1_time = time.perf_counter() - step1_start

        # Anti-pattern 2: Recreate embeddings matrix every request
        step2_start = time.perf_counter()
        # This is the killer: converting DataFrame column to numpy array
        # Why this is so slow:
        # 1. products_df['embedding'].values creates object array (100K Python objects)
        # 2. np.vstack() must iterate through each object and copy data
        # 3. Memory allocations: 100K small arrays → 1 large array
        # 4. No memory locality, cache misses everywhere
        embedding_matrix = np.vstack(products_df['embedding'].values)
        step2_time = time.perf_counter() - step2_start

        # Anti-pattern 3: Compute similarity against ALL vectors
        step3_start = time.perf_counter()
        # Why this is expensive:
        # 1. Matrix multiplication: query (1x384) @ embeddings (100K x 384)
        # 2. That's 38.4 million floating point operations
        # 3. With 100K vectors, we compute 100K similarities but only need top 10
        # 4. No early termination possible with exact cosine similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            embedding_matrix
        )[0]
        step3_time = time.perf_counter() - step3_start

        # Anti-pattern 4: Multiple DataFrame operations
        step4_start = time.perf_counter()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Why DataFrame operations are slow in production:
        # 1. products_df.iloc[idx] -> Index lookup with bounds checking
        # 2. Row conversion to dict/object -> Python object creation overhead
        # 3. No memory locality -> Each access potentially a cache miss
        # 4. Pandas overhead for each field access (product['price'])
        results = []
        for idx in top_indices:
            product = products_df.iloc[idx]  # O(1) but with high constant factor
            # Simulate additional filtering/processing
            if product['price'] > 20:  # More DataFrame access
                results.append({
                    'product_id': product['product_id'],
                    'name': product['name'],
                    'category': product['category'],
                    'price': product['price'],
                    'similarity': similarities[idx]
                })
        step4_time = time.perf_counter() - step4_start

        total_time = time.perf_counter() - start_time

        print(f"Performance breakdown:")
        print(f"  Step 1 (Embedding):     {step1_time:.3f}s")
        print(f"  Step 2 (Matrix build):  {step2_time:.3f}s")
        print(f"  Step 3 (Similarity):    {step3_time:.3f}s")
        print(f"  Step 4 (Processing):    {step4_time:.3f}s")
        print(f"  Total time:             {total_time:.3f}s")

        return results

@app.get("/recommendations/slow")
async def get_slow_recommendations(query: str = "electronics"):
    """Endpoint demonstrating the performance problem"""
    start_time = time.perf_counter()

    # This blocks the event loop!
    results = SlowRecommendationService.slow_similarity_search(query)

    total_time = time.perf_counter() - start_time

    return {
        "results": results,
        "metadata": {
            "query": query,
            "total_time": f"{total_time:.3f}s",
            "num_results": len(results)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)



# Benchmarking 127.0.0.1 (be patient).....done
#
#
# Server Software:        uvicorn
# Server Hostname:        127.0.0.1
# Server Port:            8000
#
# Document Path:          /recommendations/slow?query=electronics
# Document Length:        1421 bytes
#
# Concurrency Level:      5
# Time taken for tests:   10.233 seconds
# Complete requests:      10
# Failed requests:        0
# Total transferred:      15480 bytes
# HTML transferred:       14210 bytes
# Requests per second:    0.98 [#/sec] (mean)
#     Time per request:       5116.644 [ms] (mean)
# Time per request:       1023.329 [ms] (mean, across all concurrent requests)
# Transfer rate:          1.48 [Kbytes/sec] received
#
# Connection Times (ms)
# min  mean[+/-sd] median   max
# Connect:        0    0   0.7      0       2
# Processing:   994 3659 1716.7   5001    5116
# Waiting:      994 3659 1716.8   5001    5116
# Total:        994 3660 1716.8   5001    5117
#
# Percentage of the requests served within a certain time (ms)
# 50%   5001
# 66%   5037
# 75%   5100
# 80%   5110
# 90%   5117
# 95%   5117
# 98%   5117
# 99%   5117
# 100%   5117 (longest request)

# INFO:     127.0.0.1:65021 - "GET /recommendations/slow?query=electronics HTTP/1.0" 200 OK
# Performance breakdown:
# Step 1 (Embedding):     0.100s
# Step 2 (Matrix build):  0.353s
# Step 3 (Similarity):    0.441s
# Step 4 (Processing):    0.051s
# Total time:             0.946s


