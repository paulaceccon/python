# optimized_service_v1.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

# Global optimized data structures
optimized_data = {}
thread_pool = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global optimized_data
    print("Building optimized data structures...")

    # Load raw data
    products_df = pd.read_parquet("data/products.parquet")

    # Pre-build optimized structures
    optimized_data = {
        "embeddings_matrix": np.array(products_df["embedding"].tolist()),
        "product_lookup": {
            i: {
                "product_id": row["product_id"],
                "name": row["name"],
                "category": row["category"],
                "price": row["price"],
            }
            for i, row in products_df.iterrows()
        },
        "category_index": build_category_index(products_df),
    }

    print(f"Optimized structures built for {len(products_df)} products")
    yield


def build_category_index(products_df):
    """Pre-build category lookup to avoid O(n) DataFrame operations"""
    category_index = {}
    for idx, row in products_df.iterrows():
        category = row["category"]
        if category not in category_index:
            category_index[category] = []
        category_index[category].append(idx)
    return category_index


app = FastAPI(lifespan=lifespan)


class OptimizedRecommendationServiceV1:
    """First optimization: pre-computation and async handling"""

    @staticmethod
    async def generate_query_embedding_async(query: str) -> np.ndarray:
        """Move embedding generation to thread pool"""

        def _generate_embedding():
            time.sleep(0.1)  # Simulate API call
            np.random.seed(hash(query) % 2**32)
            embedding = np.random.normal(0, 1, 384)
            return embedding / np.linalg.norm(embedding)

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(thread_pool, _generate_embedding)

    @staticmethod
    async def optimized_similarity_search_v1(query: str, top_k: int = 10) -> list:
        """First optimization: eliminate repeated computation"""
        start_time = time.perf_counter()

        # Step 1: Generate embedding asynchronously
        step1_start = time.perf_counter()
        query_embedding = (
            await OptimizedRecommendationServiceV1.generate_query_embedding_async(query)
        )
        step1_time = time.perf_counter() - step1_start

        # Step 2: Use pre-computed embeddings matrix (major improvement!)
        step2_start = time.perf_counter()
        # Why this is now fast:
        # 1. embeddings_matrix already exists in optimal format
        # 2. No memory allocations or copies needed
        # 3. Direct memory access to contiguous array
        # 4. CPU can use vectorized operations efficiently
        embeddings_matrix = optimized_data["embeddings_matrix"]  # Already built!
        step2_time = time.perf_counter() - step2_start

        # Step 3: Compute similarity (still expensive but unavoidable with exact search)
        step3_start = time.perf_counter()

        def _compute_similarity():
            return cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)[
                0
            ]

        # Why use thread pool here:
        # 1. This is pure CPU work that can't be pre-computed (depends on query)
        # 2. Moving to thread pool keeps event loop responsive
        # 3. Other requests can be processed while this computes
        # 4. BUT: This doesn't make individual requests faster!
        loop = asyncio.get_event_loop()
        similarities = await loop.run_in_executor(thread_pool, _compute_similarity)
        step3_time = time.perf_counter() - step3_start

        # Step 4: Use pre-built lookup tables
        step4_start = time.perf_counter()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Why dictionary lookups are faster than DataFrame access:
        # 1. O(1) hash table lookup vs O(1) index lookup with high overhead
        # 2. No pandas machinery - direct Python dict access
        # 3. Data already in optimal format (no conversion needed)
        # 4. Better memory locality for small lookups
        results = []
        product_lookup = optimized_data["product_lookup"]

        for idx in top_indices:
            product = product_lookup[idx]  # O(1) lookup instead of DataFrame access
            if product["price"] > 20:
                results.append({**product, "similarity": float(similarities[idx])})

        step4_time = time.perf_counter() - step4_start
        total_time = time.perf_counter() - start_time

        print("Optimized V1 breakdown:")
        print(f"  Step 1 (Embedding):     {step1_time:.3f}s")
        print(f"  Step 2 (Matrix access): {step2_time:.3f}s")  # Should be ~0.000s
        print(f"  Step 3 (Similarity):    {step3_time:.3f}s")  # Still expensive
        print(f"  Step 4 (Processing):    {step4_time:.3f}s")
        print(f"  Total time:             {total_time:.3f}s")

        return results


@app.get("/recommendations/optimized-v1")
async def get_optimized_recommendations_v1(query: str = "electronics"):
    """First optimization: pre-computation and proper async"""
    start_time = time.perf_counter()

    results = await OptimizedRecommendationServiceV1.optimized_similarity_search_v1(
        query
    )

    total_time = time.perf_counter() - start_time

    return {
        "results": results,
        "metadata": {
            "query": query,
            "total_time": f"{total_time:.3f}s",
            "num_results": len(results),
            "optimization": "v1",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
