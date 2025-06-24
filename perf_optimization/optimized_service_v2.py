import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List

import hnswlib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI

optimized_data = {}
thread_pool = ThreadPoolExecutor(max_workers=4)


class OptimizedRecommendationServiceV2:
    """Second optimization: Approximate Nearest Neighbors using HNSWLib"""

    def __init__(self):
        self.hnsw_index = None
        self.product_lookup = {}

    @staticmethod
    async def generate_query_embedding_async(query: str) -> np.ndarray:
        """Simulate embedding generation with delay, offloaded to thread pool"""

        def _generate_embedding():
            time.sleep(0.1)  # Simulate embedding API call
            np.random.seed(hash(query) % 2**32)
            embedding = np.random.normal(0, 1, 384)
            return embedding / np.linalg.norm(embedding)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(thread_pool, _generate_embedding)

    def build_hnsw_index(self, embeddings: np.ndarray, products_df: pd.DataFrame):
        """Build HNSW index for fast approximate search using cosine distance"""
        print("Building HNSWLib index...")

        dim = embeddings.shape[1]
        num_elements = embeddings.shape[0]

        # Initialize the index
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=32)
        index.add_items(embeddings, np.arange(num_elements))
        index.set_ef(100)

        self.hnsw_index = index

        # Build lookup table
        self.product_lookup = {
            i: {
                "product_id": row["product_id"],
                "name": row["name"],
                "category": row["category"],
                "price": row["price"],
            }
            for i, row in products_df.iterrows()
        }

        print(f"HNSWLib index built with {num_elements} vectors.")

    async def hnsw_similarity_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Fast similarity search using HNSWLib"""
        start_time = time.perf_counter()

        # Step 1: Embedding
        step1_start = time.perf_counter()
        query_embedding = await self.generate_query_embedding_async(query)
        step1_time = time.perf_counter() - step1_start

        # Step 2: HNSW Search
        step2_start = time.perf_counter()

        def _hnsw_search():
            labels, distances = self.hnsw_index.knn_query(
                query_embedding.reshape(1, -1), k=top_k * 2
            )
            return labels[0], distances[0]

        loop = asyncio.get_event_loop()
        indices, distances = await loop.run_in_executor(thread_pool, _hnsw_search)
        step2_time = time.perf_counter() - step2_start

        # Step 3: Post-processing
        step3_start = time.perf_counter()
        results = []

        for idx, distance in zip(indices, distances):
            product = self.product_lookup[idx]
            if product["price"] > 20:
                results.append(
                    {
                        **product,
                        "similarity": float(
                            1 - distance
                        ),  # Convert cosine distance back to similarity
                    }
                )

            if len(results) >= top_k:
                break

        step3_time = time.perf_counter() - step3_start
        total_time = time.perf_counter() - start_time

        print("HNSWLib V2 breakdown:")
        print(f"  Step 1 (Embedding):     {step1_time:.3f}s")
        print(f"  Step 2 (ANN Search):    {step2_time:.3f}s")
        print(f"  Step 3 (Processing):    {step3_time:.3f}s")
        print(f"  Total time:             {total_time:.3f}s")

        return results


# FastAPI app with HNSW index
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommendation_service_v2

    print("Building optimized data structures V2 (HNSW)...")
    products_df = pd.read_parquet("data/products.parquet")
    embeddings = np.array(products_df["embedding"].tolist())

    recommendation_service_v2 = OptimizedRecommendationServiceV2()
    recommendation_service_v2.build_hnsw_index(embeddings, products_df)

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/recommendations/optimized-v2")
async def get_optimized_recommendations_v2(query: str = "electronics"):
    """HNSWLib-based approximate similarity search"""
    start_time = time.perf_counter()

    results = await recommendation_service_v2.hnsw_similarity_search(query)

    total_time = time.perf_counter() - start_time

    return {
        "results": results,
        "metadata": {
            "query": query,
            "total_time": f"{total_time:.3f}s",
            "num_results": len(results),
            "optimization": "v2-hnswlib",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
