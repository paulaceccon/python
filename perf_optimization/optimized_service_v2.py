import faiss
import numpy as np
from typing import Dict, List, Tuple
from fastapi import FastAPI
import time
import pandas as pd
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

optimized_data = {}
thread_pool = ThreadPoolExecutor(max_workers=4)  # Adjust based on your system

class OptimizedRecommendationServiceV2:
    """Second optimization: Approximate Nearest Neighbors for massive speedup"""

    def __init__(self):
        self.faiss_index = None
        self.product_lookup = {}

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

    def build_faiss_index(self, embeddings: np.ndarray, products_df):
        """Build FAISS index for fast approximate similarity search"""
        print("Building FAISS index...")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))

        # Create FAISS index
        dimension = embeddings.shape[1]

        # Why HNSW for 100K vectors:
        # 1. HNSW (Hierarchical Navigable Small World) builds a multi-layer graph
        # 2. Each layer has exponentially fewer connections than the layer below
        # 3. Search starts at the top (sparse) layer and works down
        # 4. Only computes similarities for promising candidates
        # 5. Excellent accuracy/speed tradeoff for this scale

        # For 100K vectors, use HNSW (Hierarchical Navigable Small World)
        # This gives excellent speed/accuracy tradeoff
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        self.faiss_index.hnsw.efConstruction = 200  # Build-time accuracy
        self.faiss_index.hnsw.efSearch = 100        # Search-time accuracy

        # Add embeddings to index
        self.faiss_index.add(embeddings.astype('float32'))

        # Build product lookup
        self.product_lookup = {
            i: {
                'product_id': row['product_id'],
                'name': row['name'],
                'category': row['category'],
                'price': row['price']
            }
            for i, row in products_df.iterrows()
        }

        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    async def faiss_similarity_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Ultra-fast approximate similarity search"""
        start_time = time.perf_counter()

        # Step 1: Generate embedding (same as before)
        step1_start = time.perf_counter()
        query_embedding = await self.generate_query_embedding_async(query)
        step1_time = time.perf_counter() - step1_start

        # Step 2: FAISS search (this is where the magic happens)
        step2_start = time.perf_counter()

        def _faiss_search():
            query_vector = query_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)

            # The key insight: Only compute similarities for top candidates
            # Instead of 100K similarity computations, HNSW typically does ~50-100
            # Returns the top_k most similar vectors directly
            similarities, indices = self.faiss_index.search(query_vector, top_k * 2)  # Get extra for filtering
            return similarities[0], indices[0]

        loop = asyncio.get_event_loop()
        similarities, indices = await loop.run_in_executor(thread_pool, _faiss_search)
        step2_time = time.perf_counter() - step2_start

        # Step 3: Build results (very fast since we only have top_k items)
        step3_start = time.perf_counter()
        results = []

        for similarity, idx in zip(similarities, indices):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            product = self.product_lookup[idx]
            if product['price'] > 20:
                results.append({
                    **product,
                    'similarity': float(similarity)
                })

            if len(results) >= top_k:
                break

        step3_time = time.perf_counter() - step3_start
        total_time = time.perf_counter() - start_time

        print(f"FAISS V2 breakdown:")
        print(f"  Step 1 (Embedding):     {step1_time:.3f}s")
        print(f"  Step 2 (FAISS search):  {step2_time:.3f}s")  # Should be ~0.001-0.010s
        print(f"  Step 3 (Processing):    {step3_time:.3f}s")
        print(f"  Total time:             {total_time:.3f}s")

        return results

# Add FAISS setup to lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommendation_service_v2

    print("Building optimized data structures V2...")
    products_df = pd.read_parquet('data/products.parquet')
    embeddings = np.array(products_df['embedding'].tolist())

    # Initialize FAISS-based service
    recommendation_service_v2 = OptimizedRecommendationServiceV2()
    recommendation_service_v2.build_faiss_index(embeddings, products_df)

    yield


app = FastAPI(lifespan=lifespan)

@app.get("/recommendations/optimized-v2")
async def get_optimized_recommendations_v2(query: str = "electronics"):
    """Second optimization: FAISS approximate search"""
    start_time = time.perf_counter()

    results = await recommendation_service_v2.faiss_similarity_search(query)

    total_time = time.perf_counter() - start_time

    return {
        "results": results,
        "metadata": {
            "query": query,
            "total_time": f"{total_time:.3f}s",
            "num_results": len(results),
            "optimization": "v2-faiss"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)