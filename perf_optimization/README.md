# 🔍 Async Python Recommendation Service

A high-performance, async Python service that demonstrates how to diagnose and eliminate latency bottlenecks using real-world techniques like:

- Precomputation and memory layout optimization
- Async-safe offloading of CPU-bound work
- Approximate nearest neighbor search with FAISS (HNSW)

The service is implemented using FastAPI and shows how architectural decisions impact throughput and latency under concurrent load.

---

## 🚀 Project Structure

This repo includes three progressively optimized versions:

1. `slow_service.py`: Naive baseline implementation with pandas and blocking code
2. `optimized_service_v1.py`: Improves latency with NumPy, precomputation, and `ThreadPoolExecutor`
3. `optimized_service_v2.py`: Replaces brute-force search with FAISS (HNSW) for massive speedups

## 🏃‍♂️ Running the Service

To run the service, you can use need to generate the data through the following command:

```bash
poetry run python generate_data.py
```

Then, you can start the service with:

```bash
poetry run python optimized_service_v2.py
```

## 📈 Benchmarking

To benchmark the service, you can use the provided `benchmark.py` script. It will run a series of requests against the services and measure throughput and latency.