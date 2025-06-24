import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import matplotlib.pyplot as plt

class PerformanceBenchmark:
    """Comprehensive benchmarking suite"""

    def __init__(self):
        pass

    async def benchmark_endpoint(self, endpoint: str, queries: List[str],
                                 concurrent_requests: int = 1,
                                 iterations: int = 10) -> Dict:
        """Benchmark a specific endpoint with multiple queries"""

        async def single_request(session: aiohttp.ClientSession, query: str) -> float:
            start_time = time.perf_counter()
            try:
                async with session.get(f"{endpoint}",
                                       params={"query": query}) as response:
                    await response.json()
                    return time.perf_counter() - start_time
            except Exception as e:
                print(f"Request failed: {e}")
                return float('inf')

        async def concurrent_batch(queries_batch: List[str]) -> List[float]:
            """Run concurrent requests for a batch of queries"""
            async with aiohttp.ClientSession() as session:
                tasks = [single_request(session, query) for query in queries_batch]
                return await asyncio.gather(*tasks)

        # Run benchmark
        all_latencies = []

        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations} for {endpoint}")

            # Create batches for concurrent testing
            query_batches = [queries[i:i+concurrent_requests]
                             for i in range(0, len(queries), concurrent_requests)]

            for batch in query_batches:
                latencies = await concurrent_batch(batch)
                all_latencies.extend(latencies)

                # Brief pause between batches to avoid overwhelming the server
                await asyncio.sleep(0.1)

        # Calculate statistics
        valid_latencies = [lat for lat in all_latencies if lat != float('inf')]

        if not valid_latencies:
            return {"error": "All requests failed"}

        return {
            "endpoint": endpoint,
            "total_requests": len(valid_latencies),
            "concurrent_requests": concurrent_requests,
            "mean_latency": statistics.mean(valid_latencies),
            "median_latency": statistics.median(valid_latencies),
            "p95_latency": statistics.quantiles(valid_latencies, n=20)[18],  # 95th percentile
            "p99_latency": statistics.quantiles(valid_latencies, n=100)[98],  # 99th percentile
            "min_latency": min(valid_latencies),
            "max_latency": max(valid_latencies),
            "throughput": len(valid_latencies) / sum(valid_latencies),
            "raw_data": valid_latencies
        }

    async def compare_all_endpoints(self) -> Dict:
        """Compare all optimization levels"""
        test_queries = [
            "electronics", "books", "clothing", "home",
            "laptop", "smartphone", "headphones", "camera"
        ]

        services = [
            "/recommendations/slow",
            "/recommendations/optimized-v1",
            "/recommendations/optimized-v2"
        ]
        endpoints=[
            "http://localhost:8001/recommendations/slow",
            "http://localhost:8003/recommendations/optimized-v1",
            "http://localhost:8002/recommendations/optimized-v2"
        ]


        results = {}

        for service, endpoint in zip(services,endpoints):
            print(f"\nBenchmarking {endpoint}...")
            result = await self.benchmark_endpoint(
                endpoint=endpoint,
                queries=test_queries,
                concurrent_requests=5,
                iterations=3
            )
            results[service] = result

            if "error" not in result:
                print(f"  Mean latency: {result['mean_latency']:.3f}s")
                print(f"  P95 latency:  {result['p95_latency']:.3f}s")
                print(f"  Throughput:   {result['throughput']:.2f} req/s")

        return results

    def visualize_results(self, results: Dict):
        """Create visualization of performance improvements"""
        endpoints = list(results.keys())
        mean_latencies = [results[ep]['mean_latency'] for ep in endpoints if 'error' not in results[ep]]
        throughputs = [results[ep]['throughput'] for ep in endpoints if 'error' not in results[ep]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Latency comparison
        ax1.bar(range(len(mean_latencies)), mean_latencies)
        ax1.set_xlabel('Optimization Level')
        ax1.set_ylabel('Mean Latency (seconds)')
        ax1.set_title('Latency Comparison')
        ax1.set_xticks(range(len(endpoints)))
        ax1.set_xticklabels(['Original', 'Optimized V1', 'Optimized V2'])

        # Add value labels on bars
        for i, v in enumerate(mean_latencies):
            ax1.text(i, v + 0.1, f'{v:.2f}s', ha='center')

        # Throughput comparison
        ax2.bar(range(len(throughputs)), throughputs)
        ax2.set_xlabel('Optimization Level')
        ax2.set_ylabel('Throughput (requests/second)')
        ax2.set_title('Throughput Comparison')
        ax2.set_xticks(range(len(endpoints)))
        ax2.set_xticklabels(['Original', 'Optimized V1', 'Optimized V2'])

        # Add value labels on bars
        for i, v in enumerate(throughputs):
            ax2.text(i, v + 0.1, f'{v:.1f} req/s', ha='center')

        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

async def main():
    benchmark = PerformanceBenchmark()
    results = await benchmark.compare_all_endpoints()

    # Print detailed results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)

    for endpoint, result in results.items():
        if "error" in result:
            print(f"\n{endpoint}: FAILED")
            continue

        print(f"\n{endpoint}:")
        print(f"  Mean Latency:    {result['mean_latency']:.3f}s")
        print(f"  Median Latency:  {result['median_latency']:.3f}s")
        print(f"  P95 Latency:     {result['p95_latency']:.3f}s")
        print(f"  Throughput:      {result['throughput']:.2f} req/s")

    # Calculate improvements
    if len(results) >= 2:
        slow_latency = results['/recommendations/slow']['mean_latency']
        fast_latency = results['/recommendations/optimized-v2']['mean_latency']

        improvement = (slow_latency - fast_latency) / slow_latency * 100
        speedup = slow_latency / fast_latency

        print(f"\n🎯 OPTIMIZATION RESULTS:")
        print(f"   Latency reduction: {improvement:.1f}%")
        print(f"   Speedup factor: {speedup:.1f}x")

    # Create visualization
    benchmark.visualize_results(results)

if __name__ == "__main__":
    asyncio.run(main())