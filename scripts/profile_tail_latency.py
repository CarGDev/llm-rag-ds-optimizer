"""Profile tail latency breakdown for retrieval pipeline.

This script profiles latency components to identify bottlenecks causing
extreme P99 tail latencies.
"""

import cProfile
import pstats
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np

from llmds.hnsw import HNSW
from llmds.retrieval_pipeline import RetrievalPipeline


def profile_hnsw_search(num_vectors: int = 10000, dim: int = 128, num_queries: int = 1000):
    """Profile HNSW search operations."""
    print(f"Profiling HNSW search with {num_vectors} vectors, dim={dim}, {num_queries} queries...")
    
    np.random.seed(42)
    hnsw = HNSW(dim=dim, M=16, ef_construction=200, ef_search=50, seed=42)
    
    # Build index
    vectors = []
    for i in range(num_vectors):
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
        hnsw.add(vec, i)
    
    # Profile search operations
    profiler = cProfile.Profile()
    profiler.enable()
    
    search_times = []
    for _ in range(num_queries):
        query = np.random.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        import time
        start = time.perf_counter()
        results = hnsw.search(query, k=10)
        elapsed = time.perf_counter() - start
        search_times.append(elapsed * 1000)  # Convert to ms
    
    profiler.disable()
    
    # Compute latency statistics
    search_times.sort()
    p50 = search_times[len(search_times) // 2]
    p95 = search_times[int(len(search_times) * 0.95)]
    p99 = search_times[int(len(search_times) * 0.99)]
    p99_9 = search_times[int(len(search_times) * 0.999)] if len(search_times) >= 1000 else p99
    
    print(f"\nHNSW Search Latency Statistics:")
    print(f"  P50:  {p50:.3f} ms")
    print(f"  P95:  {p95:.3f} ms")
    print(f"  P99:  {p99:.3f} ms")
    print(f"  P99.9: {p99_9:.3f} ms")
    print(f"  Mean: {statistics.mean(search_times):.3f} ms")
    print(f"  Max:  {max(search_times):.3f} ms")
    
    # Analyze P99 outliers
    threshold = p95 * 2  # Outliers are 2x P95
    outliers = [t for t in search_times if t > threshold]
    if outliers:
        print(f"\n  Outliers (>2x P95): {len(outliers)} queries ({len(outliers)/len(search_times)*100:.1f}%)")
        print(f"    Outlier P50: {statistics.median(outliers):.3f} ms")
        print(f"    Outlier Max: {max(outliers):.3f} ms")
    
    # Generate profiling report
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    
    print("\nTop 20 functions by cumulative time:")
    print("=" * 80)
    stats.print_stats(20)
    
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "p99_9_ms": p99_9,
        "mean_ms": statistics.mean(search_times),
        "max_ms": max(search_times),
        "outlier_count": len(outliers),
        "outlier_percent": len(outliers) / len(search_times) * 100 if search_times else 0,
    }


def profile_retrieval_pipeline(num_docs: int = 5000, num_queries: int = 500):
    """Profile complete retrieval pipeline."""
    print(f"\nProfiling RetrievalPipeline with {num_docs} docs, {num_queries} queries...")
    
    np.random.seed(42)
    random = np.random.RandomState(42)
    
    pipeline = RetrievalPipeline(embedding_dim=128, seed=42)
    
    # Build index
    for i in range(num_docs):
        text = f"document {i} about topic {i % 10}"
        embedding = random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.add_document(doc_id=i, text=text, embedding=embedding)
    
    # Profile search operations
    profiler = cProfile.Profile()
    profiler.enable()
    
    search_times = []
    for _ in range(num_queries):
        query_text = "document topic"
        query_embedding = random.randn(128).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        import time
        start = time.perf_counter()
        results = pipeline.search(
            query_text, query_embedding=query_embedding, top_k=10
        )
        elapsed = time.perf_counter() - start
        search_times.append(elapsed * 1000)  # Convert to ms
    
    profiler.disable()
    
    # Compute latency statistics
    search_times.sort()
    p50 = search_times[len(search_times) // 2]
    p95 = search_times[int(len(search_times) * 0.95)]
    p99 = search_times[int(len(search_times) * 0.99)]
    
    print(f"\nRetrieval Pipeline Latency Statistics:")
    print(f"  P50:  {p50:.3f} ms")
    print(f"  P95:  {p95:.3f} ms")
    print(f"  P99:  {p99:.3f} ms")
    print(f"  Mean: {statistics.mean(search_times):.3f} ms")
    print(f"  Max:  {max(search_times):.3f} ms")
    
    # Generate profiling report
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    
    print("\nTop 20 functions by cumulative time:")
    print("=" * 80)
    stats.print_stats(20)
    
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": statistics.mean(search_times),
        "max_ms": max(search_times),
    }


def profile_latency_breakdown(num_vectors: int = 5000, dim: int = 128):
    """Profile latency breakdown by component."""
    print(f"\nProfiling latency breakdown with {num_vectors} vectors...")
    
    np.random.seed(42)
    hnsw = HNSW(dim=dim, M=16, ef_construction=200, ef_search=50, seed=42)
    
    # Build index
    vectors = []
    for i in range(num_vectors):
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
        hnsw.add(vec, i)
    
    # Profile individual operations
    import time
    
    search_times = []
    distance_computation_times = []
    
    for _ in range(100):
        query = np.random.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Profile distance computations
        dist_start = time.perf_counter()
        distances = [np.linalg.norm(query - vec) for vec in vectors[:100]]
        dist_time = (time.perf_counter() - dist_start) * 1000
        distance_computation_times.append(dist_time)
        
        # Profile search
        search_start = time.perf_counter()
        results = hnsw.search(query, k=10)
        search_time = (time.perf_counter() - search_start) * 1000
        search_times.append(search_time)
    
    print(f"\nLatency Breakdown:")
    print(f"  Distance computation: {statistics.mean(distance_computation_times):.3f} ms (mean)")
    print(f"  HNSW search: {statistics.mean(search_times):.3f} ms (mean)")
    print(f"  Search/Distance ratio: {statistics.mean(search_times) / statistics.mean(distance_computation_times):.2f}x")


def main():
    """Run all profiling tasks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile tail latency")
    parser.add_argument("--output", type=Path, default=Path("audit/tail_latency_profile.txt"),
                       help="Output file for profiling report")
    parser.add_argument("--num-vectors", type=int, default=10000,
                       help="Number of vectors for HNSW profiling")
    parser.add_argument("--num-docs", type=int, default=5000,
                       help="Number of documents for pipeline profiling")
    parser.add_argument("--num-queries", type=int, default=1000,
                       help="Number of queries to run")
    args = parser.parse_args()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Redirect output to file
    import sys
    with open(args.output, "w") as f:
        sys.stdout = f
        try:
            # Profile HNSW
            hnsw_stats = profile_hnsw_search(args.num_vectors, 128, args.num_queries)
            
            # Profile pipeline
            pipeline_stats = profile_retrieval_pipeline(args.num_docs, args.num_queries // 2)
            
            # Breakdown
            profile_latency_breakdown(args.num_vectors, 128)
        finally:
            sys.stdout = sys.__stdout__
    
    print(f"\nProfiling complete. Report saved to: {args.output}")
    print(f"\nKey Findings:")
    print(f"  HNSW P99: {hnsw_stats['p99_ms']:.3f} ms")
    print(f"  Pipeline P99: {pipeline_stats['p99_ms']:.3f} ms")
    
    if hnsw_stats.get("outlier_count", 0) > 0:
        print(f"  HNSW Outliers: {hnsw_stats['outlier_count']} ({hnsw_stats['outlier_percent']:.1f}%)")


if __name__ == "__main__":
    main()

