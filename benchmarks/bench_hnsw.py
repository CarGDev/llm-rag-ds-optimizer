"""Benchmark script for HNSW."""

import json
from pathlib import Path

import numpy as np

from llmds.hnsw import HNSW
from llmds.utils import Timer


def benchmark_hnsw(
    num_vectors: int = 1000,
    dim: int = 128,
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    num_queries: int = 100,
    output_dir: Path = Path("benchmarks/results"),
):
    """Benchmark HNSW operations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hnsw = HNSW(dim=dim, M=M, ef_construction=ef_construction, ef_search=ef_search)

    # Build index
    build_times = []
    for i in range(num_vectors):
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        with Timer() as t:
            hnsw.add(vec, i)
        build_times.append(t.elapsed * 1000)

    # Search
    search_times = []
    for _ in range(num_queries):
        query = np.random.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        with Timer() as t:
            hnsw.search(query, k=10)
        search_times.append(t.elapsed * 1000)

    stats = hnsw.stats()

    results = {
        "benchmark": "hnsw",
        "num_vectors": num_vectors,
        "dim": dim,
        "M": M,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "num_queries": num_queries,
        "build_p50_ms": sorted(build_times)[len(build_times) // 2],
        "build_p95_ms": sorted(build_times)[int(len(build_times) * 0.95)],
        "build_p99_ms": sorted(build_times)[int(len(build_times) * 0.99)],
        "search_p50_ms": sorted(search_times)[len(search_times) // 2],
        "search_p95_ms": sorted(search_times)[int(len(search_times) * 0.95)],
        "search_p99_ms": sorted(search_times)[int(len(search_times) * 0.99)],
        "hnsw_stats": stats,
    }

    output_file = output_dir / "hnsw_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"HNSW benchmark completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    benchmark_hnsw()

