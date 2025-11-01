"""Benchmark script for HNSW."""

import json
from pathlib import Path

import numpy as np

from llmds.hnsw import HNSW
from llmds.utils import Timer, memory_profiler


def compute_ground_truth(
    query: np.ndarray, vectors: np.ndarray, k: int
) -> list[int]:
    """
    Compute ground truth nearest neighbors using exact distance computation.
    
    Args:
        query: Query vector
        vectors: All vectors in the dataset (N x dim)
        k: Number of nearest neighbors to return
        
    Returns:
        List of vector IDs (indices) sorted by distance (ascending)
    """
    # Compute L2 distances
    distances = np.linalg.norm(vectors - query, axis=1)
    
    # Get top-k indices
    top_k_indices = np.argsort(distances)[:k].tolist()
    
    return top_k_indices


def compute_recall(retrieved_ids: list[int], ground_truth_ids: list[int], k: int) -> float:
    """
    Compute recall@k.
    
    Args:
        retrieved_ids: IDs returned by the search algorithm
        ground_truth_ids: Ground truth top-k IDs
        k: Value of k for recall@k
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if len(ground_truth_ids) == 0:
        return 0.0
    
    # Count how many ground truth items appear in retrieved results
    retrieved_set = set(retrieved_ids[:k])
    ground_truth_set = set(ground_truth_ids[:k])
    
    intersection = retrieved_set & ground_truth_set
    recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0.0
    
    return recall


def benchmark_hnsw(
    num_vectors: int = 1000,
    dim: int = 128,
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    num_queries: int = 100,
    seed: int = 42,
    output_dir: Path = Path("benchmarks/results"),
):
    """Benchmark HNSW operations with recall metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducible vector generation
    np.random.seed(seed)

    # Store all vectors for ground truth computation
    all_vectors = np.zeros((num_vectors, dim), dtype=np.float32)

    # Memory profiling for entire benchmark
    with memory_profiler() as mem_profiler:
        hnsw = HNSW(
            dim=dim, M=M, ef_construction=ef_construction, ef_search=ef_search, seed=seed
        )

        # Build index
        build_times = []
        for i in range(num_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            all_vectors[i] = vec
            with Timer() as t:
                hnsw.add(vec, i)
            build_times.append(t.elapsed * 1000)
            # Sample memory periodically during build
            if (i + 1) % (num_vectors // 10 + 1) == 0:
                mem_profiler.sample()
        
        build_peak_rss_mb = mem_profiler.get_peak_rss_mb()

        # Generate query vectors and compute ground truth
        query_vectors = []
        ground_truth_10 = []
        ground_truth_100 = []
        
        for _ in range(num_queries):
            query = np.random.randn(dim).astype(np.float32)
            query = query / np.linalg.norm(query)
            query_vectors.append(query)
            
            # Compute ground truth for recall@10 and recall@100
            gt_10 = compute_ground_truth(query, all_vectors, k=10)
            gt_100 = compute_ground_truth(query, all_vectors, k=100)
            ground_truth_10.append(gt_10)
            ground_truth_100.append(gt_100)

        # Search and compute recall
        search_times = []
        recall_10_scores = []
        recall_100_scores = []
        
        for i, query in enumerate(query_vectors):
            with Timer() as t:
                results = hnsw.search(query, k=100)  # Retrieve up to 100 for recall@100
            search_times.append(t.elapsed * 1000)
            
            # Extract retrieved IDs
            retrieved_ids = [node_id for node_id, _ in results]
            
            # Compute recall@10 and recall@100
            recall_10 = compute_recall(retrieved_ids, ground_truth_10[i], k=10)
            recall_100 = compute_recall(retrieved_ids, ground_truth_100[i], k=100)
            
            recall_10_scores.append(recall_10)
            recall_100_scores.append(recall_100)
        
        peak_rss_mb = mem_profiler.get_peak_rss_mb()
        memory_delta_mb = mem_profiler.get_memory_delta_mb()

    stats = hnsw.stats()
    
    # Compute mean recall
    mean_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
    mean_recall_100 = np.mean(recall_100_scores) if recall_100_scores else 0.0

    results = {
        "benchmark": "hnsw",
        "num_vectors": num_vectors,
        "dim": dim,
        "M": M,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "num_queries": num_queries,
        "seed": seed,
        "build_p50_ms": sorted(build_times)[len(build_times) // 2],
        "build_p95_ms": sorted(build_times)[int(len(build_times) * 0.95)],
        "build_p99_ms": sorted(build_times)[int(len(build_times) * 0.99)],
        "search_p50_ms": sorted(search_times)[len(search_times) // 2],
        "search_p95_ms": sorted(search_times)[int(len(search_times) * 0.95)],
        "search_p99_ms": sorted(search_times)[int(len(search_times) * 0.99)],
        "peak_rss_mb": peak_rss_mb,
        "build_peak_rss_mb": build_peak_rss_mb,
        "memory_delta_mb": memory_delta_mb,
        "mean_recall_10": float(mean_recall_10),
        "mean_recall_100": float(mean_recall_100),
        "recall_10_p50": float(np.percentile(recall_10_scores, 50)) if recall_10_scores else 0.0,
        "recall_10_p95": float(np.percentile(recall_10_scores, 95)) if recall_10_scores else 0.0,
        "recall_100_p50": float(np.percentile(recall_100_scores, 50)) if recall_100_scores else 0.0,
        "recall_100_p95": float(np.percentile(recall_100_scores, 95)) if recall_100_scores else 0.0,
        "hnsw_stats": stats,
    }

    output_file = output_dir / "hnsw_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"HNSW benchmark completed. Results saved to {output_file}")
    print(f"  Mean Recall@10: {mean_recall_10:.4f}")
    print(f"  Mean Recall@100: {mean_recall_100:.4f}")
    return results


if __name__ == "__main__":
    benchmark_hnsw()

