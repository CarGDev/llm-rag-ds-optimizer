"""End-to-end benchmark for retrieval pipeline."""

import json
import random
from pathlib import Path

import numpy as np

from llmds.retrieval_pipeline import RetrievalPipeline
from llmds.utils import Timer, memory_profiler


def generate_document(doc_id: int) -> tuple[str, np.ndarray]:
    """Generate a synthetic document with embedding."""
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "cat",
        "mouse",
    ]
    doc_length = random.randint(10, 100)
    text = " ".join(random.choices(words, k=doc_length))

    # Generate random embedding
    embedding = np.random.randn(384).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    return text, embedding


def compute_ground_truth_pipeline(
    query_embedding: np.ndarray, doc_embeddings: dict[int, np.ndarray], k: int
) -> list[int]:
    """
    Compute ground truth nearest neighbors using exact distance computation.
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: Dictionary mapping doc_id to embedding
        k: Number of nearest neighbors to return
        
    Returns:
        List of doc_ids sorted by distance (ascending)
    """
    distances = []
    for doc_id, embedding in doc_embeddings.items():
        dist = np.linalg.norm(query_embedding - embedding)
        distances.append((dist, doc_id))
    
    # Sort by distance and return top-k doc_ids
    distances.sort()
    return [doc_id for _, doc_id in distances[:k]]


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
    
    retrieved_set = set(retrieved_ids[:k])
    ground_truth_set = set(ground_truth_ids[:k])
    
    intersection = retrieved_set & ground_truth_set
    recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0.0
    
    return recall


def benchmark_end2end(
    num_docs: int = 500,
    num_queries: int = 50,
    seed: int = 42,
    output_dir: Path = Path("benchmarks/results"),
):
    """Benchmark end-to-end retrieval pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Store all document embeddings for ground truth computation
    doc_embeddings = {}

    # Memory profiling for entire benchmark
    with memory_profiler() as mem_profiler:
        pipeline = RetrievalPipeline(embedding_dim=384, seed=seed)

        # Build index
        build_times = []
        for i in range(num_docs):
            text, embedding = generate_document(i)
            doc_embeddings[i] = embedding
            with Timer() as t:
                pipeline.add_document(doc_id=i, text=text, embedding=embedding)
            build_times.append(t.elapsed * 1000)
            # Sample memory periodically during build
            if (i + 1) % (num_docs // 10 + 1) == 0:
                mem_profiler.sample()
        
        build_peak_rss_mb = mem_profiler.get_peak_rss_mb()

        # Generate queries and compute ground truth
        queries = []
        ground_truth_10 = []
        ground_truth_100 = []
        
        for _ in range(num_queries):
            query_text, query_embedding = generate_document(-1)
            queries.append((query_text, query_embedding))
            
            # Compute ground truth for recall@10 and recall@100
            gt_10 = compute_ground_truth_pipeline(query_embedding, doc_embeddings, k=10)
            gt_100 = compute_ground_truth_pipeline(query_embedding, doc_embeddings, k=100)
            ground_truth_10.append(gt_10)
            ground_truth_100.append(gt_100)

        # Search and compute recall
        search_times = []
        recall_10_scores = []
        recall_100_scores = []
        
        for i, (query_text, query_embedding) in enumerate(queries):
            with Timer() as t:
                results = pipeline.search(
                    query_text, query_embedding=query_embedding, top_k=100
                )
            search_times.append(t.elapsed * 1000)
            
            # Extract retrieved IDs
            retrieved_ids = [doc_id for doc_id, _ in results]
            
            # Compute recall@10 and recall@100
            recall_10 = compute_recall(retrieved_ids, ground_truth_10[i], k=10)
            recall_100 = compute_recall(retrieved_ids, ground_truth_100[i], k=100)
            
            recall_10_scores.append(recall_10)
            recall_100_scores.append(recall_100)
        
        peak_rss_mb = mem_profiler.get_peak_rss_mb()
        memory_delta_mb = mem_profiler.get_memory_delta_mb()

    stats = pipeline.stats()
    
    # Compute mean recall
    mean_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
    mean_recall_100 = np.mean(recall_100_scores) if recall_100_scores else 0.0

    results = {
        "benchmark": "end2end",
        "num_docs": num_docs,
        "num_queries": num_queries,
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
        "pipeline_stats": stats,
    }

    output_file = output_dir / "end2end_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"End-to-end benchmark completed. Results saved to {output_file}")
    print(f"  Mean Recall@10: {mean_recall_10:.4f}")
    print(f"  Mean Recall@100: {mean_recall_100:.4f}")
    return results


if __name__ == "__main__":
    benchmark_end2end()

