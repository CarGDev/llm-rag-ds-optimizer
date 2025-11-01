"""Run end-to-end benchmarks on real corpora with variance analysis."""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Fallback for confidence intervals without scipy
    import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from llmds.data_sources.beir_loader import load_beir
from llmds.data_sources.amazon_reviews import load_amazon_reviews
from llmds.retrieval_pipeline import RetrievalPipeline
from llmds.utils import Timer, memory_profiler


def calculate_statistics(values: list[float], confidence_level: float = 0.95) -> dict[str, Any]:
    """
    Calculate statistical summary for a list of values.
    
    Args:
        values: List of numeric values
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Dictionary with mean, std, min, max, percentiles, and confidence intervals
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "cv": 0.0,  # Coefficient of variation
        }
    
    values_array = np.array(values)
    mean = float(np.mean(values_array))
    std = float(np.std(values_array, ddof=1))  # Sample std dev (ddof=1)
    min_val = float(np.min(values_array))
    max_val = float(np.max(values_array))
    
    # Percentiles
    p50 = float(np.percentile(values_array, 50))
    p95 = float(np.percentile(values_array, 95))
    p99 = float(np.percentile(values_array, 99))
    
    # Confidence interval (t-distribution for small samples)
    n = len(values)
    if n > 1:
        alpha = 1 - confidence_level
        if HAS_SCIPY:
            # Use t-distribution for small samples
            t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_critical * (std / np.sqrt(n))
        else:
            # Fallback: use normal distribution approximation (z-score)
            # For 95% CI: z = 1.96, for 90% CI: z = 1.645
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_critical = z_scores.get(confidence_level, 1.96)
            margin = z_critical * (std / np.sqrt(n))
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = mean
        ci_upper = mean
    
    # Coefficient of variation (relative standard deviation)
    cv = (std / mean * 100) if mean > 0 else 0.0
    
    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cv": cv,  # Coefficient of variation (%)
        "count": n,
    }


def aggregate_repetitions(results: list[dict]) -> dict[str, Any]:
    """
    Aggregate results across repetitions with variance analysis.
    
    Args:
        results: List of result dictionaries from multiple repetitions
    
    Returns:
        Dictionary with aggregated statistics including variance metrics
    """
    if not results:
        return {}
    
    # Extract metric names (all numeric keys except metadata)
    metadata_keys = {"corpus", "size", "ef_search", "M", "num_queries", "repetition"}
    metric_keys = [k for k in results[0].keys() if k not in metadata_keys]
    
    aggregated = {
        "corpus": results[0].get("corpus"),
        "size": results[0].get("size"),
        "ef_search": results[0].get("ef_search"),
        "M": results[0].get("M"),
        "num_queries": results[0].get("num_queries"),
        "repetitions": len(results),
    }
    
    # Calculate statistics for each metric
    for metric in metric_keys:
        values = [r.get(metric, 0.0) for r in results if metric in r]
        if values:
            stats_dict = calculate_statistics(values)
            # Store both mean/std and full statistics
            aggregated[f"{metric}_mean"] = stats_dict["mean"]
            aggregated[f"{metric}_std"] = stats_dict["std"]
            aggregated[f"{metric}_min"] = stats_dict["min"]
            aggregated[f"{metric}_max"] = stats_dict["max"]
            aggregated[f"{metric}_ci_lower"] = stats_dict["ci_lower"]
            aggregated[f"{metric}_ci_upper"] = stats_dict["ci_upper"]
            aggregated[f"{metric}_cv"] = stats_dict["cv"]  # Coefficient of variation
    
    # Identify flaky benchmarks (high variance)
    # Mark as flaky if CV > 20% for critical metrics
    critical_metrics = ["search_p50_ms", "search_p95_ms", "qps"]
    flaky_metrics = []
    for metric in critical_metrics:
        cv_key = f"{metric}_cv"
        if cv_key in aggregated and aggregated[cv_key] > 20.0:
            flaky_metrics.append(metric)
    
    aggregated["flaky_metrics"] = flaky_metrics
    aggregated["is_flaky"] = len(flaky_metrics) > 0
    
    return aggregated


def load_corpus_sample(corpus_file: Path, size: int, seed: int = 42) -> list[dict]:
    """Load a sample of documents from corpus."""
    random.seed(seed)
    np.random.seed(seed)
    
    all_docs = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_docs.append(json.loads(line))
    
    if len(all_docs) <= size:
        return all_docs
    
    # Sample without replacement
    return random.sample(all_docs, size)


def run_benchmark(
    corpus_file: Path,
    emb_file: Path | None,
    corpus_name: str,
    size: int,
    ef_search: int,
    M: int,
    num_queries: int = 100,
    embedding_dim: int = 384,
) -> dict:
    """
    Run benchmark on a corpus sample.

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n=== Benchmarking {corpus_name} (size={size}, ef={ef_search}, M={M}) ===")
    
    # Load corpus sample
    print(f"Loading corpus sample...")
    docs = load_corpus_sample(corpus_file, size)
    print(f"Loaded {len(docs)} documents")
    
    # Load or generate embeddings
    if emb_file and emb_file.exists():
        embeddings = np.load(emb_file)
        # Trim to sample size
        embeddings = embeddings[:len(docs)]
    else:
        print("Generating deterministic embeddings...")
        rng = np.random.RandomState(42)
        embeddings = []
        for i in range(len(docs)):
            emb = rng.randn(embedding_dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        embeddings = np.stack(embeddings)
    
    # Build pipeline with deterministic seed
    print("Building pipeline...")
    
    # Memory profiling for build phase
    with memory_profiler() as mem_profiler:
        pipeline = RetrievalPipeline(
            embedding_dim=embedding_dim,
            hnsw_M=M,
            hnsw_ef_search=ef_search,
            hnsw_ef_construction=ef_search * 4,
            seed=42,  # Fixed seed for reproducible HNSW structure
        )
        
        # Add documents
        build_times = []
        for i, doc in enumerate(docs):
            with Timer() as t:
                pipeline.add_document(
                    doc_id=i,
                    text=doc["text"],
                    embedding=embeddings[i],
                )
            build_times.append(t.elapsed * 1000)
            # Sample memory periodically during build
            if (i + 1) % (len(docs) // 10 + 1) == 0:
                mem_profiler.sample()
        
        build_peak_rss_mb = mem_profiler.get_peak_rss_mb()
        build_memory_delta_mb = mem_profiler.get_memory_delta_mb()
    
    # Run queries with memory profiling
    print(f"Running {num_queries} queries...")
    search_times = []
    rng = np.random.RandomState(42)
    
    # Generate query embeddings
    query_embeddings = []
    for _ in range(num_queries):
        qemb = rng.randn(embedding_dim).astype(np.float32)
        qemb = qemb / np.linalg.norm(qemb)
        query_embeddings.append(qemb)
    
    # Use document texts as queries (simplified)
    query_texts = [docs[i % len(docs)]["text"][:100] for i in range(num_queries)]
    
    # Memory profiling for search phase
    with memory_profiler() as search_mem_profiler:
        for i, (query_text, query_emb) in enumerate(zip(query_texts, query_embeddings)):
            with Timer() as t:
                pipeline.search(query_text, query_embedding=query_emb, top_k=10)
            search_times.append(t.elapsed * 1000)
            
            # Sample memory periodically during search
            if (i + 1) % 20 == 0:
                search_mem_profiler.sample()
                print(f"Completed {i + 1}/{num_queries} queries...")
        
        search_peak_rss_mb = search_mem_profiler.get_peak_rss_mb()
    
    # Overall peak RSS (maximum of build and search phases)
    overall_peak_rss_mb = max(build_peak_rss_mb, search_peak_rss_mb)
    
    # Compute statistics
    build_times_sorted = sorted(build_times)
    search_times_sorted = sorted(search_times)
    
    results = {
        "corpus": corpus_name,
        "size": size,
        "ef_search": ef_search,
        "M": M,
        "num_queries": num_queries,
        "build_p50_ms": build_times_sorted[len(build_times_sorted) // 2],
        "build_p95_ms": build_times_sorted[int(len(build_times_sorted) * 0.95)],
        "build_p99_ms": build_times_sorted[int(len(build_times_sorted) * 0.99)],
        "search_p50_ms": search_times_sorted[len(search_times_sorted) // 2],
        "search_p95_ms": search_times_sorted[int(len(search_times_sorted) * 0.95)],
        "search_p99_ms": search_times_sorted[int(len(search_times_sorted) * 0.99)],
        "avg_build_time_ms": sum(build_times) / len(build_times),
        "avg_search_time_ms": sum(search_times) / len(search_times),
        "qps": 1000.0 / (sum(search_times) / len(search_times)) if search_times else 0.0,
        # Memory metrics
        "peak_rss_mb": overall_peak_rss_mb,
        "build_peak_rss_mb": build_peak_rss_mb,
        "build_memory_delta_mb": build_memory_delta_mb,
        "search_peak_rss_mb": search_peak_rss_mb,
    }
    
    print(f"✓ Results: P50={results['search_p50_ms']:.2f}ms, P95={results['search_p95_ms']:.2f}ms, QPS={results['qps']:.2f}, Peak RSS={results['peak_rss_mb']:.2f}MB")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on real corpora")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus name")
    parser.add_argument("--corpus-file", type=Path, required=True, help="Corpus JSONL file")
    parser.add_argument("--emb-file", type=Path, help="Embeddings .npy file")
    parser.add_argument("--sizes", nargs="+", type=str, default=["10k"], help="Corpus sizes (e.g., 10k 50k 100k)")
    parser.add_argument("--ef", nargs="+", type=int, default=[50], help="HNSW efSearch values")
    parser.add_argument("--M", nargs="+", type=int, default=[16], help="HNSW M values")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of repetitions for variance analysis (default: 5)")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"), help="Output directory")
    
    args = parser.parse_args()
    
    # Parse sizes
    def parse_size(s: str) -> int:
        s = s.lower()
        if s.endswith("k"):
            return int(s[:-1]) * 1000
        elif s.endswith("m"):
            return int(s[:-1]) * 1000000
        return int(s)
    
    sizes = [parse_size(s) for s in args.sizes]
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / args.corpus / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    aggregated_results = []
    
    print(f"\n{'='*70}")
    print(f"Running benchmarks with {args.repetitions} repetitions per configuration")
    print(f"{'='*70}\n")
    
    # Run benchmarks
    for size in sizes:
        for ef in args.ef:
            for M in args.M:
                config_key = f"{size}_{ef}_{M}"
                print(f"Configuration: size={size}, ef={ef}, M={M}")
                
                repetition_results = []
                for rep in range(args.repetitions):
                    print(f"  Repetition {rep + 1}/{args.repetitions}...", end=" ", flush=True)
                    result = run_benchmark(
                        corpus_file=args.corpus_file,
                        emb_file=args.emb_file,
                        corpus_name=args.corpus,
                        size=size,
                        ef_search=ef,
                        M=M,
                        num_queries=args.num_queries,
                    )
                    result["repetition"] = rep
                    repetition_results.append(result)
                    all_results.append(result)
                    print("✓")
                
                # Aggregate across repetitions
                aggregated = aggregate_repetitions(repetition_results)
                if aggregated:
                    # Keep original metrics for backward compatibility
                    for metric in ["search_p50_ms", "search_p95_ms", "search_p99_ms", "qps"]:
                        if f"{metric}_mean" in aggregated:
                            aggregated[metric] = aggregated[f"{metric}_mean"]
                    
                    aggregated_results.append(aggregated)
                    
                    # Print variance summary
                    print(f"\n  Variance Summary:")
                    print(f"    Search P50: {aggregated.get('search_p50_ms_mean', 0):.2f} ± {aggregated.get('search_p50_ms_std', 0):.2f} ms (CV: {aggregated.get('search_p50_ms_cv', 0):.1f}%)")
                    print(f"    Search P95: {aggregated.get('search_p95_ms_mean', 0):.2f} ± {aggregated.get('search_p95_ms_std', 0):.2f} ms (CV: {aggregated.get('search_p95_ms_cv', 0):.1f}%)")
                    print(f"    QPS: {aggregated.get('qps_mean', 0):.2f} ± {aggregated.get('qps_std', 0):.2f} (CV: {aggregated.get('qps_cv', 0):.1f}%)")
                    
                    if aggregated.get("is_flaky", False):
                        print(f"    ⚠️  FLAKY: High variance detected in {', '.join(aggregated.get('flaky_metrics', []))}")
                    print()
    
    # Save detailed results (all repetitions)
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregated results with variance statistics
    aggregated_file = output_dir / "results_aggregated.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Save CSV with all repetitions
    csv_file = output_dir / "results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    # Save aggregated CSV
    aggregated_csv_file = output_dir / "results_aggregated.csv"
    if aggregated_results:
        agg_fieldnames = list(aggregated_results[0].keys())
        with open(aggregated_csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
            writer.writeheader()
            writer.writerows(aggregated_results)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Benchmark Summary")
    print(f"{'='*70}")
    print(f"Total configurations: {len(aggregated_results)}")
    print(f"Total repetitions: {len(all_results)}")
    flaky_count = sum(1 for r in aggregated_results if r.get("is_flaky", False))
    if flaky_count > 0:
        print(f"⚠️  Flaky configurations: {flaky_count}")
    print(f"\nResults saved to:")
    print(f"  - Detailed: {results_file}")
    print(f"  - Aggregated: {aggregated_file}")
    print(f"  - CSV: {csv_file}")
    print(f"  - Aggregated CSV: {aggregated_csv_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
