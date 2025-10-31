"""Run end-to-end benchmarks on real corpora."""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from llmds.data_sources.beir_loader import load_beir
from llmds.data_sources.amazon_reviews import load_amazon_reviews
from llmds.retrieval_pipeline import RetrievalPipeline
from llmds.utils import Timer


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
    
    # Build pipeline
    print("Building pipeline...")
    pipeline = RetrievalPipeline(
        embedding_dim=embedding_dim,
        hnsw_M=M,
        hnsw_ef_search=ef_search,
        hnsw_ef_construction=ef_search * 4,
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
    
    # Run queries
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
    
    for i, (query_text, query_emb) in enumerate(zip(query_texts, query_embeddings)):
        with Timer() as t:
            pipeline.search(query_text, query_embedding=query_emb, top_k=10)
        search_times.append(t.elapsed * 1000)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_queries} queries...")
    
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
    }
    
    print(f"✓ Results: P50={results['search_p50_ms']:.2f}ms, P95={results['search_p95_ms']:.2f}ms, QPS={results['qps']:.2f}")
    
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
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions")
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
    
    # Run benchmarks
    for size in sizes:
        for ef in args.ef:
            for M in args.M:
                for rep in range(args.repetitions):
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
                    all_results.append(result)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save CSV
    csv_file = output_dir / "results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"\n✓ All results saved to {output_dir}")


if __name__ == "__main__":
    main()
