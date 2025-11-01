"""Benchmark script for inverted index with real or synthetic data."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from llmds.inverted_index import InvertedIndex
from llmds.tokenizer import Tokenizer
from llmds.utils import Timer, memory_profiler


def generate_document(doc_id: int, vocab_size: int = 100) -> str:
    """Generate a synthetic document."""
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "cat", "mouse", "elephant", "tiger", "lion", "bear", "wolf",
        "rabbit", "deer", "bird", "fish", "snake", "monkey", "panda",
        "computer", "science", "machine", "learning", "artificial", "intelligence",
        "neural", "network", "deep", "learning", "transformer", "attention",
        "language", "model", "natural", "processing", "text", "generation",
    ]
    words.extend([f"word{i}" for i in range(vocab_size)])
    doc_length = random.randint(10, 100)
    return " ".join(random.choices(words, k=doc_length))


def benchmark_inverted_index(
    corpus_file: Path | None = None,
    num_docs: int = 1000,
    num_queries: int = 100,
    output_dir: Path = Path("benchmarks/results"),
):
    """Benchmark inverted index operations with real or synthetic data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Memory profiling for entire benchmark
    with memory_profiler() as mem_profiler:
        index = InvertedIndex()
        
        # Load documents
        if corpus_file and corpus_file.exists():
            print(f"Loading corpus from {corpus_file}...")
            docs = []
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
            
            # Sample if needed
            if len(docs) > num_docs:
                random.seed(42)
                docs = random.sample(docs, num_docs)
            
            print(f"Using {len(docs)} documents from corpus")
        else:
            print(f"Generating {num_docs} synthetic documents...")
            docs = [{"id": i, "text": generate_document(i)} for i in range(num_docs)]
        
        mem_profiler.sample()  # Sample after loading docs
        
        # Build index
        build_times = []
        for i, doc in enumerate(docs):
            text = doc["text"] if isinstance(doc, dict) else doc
            doc_id = doc.get("id", i) if isinstance(doc, dict) else i
            
            with Timer() as t:
                index.add_document(doc_id=doc_id, text=text)
            build_times.append(t.elapsed * 1000)
            # Sample memory periodically during build
            if (i + 1) % (len(docs) // 10 + 1) == 0:
                mem_profiler.sample()
        
        build_peak_rss_mb = mem_profiler.get_peak_rss_mb()
        
        # Search
        search_times = []
        queries = []
        
        # Generate queries from documents or synthetic
        for _ in range(num_queries):
            if docs and isinstance(docs[0], dict):
                query_doc = random.choice(docs)
                query = query_doc["text"][:100]  # Use first 100 chars
            else:
                query = generate_document(-1, vocab_size=10)
            queries.append(query)
        
        for query in queries:
            with Timer() as t:
                index.search(query, top_k=10)
            search_times.append(t.elapsed * 1000)
        
        peak_rss_mb = mem_profiler.get_peak_rss_mb()
        memory_delta_mb = mem_profiler.get_memory_delta_mb()

    stats = index.stats()
    
    results = {
        "benchmark": "inverted_index",
        "corpus": corpus_file.name if corpus_file else "synthetic",
        "num_docs": len(docs),
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
        "index_stats": stats,
    }
    
    output_file = output_dir / "inverted_index_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Inverted index benchmark completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=1000)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--corpus", type=Path, help="Corpus JSONL file (optional)")
    args = parser.parse_args()
    
    benchmark_inverted_index(
        corpus_file=args.corpus,
        num_docs=args.num_docs,
        num_queries=args.num_queries,
    )
