"""End-to-end benchmark for retrieval pipeline."""

import json
import random
from pathlib import Path

import numpy as np

from llmds.retrieval_pipeline import RetrievalPipeline
from llmds.utils import Timer


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


def benchmark_end2end(
    num_docs: int = 500,
    num_queries: int = 50,
    output_dir: Path = Path("benchmarks/results"),
):
    """Benchmark end-to-end retrieval pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RetrievalPipeline(embedding_dim=384)

    # Build index
    build_times = []
    for i in range(num_docs):
        text, embedding = generate_document(i)
        with Timer() as t:
            pipeline.add_document(doc_id=i, text=text, embedding=embedding)
        build_times.append(t.elapsed * 1000)

    # Search
    search_times = []
    for _ in range(num_queries):
        query_text, query_embedding = generate_document(-1)
        with Timer() as t:
            pipeline.search(query_text, query_embedding=query_embedding, top_k=10)
        search_times.append(t.elapsed * 1000)

    stats = pipeline.stats()

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
        "pipeline_stats": stats,
    }

    output_file = output_dir / "end2end_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"End-to-end benchmark completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    benchmark_end2end()

