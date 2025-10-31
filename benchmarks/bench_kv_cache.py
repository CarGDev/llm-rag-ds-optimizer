"""Benchmark script for KV cache."""

import json
import time
from pathlib import Path

from llmds.kv_cache import KVCache
from llmds.utils import MetricsCollector, Timer


def benchmark_kv_cache(
    num_sequences: int = 1000,
    tokens_per_seq: int = 1000,
    page_size: int = 512,
    output_dir: Path = Path("benchmarks/results"),
):
    """Benchmark KV cache operations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = KVCache(page_size=page_size, max_pages=10000)
    collector = MetricsCollector()

    # Benchmark attach
    attach_times = []
    for i in range(num_sequences):
        kv_tokens = list(range(tokens_per_seq))
        with Timer() as t:
            cache.attach(seq_id=i, kv_tokens=kv_tokens)
        attach_times.append(t.elapsed * 1000)  # Convert to ms

    collector.record_latency(sum(attach_times) / len(attach_times))

    # Benchmark get
    get_times = []
    for i in range(num_sequences):
        with Timer() as t:
            cache.get(seq_id=i)
        get_times.append(t.elapsed * 1000)

    collector.record_latency(sum(get_times) / len(get_times))

    # Benchmark detach
    detach_times = []
    for i in range(num_sequences):
        with Timer() as t:
            cache.detach(seq_id=i)
        detach_times.append(t.elapsed * 1000)

    collector.record_latency(sum(detach_times) / len(detach_times))

    # Get statistics
    stats = cache.stats()
    metrics = collector.get_metrics()

    results = {
        "benchmark": "kv_cache",
        "num_sequences": num_sequences,
        "tokens_per_seq": tokens_per_seq,
        "page_size": page_size,
        "attach_p50_ms": attach_times[len(attach_times) // 2],
        "attach_p95_ms": attach_times[int(len(attach_times) * 0.95)],
        "attach_p99_ms": attach_times[int(len(attach_times) * 0.99)],
        "get_p50_ms": get_times[len(get_times) // 2],
        "get_p95_ms": get_times[int(len(get_times) * 0.95)],
        "get_p99_ms": get_times[int(len(get_times) * 0.99)],
        "detach_p50_ms": detach_times[len(detach_times) // 2],
        "detach_p95_ms": detach_times[int(len(detach_times) * 0.95)],
        "detach_p99_ms": detach_times[int(len(detach_times) * 0.99)],
        "cache_stats": stats,
    }

    # Save results
    output_file = output_dir / "kv_cache_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"KV cache benchmark completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--tokens_per_seq", type=int, default=1000)
    parser.add_argument("--page_size", type=int, default=512)
    args = parser.parse_args()

    benchmark_kv_cache(
        num_sequences=args.num_sequences,
        tokens_per_seq=args.tokens_per_seq,
        page_size=args.page_size,
    )

