"""Benchmark script for scheduler."""

import json
import time
from pathlib import Path

from llmds.scheduler import Scheduler
from llmds.utils import MetricsCollector, Timer


def benchmark_scheduler(
    num_requests: int = 1000,
    max_batch_size: int = 32,
    max_wait_ms: float = 50.0,
    output_dir: Path = Path("benchmarks/results"),
):
    """Benchmark scheduler operations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler = Scheduler(max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
    collector = MetricsCollector()

    # Submit requests
    submit_times = []
    for i in range(num_requests):
        tokens = (i % 100) + 10  # Vary token counts
        with Timer() as t:
            scheduler.submit(tokens=tokens)
        submit_times.append(t.elapsed * 1000)

    collector.record_latency(sum(submit_times) / len(submit_times))

    # Get batches
    batch_times = []
    batch_sizes = []
    total_batches = 0

    while True:
        with Timer() as t:
            batch = scheduler.get_batch(force=False)
        if batch is None:
            time.sleep(0.01)
            batch = scheduler.get_batch(force=True)
        if batch is None:
            break

        batch_times.append(t.elapsed * 1000)
        batch_sizes.append(len(batch))
        scheduler.complete_batch(batch)
        total_batches += 1

        if total_batches * max_batch_size >= num_requests:
            break

    if batch_times:
        collector.record_latency(sum(batch_times) / len(batch_times))

    stats = scheduler.stats()
    metrics = collector.get_metrics()

    results = {
        "benchmark": "scheduler",
        "num_requests": num_requests,
        "max_batch_size": max_batch_size,
        "max_wait_ms": max_wait_ms,
        "submit_p50_ms": submit_times[len(submit_times) // 2] if submit_times else 0,
        "batch_p50_ms": batch_times[len(batch_times) // 2] if batch_times else 0,
        "avg_batch_size": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0,
        "total_batches": total_batches,
        "scheduler_stats": stats,
    }

    output_file = output_dir / "scheduler_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Scheduler benchmark completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    benchmark_scheduler()

