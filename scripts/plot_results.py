"""Plot benchmark results and save to PNG, export to CSV."""

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(result_dir: Path = Path("benchmarks/results")) -> dict:
    """Load all benchmark results."""
    results = {}
    
    # Load old-style results (flat JSON files)
    for json_file in result_dir.glob("*.json"):
        if "benchmark" in json_file.stem:
            with open(json_file) as f:
                data = json.load(f)
                benchmark_name = data.get("benchmark", json_file.stem.replace("_benchmark", ""))
                results[benchmark_name] = data
    
    # Load new-style results (corpus/date/results.json)
    for corpus_dir in result_dir.iterdir():
        if corpus_dir.is_dir():
            for date_dir in corpus_dir.iterdir():
                if date_dir.is_dir():
                    results_file = date_dir / "results.json"
                    if results_file.exists():
                        with open(results_file) as f:
                            data_list = json.load(f)
                            if isinstance(data_list, list) and data_list:
                                # Use first result as representative or aggregate
                                corpus_name = corpus_dir.name
                                date_str = date_dir.name
                                key = f"{corpus_name}_{date_str}"
                                results[key] = data_list[0]  # Simplified
    
    return results


def export_to_csv(results: dict, output_file: Path = Path("benchmarks/results/benchmark_results.csv")):
    """Export benchmark results to CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for bench_name, data in results.items():
        # Extract key metrics
        row = {
            "benchmark": bench_name,
            "p50_ms": data.get("attach_p50_ms") or data.get("search_p50_ms") or data.get("batch_p50_ms") or data.get("build_p50_ms") or 0.0,
            "p95_ms": data.get("attach_p95_ms") or data.get("search_p95_ms") or data.get("batch_p95_ms") or data.get("build_p95_ms") or 0.0,
            "p99_ms": data.get("attach_p99_ms") or data.get("search_p99_ms") or data.get("batch_p99_ms") or data.get("build_p99_ms") or 0.0,
            "peak_rss_mb": data.get("peak_rss_mb", 0.0),
            "memory_delta_mb": data.get("memory_delta_mb", 0.0),
        }
        
        # Add specific metrics if available
        if "attach_p50_ms" in data:
            row.update({
                "attach_p50_ms": data.get("attach_p50_ms", 0),
                "attach_p95_ms": data.get("attach_p95_ms", 0),
                "attach_p99_ms": data.get("attach_p99_ms", 0),
                "get_p50_ms": data.get("get_p50_ms", 0),
                "get_p95_ms": data.get("get_p95_ms", 0),
                "get_p99_ms": data.get("get_p99_ms", 0),
            })
        if "search_p50_ms" in data:
            row.update({
                "search_p50_ms": data.get("search_p50_ms", 0),
                "search_p95_ms": data.get("search_p95_ms", 0),
                "search_p99_ms": data.get("search_p99_ms", 0),
            })
        
        # Add build peak RSS if available
        if "build_peak_rss_mb" in data:
            row["build_peak_rss_mb"] = data.get("build_peak_rss_mb", 0.0)
        
        rows.append(row)
    
    if rows:
        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)
        
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Results exported to CSV: {output_file}")


def plot_latency_distribution(results: dict, output_dir: Path = Path("benchmarks/figures")):
    """Plot latency distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = []
    p50_values = []
    p95_values = []
    p99_values = []

    for name, data in results.items():
        # Try different metric names
        p50 = data.get("search_p50_ms") or data.get("attach_p50_ms") or data.get("batch_p50_ms") or data.get("build_p50_ms", 0)
        p95 = data.get("search_p95_ms") or data.get("attach_p95_ms") or data.get("batch_p95_ms") or data.get("build_p95_ms", 0)
        p99 = data.get("search_p99_ms") or data.get("attach_p99_ms") or data.get("batch_p99_ms") or data.get("build_p99_ms", 0)
        
        if p50 > 0 or p95 > 0 or p99 > 0:
            benchmarks.append(name)
            p50_values.append(p50)
            p95_values.append(p95)
            p99_values.append(p99)

    if benchmarks:
        fig, ax = plt.subplots(figsize=(12, 7))
        x = range(len(benchmarks))
        width = 0.25

        ax.bar([i - width for i in x], p50_values, width, label="P50", alpha=0.8, color="#2ecc71")
        ax.bar(x, p95_values, width, label="P95", alpha=0.8, color="#3498db")
        ax.bar([i + width for i in x], p99_values, width, label="P99", alpha=0.8, color="#e74c3c")

        ax.set_xlabel("Benchmark", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
        ax.set_title("Latency Percentiles by Benchmark", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Add value labels on bars
        for i, (p50, p95, p99) in enumerate(zip(p50_values, p95_values, p99_values)):
            if p50 > 0:
                ax.text(i - width, p50, f"{p50:.2f}", ha="center", va="bottom", fontsize=8)
            if p95 > 0:
                ax.text(i, p95, f"{p95:.2f}", ha="center", va="bottom", fontsize=8)
            if p99 > 0:
                ax.text(i + width, p99, f"{p99:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        output_file = output_dir / "latency_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Latency plot saved to {output_file}")
        plt.close()


def plot_comparison_chart(results: dict, output_dir: Path = Path("benchmarks/figures")):
    """Plot comparison chart of all benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = []
    p95_latencies = []

    for name, data in results.items():
        p95 = data.get("search_p95_ms") or data.get("attach_p95_ms") or data.get("batch_p95_ms") or data.get("build_p95_ms", 0)
        if p95 > 0:
            benchmarks.append(name)
            p95_latencies.append(p95)

    if benchmarks:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(range(len(benchmarks)))
        bars = ax.barh(benchmarks, p95_latencies, color=colors, alpha=0.8)
        
        ax.set_xlabel("P95 Latency (ms)", fontsize=12, fontweight="bold")
        ax.set_title("Benchmark Performance Comparison (P95 Latency)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
        
        # Add value labels
        for bar, latency in zip(bars, p95_latencies):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f"{latency:.2f}ms",
                   ha="left", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        output_file = output_dir / "benchmark_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to {output_file}")
        plt.close()


def plot_memory_usage(results: dict, output_dir: Path = Path("benchmarks/figures")):
    """Plot memory usage (peak RSS) by benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = []
    peak_rss_values = []
    memory_delta_values = []

    for name, data in results.items():
        peak_rss = data.get("peak_rss_mb", 0.0)
        memory_delta = data.get("memory_delta_mb", 0.0)
        if peak_rss > 0:
            benchmarks.append(name)
            peak_rss_values.append(peak_rss)
            memory_delta_values.append(memory_delta)

    if benchmarks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Peak RSS
        colors1 = plt.cm.plasma(range(len(benchmarks)))
        bars1 = ax1.barh(benchmarks, peak_rss_values, color=colors1, alpha=0.8)
        ax1.set_xlabel("Peak RSS (MB)", fontsize=12, fontweight="bold")
        ax1.set_title("Peak Memory Usage by Benchmark", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, linestyle="--", axis="x")
        
        # Add value labels
        for bar, rss in zip(bars1, peak_rss_values):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, f"{rss:.2f}MB",
                   ha="left", va="center", fontsize=9, fontweight="bold")
        
        # Plot 2: Memory Delta
        colors2 = plt.cm.coolwarm(range(len(benchmarks)))
        bars2 = ax2.barh(benchmarks, memory_delta_values, color=colors2, alpha=0.8)
        ax2.set_xlabel("Memory Delta (MB)", fontsize=12, fontweight="bold")
        ax2.set_title("Memory Allocation Delta by Benchmark", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, linestyle="--", axis="x")
        
        # Add value labels
        for bar, delta in zip(bars2, memory_delta_values):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, f"{delta:.2f}MB",
                   ha="left", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        output_file = output_dir / "memory_usage.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Memory usage plot saved to {output_file}")
        plt.close()


if __name__ == "__main__":
    results = load_results()
    if results:
        export_to_csv(results)
        plot_latency_distribution(results)
        plot_comparison_chart(results)
        plot_memory_usage(results)
        print(f"\nProcessed {len(results)} benchmark results")
    else:
        print("No benchmark results found. Run benchmarks first.")
