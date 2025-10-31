"""Generate detailed plots for corpus-based benchmarks."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_corpus_results(results_dir: Path) -> list[dict]:
    """Load all corpus benchmark results."""
    results = []
    
    for corpus_dir in results_dir.iterdir():
        if not corpus_dir.is_dir():
            continue
        
        for date_dir in corpus_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            results_file = date_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results.extend(data)
    
    return results


def plot_latency_by_corpus_size(results: list[dict], output_dir: Path):
    """Plot latency vs corpus size."""
    # Group by corpus size
    by_size = {}
    for r in results:
        size = r["size"]
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r)
    
    sizes = sorted(by_size.keys())
    p50s = [np.mean([r["search_p50_ms"] for r in by_size[s]]) for s in sizes]
    p95s = [np.mean([r["search_p95_ms"] for r in by_size[s]]) for s in sizes]
    p99s = [np.mean([r["search_p99_ms"] for r in by_size[s]]) for s in sizes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.25
    
    ax.bar(x - width, p50s, width, label="P50", alpha=0.8)
    ax.bar(x, p95s, width, label="P95", alpha=0.8)
    ax.bar(x + width, p99s, width, label="P99", alpha=0.8)
    
    ax.set_xlabel("Corpus Size (documents)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Search Latency vs Corpus Size (FIQA Dataset)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s//1000}k" for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "corpus_size_latency.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_qps_vs_size(results: list[dict], output_dir: Path):
    """Plot QPS vs corpus size."""
    by_size = {}
    for r in results:
        size = r["size"]
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r)
    
    sizes = sorted(by_size.keys())
    qps = [np.mean([r["qps"] for r in by_size[s]]) for s in sizes]
    qps_std = [np.std([r["qps"] for r in by_size[s]]) for s in sizes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar([s/1000 for s in sizes], qps, yerr=qps_std, marker="o", 
                linestyle="-", linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Corpus Size (thousands of documents)")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_title("Throughput vs Corpus Size (FIQA Dataset)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "corpus_size_qps.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_scaling_analysis(results: list[dict], output_dir: Path):
    """Plot scaling analysis with multiple metrics."""
    by_size = {}
    for r in results:
        size = r["size"]
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r)
    
    sizes = sorted(by_size.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Latency
    p50s = [np.mean([r["search_p50_ms"] for r in by_size[s]]) for s in sizes]
    p95s = [np.mean([r["search_p95_ms"] for r in by_size[s]]) for s in sizes]
    
    ax1.plot([s/1000 for s in sizes], p50s, "o-", label="P50", linewidth=2, markersize=8)
    ax1.plot([s/1000 for s in sizes], p95s, "s-", label="P95", linewidth=2, markersize=8)
    ax1.set_xlabel("Corpus Size (thousands)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: QPS
    qps = [np.mean([r["qps"] for r in by_size[s]]) for s in sizes]
    ax2.plot([s/1000 for s in sizes], qps, "o-", color="green", linewidth=2, markersize=8)
    ax2.set_xlabel("Corpus Size (thousands)")
    ax2.set_ylabel("Queries Per Second")
    ax2.set_title("Throughput Scaling")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "scaling_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def main():
    results_dir = Path("benchmarks/results")
    output_dir = Path("benchmarks/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = load_corpus_results(results_dir)
    
    if not results:
        print("No corpus benchmark results found")
        return
    
    print(f"Loaded {len(results)} benchmark runs")
    
    # Generate plots
    plot_latency_by_corpus_size(results, output_dir)
    plot_qps_vs_size(results, output_dir)
    plot_scaling_analysis(results, output_dir)
    
    print(f"\n✓ Generated corpus analysis plots in {output_dir}")


if __name__ == "__main__":
    main()

