"""Analyze variance in benchmark results and identify flaky benchmarks."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_benchmark_results(results_file: Path) -> list[dict]:
    """Load benchmark results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def identify_flaky_configurations(
    results: list[dict],
    cv_threshold: float = 20.0,
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Identify flaky benchmark configurations based on coefficient of variation.
    
    Args:
        results: List of aggregated result dictionaries
        cv_threshold: CV threshold (%) above which a benchmark is considered flaky
        metrics: List of metrics to check (default: critical metrics)
    
    Returns:
        List of flaky configuration summaries
    """
    if metrics is None:
        metrics = ["search_p50_ms", "search_p95_ms", "qps"]
    
    flaky_configs = []
    
    for result in results:
        flaky_metrics = []
        for metric in metrics:
            cv_key = f"{metric}_cv"
            if cv_key in result:
                cv = result[cv_key]
                if cv > cv_threshold:
                    mean_val = result.get(f"{metric}_mean", 0)
                    std_val = result.get(f"{metric}_std", 0)
                    flaky_metrics.append({
                        "metric": metric,
                        "mean": mean_val,
                        "std": std_val,
                        "cv": cv,
                    })
        
        if flaky_metrics:
            flaky_configs.append({
                "corpus": result.get("corpus"),
                "size": result.get("size"),
                "ef_search": result.get("ef_search"),
                "M": result.get("M"),
                "repetitions": result.get("repetitions"),
                "flaky_metrics": flaky_metrics,
            })
    
    return flaky_configs


def generate_variance_report(
    aggregated_file: Path,
    output_file: Path | None = None,
    cv_threshold: float = 20.0,
) -> dict[str, Any]:
    """
    Generate a variance analysis report.
    
    Args:
        aggregated_file: Path to aggregated results JSON
        output_file: Optional output file for report
        cv_threshold: CV threshold for flaky detection
    
    Returns:
        Report dictionary
    """
    results = load_benchmark_results(aggregated_file)
    
    if not results:
        return {"error": "No results found"}
    
    # Calculate overall statistics
    all_cvs = []
    for result in results:
        for key in result.keys():
            if key.endswith("_cv") and isinstance(result[key], (int, float)):
                all_cvs.append(result[key])
    
    # Identify flaky configurations
    flaky_configs = identify_flaky_configurations(results, cv_threshold)
    
    # Group by corpus
    by_corpus = {}
    for result in results:
        corpus = result.get("corpus", "unknown")
        if corpus not in by_corpus:
            by_corpus[corpus] = []
        by_corpus[corpus].append(result)
    
    report = {
        "summary": {
            "total_configurations": len(results),
            "flaky_configurations": len(flaky_configs),
            "flaky_percentage": (len(flaky_configs) / len(results) * 100) if results else 0,
            "average_cv": float(np.mean(all_cvs)) if all_cvs else 0.0,
            "max_cv": float(np.max(all_cvs)) if all_cvs else 0.0,
        },
        "flaky_configurations": flaky_configs,
        "by_corpus": {
            corpus: {
                "count": len(configs),
                "flaky_count": sum(1 for c in configs if any(m["cv"] > cv_threshold for m in identify_flaky_configurations([c], cv_threshold)[0].get("flaky_metrics", []))),
            }
            for corpus, configs in by_corpus.items()
        },
    }
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Variance report saved to {output_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze variance in benchmark results")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to aggregated results JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for variance report"
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=20.0,
        help="Coefficient of variation threshold (%) for flaky detection (default: 20.0)"
    )
    
    args = parser.parse_args()
    
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        return
    
    report = generate_variance_report(
        aggregated_file=args.results,
        output_file=args.output,
        cv_threshold=args.cv_threshold,
    )
    
    # Print summary
    print("\n" + "="*70)
    print("Variance Analysis Report")
    print("="*70)
    summary = report.get("summary", {})
    print(f"Total configurations: {summary.get('total_configurations', 0)}")
    print(f"Flaky configurations: {summary.get('flaky_configurations', 0)} ({summary.get('flaky_percentage', 0):.1f}%)")
    print(f"Average CV: {summary.get('average_cv', 0):.2f}%")
    print(f"Max CV: {summary.get('max_cv', 0):.2f}%")
    
    flaky = report.get("flaky_configurations", [])
    if flaky:
        print(f"\n⚠️  Flaky Configurations ({len(flaky)}):")
        for config in flaky[:10]:  # Show first 10
            print(f"  - {config.get('corpus')} (size={config.get('size')}, ef={config.get('ef_search')}, M={config.get('M')}):")
            for metric in config.get("flaky_metrics", []):
                print(f"    • {metric['metric']}: CV={metric['cv']:.1f}% (mean={metric['mean']:.2f}±{metric['std']:.2f})")
        if len(flaky) > 10:
            print(f"  ... and {len(flaky) - 10} more")
    else:
        print("\n✅ No flaky configurations detected!")
    
    print("="*70)


if __name__ == "__main__":
    main()

