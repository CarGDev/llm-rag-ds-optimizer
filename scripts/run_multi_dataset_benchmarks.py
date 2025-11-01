"""Run benchmarks across multiple datasets for comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_dataset(
    source: str,
    corpus_name: str,
    output_dir: Path,
    limit: int | None = None,
    download: bool = True,
) -> Path | None:
    """Prepare a dataset: download, prepare embeddings, ready for benchmarking."""
    corpus_dir = output_dir / "raw" / corpus_name
    embeddings_dir = output_dir / "embeddings"
    corpus_file = None
    
    # Find existing corpus file (check multiple possible names)
    possible_files = ["corpus.jsonl", "reviews.jsonl", "business_reviews.jsonl", "pages.jsonl"]
    for filename in possible_files:
        if (corpus_dir / filename).exists():
            corpus_file = corpus_dir / filename
            break
    
    # Also check beir subdirectory for fiqa
    if corpus_file is None and corpus_name == "fiqa":
        beir_dir = output_dir / "raw" / "beir" / corpus_name
        if (beir_dir / "corpus.jsonl").exists():
            corpus_file = beir_dir / "corpus.jsonl"
    
    # Download if needed and not exists
    if download and corpus_file is None:
        print(f"\n📥 Downloading {corpus_name}...")
        try:
            if source.startswith("beir:"):
                cmd = [
                    sys.executable,
                    "scripts/download_corpus.py",
                    "--source", source,
                    "--output", str(corpus_dir),
                ]
            else:
                cmd = [
                    sys.executable,
                    "scripts/download_corpus.py",
                    "--source", source,
                    "--output", str(corpus_dir),
                ]
                if limit:
                    cmd.extend(["--limit", str(limit)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️  Download failed: {result.stderr}")
                return None
            
            # Find corpus file after download
            if (corpus_dir / "corpus.jsonl").exists():
                corpus_file = corpus_dir / "corpus.jsonl"
            elif corpus_name == "amazon23" and (corpus_dir / "reviews.jsonl").exists():
                corpus_file = corpus_dir / "reviews.jsonl"
        except Exception as e:
            print(f"⚠️  Error downloading {corpus_name}: {e}")
            return None
    
    if corpus_file is None or not corpus_file.exists():
        print(f"⚠️  Corpus file not found for {corpus_name}")
        return None
    
    # Check embeddings
    emb_file = embeddings_dir / f"{corpus_name}.npy"
    if not emb_file.exists():
        print(f"\n🔢 Preparing embeddings for {corpus_name}...")
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/prepare_embeddings.py",
            "--input", str(corpus_file),
            "--output", str(emb_file),
            "--dim", "384",
            "--seed", "42",
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Embedding preparation failed: {result.stderr}")
            return None
    
    return corpus_file


def run_benchmarks_for_dataset(
    corpus_name: str,
    corpus_file: Path,
    emb_file: Path,
    sizes: list[str],
    ef_values: list[int],
    M_values: list[int],
    num_queries: int = 50,  # Reduced for faster multi-dataset runs
    output_dir: Path = Path("benchmarks/results"),
) -> Path | None:
    """Run benchmarks for a single dataset."""
    print(f"\n🚀 Running benchmarks for {corpus_name}...")
    
    cmd = [
        sys.executable,
        "scripts/run_benchmarks.py",
        "--corpus", corpus_name,
        "--corpus-file", str(corpus_file),
        "--emb-file", str(emb_file),
        "--sizes", *sizes,
        "--ef", *[str(e) for e in ef_values],
        "--M", *[str(m) for m in M_values],
        "--num-queries", str(num_queries),
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  Benchmark failed for {corpus_name}: {result.stderr}")
        return None
    
    # Find the results directory
    results_dir = output_dir / corpus_name
    if results_dir.exists():
        timestamp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        if timestamp_dirs:
            return timestamp_dirs[-1] / "results.json"
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks across multiple datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fiqa", "amazon23", "msmarco", "yelp", "wikipedia"],
        help="Datasets to benchmark"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["10k", "25k", "50k"],
        help="Corpus sizes (e.g., 10k 25k 50k)"
    )
    parser.add_argument(
        "--ef",
        nargs="+",
        type=int,
        default=[50, 100],
        help="HNSW efSearch values"
    )
    parser.add_argument(
        "--M",
        nargs="+",
        type=int,
        default=[8, 16],
        help="HNSW M values"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Number of queries per benchmark"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading datasets (use existing)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit documents per dataset (for large datasets)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Dataset sources mapping
    dataset_sources = {
        "fiqa": "beir:fiqa",
        "amazon23": "amazon23",
        "msmarco": "msmarco",
        "yelp": "yelp",
        "wikipedia": "wikipedia",
    }
    
    data_dir = Path("data")
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("=" * 70)
    print("Multi-Dataset Benchmark Runner")
    print("=" * 70)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Sizes: {', '.join(args.sizes)}")
    print(f"efSearch: {', '.join(map(str, args.ef))}")
    print(f"M: {', '.join(map(str, args.M))}")
    print("=" * 70)
    
    for corpus_name in args.datasets:
        if corpus_name not in dataset_sources:
            print(f"⚠️  Unknown dataset: {corpus_name}, skipping")
            continue
        
        source = dataset_sources[corpus_name]
        limit = args.limit if corpus_name in ["amazon23", "wikipedia", "msmarco"] else None
        
        # Prepare dataset
        corpus_file = prepare_dataset(
            source=source,
            corpus_name=corpus_name,
            output_dir=data_dir,
            limit=limit,
            download=not args.skip_download,
        )
        
        if corpus_file is None:
            print(f"⚠️  Skipping {corpus_name} - preparation failed")
            continue
        
        # Check embeddings
        emb_file = embeddings_dir / f"{corpus_name}.npy"
        if not emb_file.exists():
            print(f"⚠️  Embeddings not found for {corpus_name}, skipping")
            continue
        
        # Run benchmarks
        results_file = run_benchmarks_for_dataset(
            corpus_name=corpus_name,
            corpus_file=corpus_file,
            emb_file=emb_file,
            sizes=args.sizes,
            ef_values=args.ef,
            M_values=args.M,
            num_queries=args.num_queries,
            output_dir=args.output_dir,
        )
        
        if results_file and results_file.exists():
            with open(results_file) as f:
                results[corpus_name] = json.load(f)
            print(f"✓ {corpus_name} benchmarks completed")
        else:
            print(f"⚠️  {corpus_name} benchmarks incomplete")
    
    # Save combined results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = args.output_dir / f"multi_dataset_{timestamp}.json"
        combined_file.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Combined results saved to {combined_file}")
    
    print("\n" + "=" * 70)
    print("Multi-dataset benchmarks completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

