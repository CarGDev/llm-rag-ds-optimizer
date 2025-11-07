"""Download and prepare datasets."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmds.data_sources.msmarco import download_msmarco
from llmds.data_sources.beir_loader import download_beir
from llmds.data_sources.amazon_reviews import download_amazon_reviews
from llmds.data_sources.yelp import download_yelp
from llmds.data_sources.wikipedia import download_wikipedia
from llmds.data_sources.commoncrawl import download_commoncrawl


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--source",
        required=True,
        help="Dataset source: msmarco, beir:task (e.g., beir:fiqa), amazon23, yelp, wikipedia, commoncrawl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for corpus"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents"
    )
    parser.add_argument(
        "--cc-month",
        type=str,
        help="Common Crawl month (e.g., 'CC-MAIN-2025-14')"
    )
    
    args = parser.parse_args()
    
    # Parse source (handle beir:task format)
    source_parts = args.source.split(":", 1)
    source_base = source_parts[0]
    task = source_parts[1] if len(source_parts) > 1 else None
    
    if source_base == "msmarco":
        download_msmarco(args.output)
    elif source_base == "beir":
        if not task:
            print("Error: BEIR requires task name (e.g., 'beir:fiqa', 'beir:scidocs')")
            sys.exit(1)
        download_beir(task, args.output)
    elif source_base == "amazon23":
        download_amazon_reviews(args.output, limit=args.limit)
    elif source_base == "yelp":
        download_yelp(args.output)
    elif source_base == "wikipedia":
        download_wikipedia(args.output)
    elif source_base == "commoncrawl":
        download_commoncrawl(args.output, cc_month=args.cc_month, limit=args.limit)
    else:
        print(f"Error: Unknown source '{source_base}'. Use: msmarco, beir:task, amazon23, yelp, wikipedia, commoncrawl")
        sys.exit(1)
    
    print(f"✓ Dataset downloaded to {args.output}")


if __name__ == "__main__":
    main()

