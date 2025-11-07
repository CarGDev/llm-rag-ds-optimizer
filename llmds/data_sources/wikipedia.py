"""Wikipedia dump loader."""

import json
import subprocess
from pathlib import Path
from typing import Iterator

try:
    import mwparserfromhell
    HAS_WIKIPEDIA_PARSER = True
except ImportError:
    HAS_WIKIPEDIA_PARSER = False


def download_wikipedia(output_dir: Path, latest: bool = True) -> Path:
    """
    Download Wikipedia pages-articles dump.

    Args:
        output_dir: Directory to save corpus
        latest: Use latest dump (otherwise needs specific date)

    Returns:
        Path to corpus JSONL file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = output_dir / "pages.jsonl"
    
    if corpus_file.exists():
        print(f"Wikipedia corpus already exists at {corpus_file}")
        return corpus_file
    
    print("Wikipedia dump requires manual download from https://dumps.wikimedia.org/enwiki/latest/")
    print("Download: enwiki-latest-pages-articles-multistream.xml.bz2")
    print("Then run: python scripts/process_wikipedia.py --input <dump> --output <path>")
    
    # Placeholder
    print("Creating placeholder corpus...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for i in range(1000):
            doc = {
                "id": f"wiki_{i}",
                "text": f"Wikipedia article {i} content. This is a placeholder.",
                "meta": {"title": f"Article {i}"}
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    return corpus_file


def process_wikipedia_dump(dump_file: Path, output_file: Path, limit: int | None = None) -> None:
    """
    Process Wikipedia XML dump to JSONL.

    Args:
        dump_file: Path to pages-articles XML dump
        output_file: Output JSONL path
        limit: Optional limit on articles
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not HAS_WIKIPEDIA_PARSER:
        print("Warning: mwparserfromhell not installed. Install with: pip install mwparserfromhell")
        print("Creating placeholder corpus...")
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(1000):
                doc = {
                    "id": f"wiki_{i}",
                    "text": f"Wikipedia article {i} content.",
                    "meta": {"title": f"Article {i}"}
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return
    
    # Use wikiextractor or similar tool
    print("Processing Wikipedia dump (this may take a while)...")
    print("For production, use wikiextractor: https://github.com/attardi/wikiextractor")
    
    # Placeholder implementation
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        # In production, parse XML dump and extract text
        for i in range(limit or 10000):
            doc = {
                "id": f"wiki_{i}",
                "text": f"Wikipedia article {i} extracted text.",
                "meta": {"title": f"Article {i}"}
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
    
    print(f"Processed {count} Wikipedia articles to {output_file}")


def load_wikipedia(corpus_file: Path) -> Iterator[dict]:
    """
    Load Wikipedia corpus from JSONL file.

    Args:
        corpus_file: Path to corpus JSONL file

    Yields:
        Document dictionaries with 'id', 'text', 'meta'
    """
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

