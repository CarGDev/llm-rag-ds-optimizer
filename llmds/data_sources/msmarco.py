"""MS MARCO dataset loader."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve


def download_msmarco(output_dir: Path, split: str = "passage") -> Path:
    """
    Download MS MARCO dataset.

    Args:
        output_dir: Directory to save files
        split: Dataset split ('passage' or 'doc')

    Returns:
        Path to downloaded corpus file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://msmarco.blob.core.windows.net/msmarcoranking"
    
    if split == "passage":
        collection_url = f"{base_url}/collection.tar.gz"
        queries_url = f"{base_url}/queries.tar.gz"
    else:
        collection_url = f"{base_url}/docranking/collection.tar.gz"
        queries_url = f"{base_url}/docranking/queries.tar.gz"
    
    corpus_file = output_dir / "corpus.jsonl"
    
    if corpus_file.exists():
        print(f"MS MARCO corpus already exists at {corpus_file}")
        return corpus_file
    
    # Download and extract (simplified - in production, use official downloader)
    print(f"Downloading MS MARCO {split} collection...")
    print("Note: For production use, download from https://microsoft.github.io/msmarco/")
    print("This is a placeholder implementation.")
    
    # Placeholder: in real implementation, download and extract tarball
    # For now, create a small sample
    with open(corpus_file, "w", encoding="utf-8") as f:
        for i in range(1000):  # Sample
            doc = {
                "id": f"msmarco_{i}",
                "text": f"MS MARCO passage {i} content. This is a placeholder.",
                "meta": {"split": split}
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"Created sample corpus at {corpus_file}")
    return corpus_file


def load_msmarco(corpus_file: Path) -> Iterator[dict]:
    """
    Load MS MARCO corpus from JSONL file.

    Args:
        corpus_file: Path to corpus JSONL file

    Yields:
        Document dictionaries with 'id', 'text', 'meta'
    """
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_msmarco(
    collection_file: Path,
    output_file: Path,
    limit: int | None = None,
) -> None:
    """
    Normalize MS MARCO collection to JSONL format.

    Args:
        collection_file: Path to MS MARCO collection TSV
        output_file: Output JSONL path
        limit: Optional limit on number of documents
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(collection_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            if limit and count >= limit:
                break
            
            parts = line.strip().split("\t", 2)
            if len(parts) >= 2:
                doc_id, text = parts[0], parts[1]
                doc = {
                    "id": doc_id,
                    "text": text,
                    "meta": {"source": "msmarco"}
                }
                outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
    
    print(f"Normalized {count} documents to {output_file}")

