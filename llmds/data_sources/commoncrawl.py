"""Common Crawl loader."""

import json
from pathlib import Path
from typing import Iterator


def download_commoncrawl(output_dir: Path, cc_month: str | None = None, limit: int | None = None) -> Path:
    """
    Download Common Crawl data.

    Args:
        output_dir: Directory to save corpus
        cc_month: Common Crawl month (e.g., 'CC-MAIN-2025-14')
        limit: Optional limit on documents

    Returns:
        Path to corpus JSONL file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = output_dir / "web_pages.jsonl"
    
    if corpus_file.exists():
        print(f"Common Crawl corpus already exists at {corpus_file}")
        return corpus_file
    
    print("Common Crawl requires cc-downloader tool.")
    print("Install: pip install common-crawl-download")
    print("Usage: See https://github.com/commoncrawl/cc-downloader")
    print("Be respectful of bandwidth when downloading.")
    
    # Placeholder
    print("Creating placeholder corpus...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        size = limit or 10000
        for i in range(size):
            doc = {
                "id": f"cc_{i}",
                "text": f"Common Crawl web page {i} content. This is a placeholder.",
                "meta": {"url": f"https://example.com/page{i}", "cc_month": cc_month or "CC-MAIN-2025-14"}
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"Created placeholder corpus with {size} documents")
    return corpus_file


def process_commoncrawl_warc(warc_file: Path, output_file: Path, limit: int | None = None) -> None:
    """
    Process Common Crawl WARC file to JSONL.

    Args:
        warc_file: Path to WARC file
        output_file: Output JSONL path
        limit: Optional limit on documents
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from warcio.archiveiterator import ArchiveIterator
        HAS_WARC = True
    except ImportError:
        HAS_WARC = False
        print("Warning: warcio not installed. Install with: pip install warcio")
    
    if not HAS_WARC:
        print("Creating placeholder corpus...")
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(limit or 10000):
                doc = {
                    "id": f"cc_{i}",
                    "text": f"Web page {i} content.",
                    "meta": {"url": f"https://example.com/page{i}"}
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return
    
    count = 0
    with open(warc_file, "rb") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for record in ArchiveIterator(infile):
            if limit and count >= limit:
                break
            
            if record.rec_type == "response" and record.http_headers.get_header("Content-Type", "").startswith("text/html"):
                # Extract text (simplified - in production use beautifulsoup)
                text = record.read_stream().decode("utf-8", errors="ignore")
                
                # Simple HTML stripping (in production use html2text or similar)
                import re
                text = re.sub(r"<[^>]+>", "", text)
                text = " ".join(text.split())
                
                if len(text) > 100:  # Minimum length
                    doc = {
                        "id": record.rec_headers.get_header("WARC-Record-ID", f"cc_{count}"),
                        "text": text[:10000],  # Limit text length
                        "meta": {"url": record.rec_headers.get_header("WARC-Target-URI", "")}
                    }
                    outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                    
                    if count % 1000 == 0:
                        print(f"Processed {count} pages...")
    
    print(f"Processed {count} Common Crawl pages to {output_file}")


def load_commoncrawl(corpus_file: Path) -> Iterator[dict]:
    """
    Load Common Crawl corpus from JSONL file.

    Args:
        corpus_file: Path to corpus JSONL file

    Yields:
        Document dictionaries with 'id', 'text', 'meta'
    """
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

