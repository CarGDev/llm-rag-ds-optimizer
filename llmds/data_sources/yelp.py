"""Yelp Open Dataset loader."""

import json
from pathlib import Path
from typing import Iterator


def download_yelp(output_dir: Path) -> Path:
    """
    Download Yelp Open Dataset.

    Args:
        output_dir: Directory to save corpus

    Returns:
        Path to corpus JSONL file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = output_dir / "business_reviews.jsonl"
    
    if corpus_file.exists():
        print(f"Yelp corpus already exists at {corpus_file}")
        return corpus_file
    
    print("Yelp Open Dataset requires manual download from https://www.yelp.com/dataset")
    print("After downloading, extract business.json and review.json")
    print("Then run: python scripts/process_yelp.py --business <path> --review <path> --output <path>")
    
    # Placeholder implementation
    print("Creating placeholder corpus...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for i in range(1000):
            doc = {
                "id": f"yelp_{i}",
                "text": f"Yelp business {i} review content. This is a placeholder.",
                "meta": {"business_id": f"biz_{i}", "rating": 4.5}
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    return corpus_file


def process_yelp_files(business_file: Path, review_file: Path, output_file: Path, limit: int | None = None) -> None:
    """
    Process Yelp JSON files into normalized JSONL.

    Args:
        business_file: Path to business.json
        review_file: Path to review.json
        output_file: Output JSONL path
        limit: Optional limit on documents
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load businesses
    businesses = {}
    if business_file.exists():
        with open(business_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    biz = json.loads(line)
                    businesses[biz["business_id"]] = biz
    
    count = 0
    with open(review_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            if limit and count >= limit:
                break
            
            if line.strip():
                review = json.loads(line)
                biz_id = review.get("business_id")
                biz = businesses.get(biz_id, {})
                
                # Combine business name + review text
                biz_name = biz.get("name", "")
                review_text = review.get("text", "")
                combined = f"{biz_name} {review_text}".strip()
                
                if combined:
                    doc = {
                        "id": f"yelp_{review.get('review_id', count)}",
                        "text": combined,
                        "meta": {
                            "business_id": biz_id,
                            "rating": review.get("stars"),
                            "category": biz.get("categories"),
                        }
                    }
                    outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
    
    print(f"Processed {count} Yelp reviews to {output_file}")


def load_yelp(corpus_file: Path) -> Iterator[dict]:
    """
    Load Yelp corpus from JSONL file.

    Args:
        corpus_file: Path to corpus JSONL file

    Yields:
        Document dictionaries with 'id', 'text', 'meta'
    """
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

