# Dataset Sources and Licenses

This document describes the datasets used for benchmarking the LLM RAG Data Structures Optimizer. All datasets are publicly available and suitable for research use.

## Datasets

### Datasets with Published Benchmark Results

We benchmark on three publicly available datasets with published results:

### 1. BEIR FIQA (Financial Question Answering)

**Source**: [BEIR Paper](https://arxiv.org/abs/2104.08663) | [Hugging Face Datasets](https://huggingface.co/datasets/BeIR)

**Description**: Financial question-answering dataset from BEIR benchmark suite. 50,000 documents with financial Q&A pairs. Used as primary evaluation dataset in our research.

**License**: Varies by task. Most BEIR tasks use CC-BY or similar open licenses. Check individual task licenses.

**Download**:
```bash
python scripts/download_corpus.py --source beir:fiqa --output data/raw/beir/fiqa
```

**Citation**:
```
Thakur, N., et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
```

### 2. Amazon Reviews 2023 (McAuley Lab)

**Source**: [Hugging Face - McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

**Description**: Large corpus of Amazon product reviews with metadata (ratings, categories, product IDs). Excellent for e-commerce-style RAG workloads. Benchmark results available for 10k subset.

**License**: CC BY 4.0

**Download**:
```bash
python scripts/download_corpus.py --source amazon23 --output data/raw/amazon23 --limit 500000
```

**Note**: Full dataset is very large (>100M reviews). Use `--limit` for manageable subsets. Benchmark results use 10k document subset.

### 3. MS MARCO (Microsoft Machine Reading Comprehension)

**Source**: [MS MARCO Datasets](https://microsoft.github.io/msmarco/)

**Description**: Large-scale passage ranking dataset with 8.8M passages and 1M queries. Widely used as a canonical information retrieval benchmark. Benchmark results available for 10k subset.

**License**: Research use only. See [MS MARCO Terms](https://microsoft.github.io/msmarco/) for details.

**Download**: 
```bash
python scripts/download_corpus.py --source msmarco --output data/raw/msmarco
```

**Citation**:
```
Bajaj, P., et al. (2016). MS MARCO: A human generated machine reading comprehension dataset.
```

### Additional Available Datasets

The following datasets are available in the codebase but do not yet have published benchmark results:

#### 4. Yelp Open Dataset

**Source**: [Yelp Open Dataset](https://www.yelp.com/dataset)

**Description**: Business listings and reviews from Yelp. Useful for local business and review-based RAG.

**License**: See [Yelp Dataset License](https://www.yelp.com/dataset/license). Research use allowed.

**Download**:
```bash
# First accept license at https://www.yelp.com/dataset/download
python scripts/download_corpus.py --source yelp --output data/raw/yelp
```

#### 5. Wikipedia (English)

**Source**: [Wikimedia Downloads](https://dumps.wikimedia.org/enwiki/latest/)

**Description**: English Wikipedia pages-articles dump. Broad factual corpus for general knowledge RAG.

**License**: CC BY-SA 3.0 and GFDL

**Download**:
```bash
python scripts/download_corpus.py --source wikipedia --output data/raw/wikipedia
```

**Note**: Latest dump is ~20GB compressed. Extracts plain text and titles.

#### 6. Common Crawl (Optional)

**Source**: [Common Crawl](https://commoncrawl.org/) | [cc-downloader](https://github.com/commoncrawl/cc-downloader)

**Description**: Web-scale corpus from billions of web pages. Use for large-scale testing.

**License**: Public domain / various site licenses

**Download**:
```bash
# Be respectful of bandwidth - use specific months
python scripts/download_corpus.py --source commoncrawl --cc-month CC-MAIN-2025-14 --output data/raw/cc --limit 10M
```

**Note**: Common Crawl is extremely large. Use `--limit` and specific months for reproducible, manageable subsets.

## Data Format

All datasets are normalized to JSONL format:

```json
{"id": "doc_123", "text": "Document text content...", "meta": {"field1": "value1", "field2": 42}}
```

Each line contains:
- `id`: Unique document identifier
- `text`: Main text content
- `meta`: Optional metadata (ratings, categories, timestamps, etc.)

## Checksums

Dataset checksums are stored in `data/dataset_cards/` as YAML files:

```yaml
name: amazon_reviews_2023
source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
license: CC BY 4.0
sha256: <checksum>
size_bytes: <size>
download_date: 2024-10-30
```

## Quick Start

### Download All Datasets

```bash
# Create directories
mkdir -p data/raw data/processed data/indices data/embeddings data/dataset_cards

# Download datasets (start with smaller ones)
python scripts/download_corpus.py --source beir:fiqa --output data/raw/beir/fiqa
python scripts/download_corpus.py --source amazon23 --output data/raw/amazon23 --limit 500000
python scripts/download_corpus.py --source msmarco --output data/raw/msmarco
```

### Prepare Embeddings

```bash
python scripts/prepare_embeddings.py \
    --input data/raw/beir/fiqa/corpus.jsonl \
    --output data/embeddings/fiqa.npy \
    --dim 384 \
    --seed 42
```

### Build Indices

```bash
python scripts/build_indices.py \
    --corpus data/raw/beir/fiqa/corpus.jsonl \
    --emb data/embeddings/fiqa.npy \
    --index-dir data/indices/fiqa \
    --bm25 \
    --hnsw \
    --ef 200 \
    --M 16
```

### Run Benchmarks

```bash
python scripts/run_benchmarks.py \
    --corpus fiqa \
    --sizes 10k 50k 100k \
    --ef 50 100 200 \
    --M 8 16 32 \
    --repetitions 5
```

## License Compliance

**Important**: 
- Always check individual dataset licenses before use
- **MS MARCO**: Research use only
- **Amazon Reviews**: CC BY 4.0
- **BEIR (FIQA)**: Varies by task, typically CC-BY or similar open licenses

**Do NOT**:
- Scrape websites without permission
- Redistribute datasets without proper attribution
- Use commercial datasets for commercial purposes without checking licenses

## Reproducibility

All dataset processing is deterministic:
- Fixed random seeds (42) for sampling and embeddings
- SHA256 checksums for verification
- Versioned dataset cards with download dates
- Exact corpus sizes documented in benchmark results

## Dataset Statistics

### Datasets with Published Results

| Dataset | Documents | Size | License | Use Case | Benchmark Results |
|---------|-----------|------|---------|----------|-------------------|
| BEIR (FIQA) | 50,000 | ~13MB | Varies | Financial QA | Yes (10k, 25k, 50k subsets) |
| Amazon Reviews 2023 | 100M+ | ~500GB+ | CC BY 4.0 | E-commerce | Yes (10k subset) |
| MS MARCO | 8.8M passages | ~30GB | Research | IR benchmark | Yes (10k subset) |

### Available Datasets (No Published Results Yet)

| Dataset | Documents | Size | License | Use Case | Status |
|---------|-----------|------|---------|----------|--------|
| Yelp | ~8M businesses | ~8GB | Yelp License | Local business | Data available, no results |
| Wikipedia | 6.7M articles | ~20GB | CC BY-SA 3.0 | General knowledge | Data available, no results |
| Common Crawl | Billions | TB+ | Public domain | Web-scale | Code available, optional |

**Note**: Benchmark results are available for 10k document subsets of FIQA, Amazon23, and MS MARCO. FIQA has additional results for 25k and 50k document subsets. Yelp and Wikipedia datasets are available in the codebase but do not yet have published benchmark results.

*Full dataset statistics are approximate and vary by version. Benchmark results use manageable subsets for reproducible evaluation.*

