# Usage Guide

## Basic Examples

### KV Cache

```python
from llmds import KVCache

# Create cache
cache = KVCache(page_size=512, max_pages=10000)

# Attach KV tokens for a sequence
kv_tokens = [1, 2, 3, 4, 5] * 100  # 500 tokens
cache.attach(seq_id=1, kv_tokens=kv_tokens, prefix_tokens=[1, 2, 3])

# Retrieve
cached = cache.get(seq_id=1)

# Detach when done
cache.detach(seq_id=1)
```

### Scheduler

```python
from llmds import Scheduler

# Create scheduler
scheduler = Scheduler(max_batch_size=32, max_wait_ms=50.0)

# Submit requests
req_id1 = scheduler.submit(tokens=100)
req_id2 = scheduler.submit(tokens=200, slo_ms=100.0)  # SLO deadline

# Get batch (waits for max_wait_ms or until batch is full)
batch = scheduler.get_batch(force=False)

# Process batch...
# scheduler.complete_batch(batch)
```

### Admission Control

```python
from llmds import AdmissionController

# Create controller
controller = AdmissionController(qps_target=10.0, token_rate_limit=10000)

# Check admission
should_admit, reason = controller.should_admit(estimated_tokens=100)
if should_admit:
    # Process request
    controller.record_request(tokens=100)
else:
    # Reject request
    print(f"Rejected: {reason}")
```

### Retrieval Pipeline

```python
from llmds import RetrievalPipeline
import numpy as np

# Create pipeline
pipeline = RetrievalPipeline(embedding_dim=384)

# Add documents
for i in range(100):
    text = f"Document {i} content"
    embedding = np.random.randn(384).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    pipeline.add_document(doc_id=i, text=text, embedding=embedding)

# Search
query = "example query"
query_embedding = np.random.randn(384).astype(np.float32)
query_embedding = query_embedding / np.linalg.norm(query_embedding)

results = pipeline.search(query, query_embedding=query_embedding, top_k=10)
for doc_id, score in results:
    print(f"Doc {doc_id}: {score:.4f}")
```

## Advanced Usage

### Custom Priority Function

```python
from llmds import Scheduler

def custom_priority_fn(req):
    # Prioritize by inverse token count
    return 1.0 / (req.tokens + 1.0)

scheduler = Scheduler(
    max_batch_size=32,
    max_wait_ms=50.0,
    priority_fn=custom_priority_fn
)
```

### Token Budget Management

```python
from llmds import TokenLRU

def token_counter(value):
    return len(str(value))

cache = TokenLRU(token_budget=1000, token_of=token_counter)

# Add items (evicts LRU if budget exceeded)
cache.put("key1", "value with many tokens")
cache.put("key2", "another value")

# Evict until target budget
evicted = cache.evict_until_budget(target_budget=500)
```

### HNSW Parameter Tuning

```python
from llmds import HNSW
import numpy as np

# Tune for better recall (higher memory)
hnsw_high_recall = HNSW(
    dim=384,
    M=32,              # More connections
    ef_construction=400,  # More candidates during build
    ef_search=100      # More candidates during search
)

# Tune for faster search (lower memory)
hnsw_fast = HNSW(
    dim=384,
    M=8,               # Fewer connections
    ef_construction=100,
    ef_search=20       # Fewer candidates
)
```

## Benchmarking

### Running Benchmarks

```python
from benchmarks.bench_kv_cache import benchmark_kv_cache

results = benchmark_kv_cache(
    num_sequences=1000,
    tokens_per_seq=1000,
    page_size=512
)
print(f"P95 latency: {results['attach_p95_ms']:.2f} ms")
```

### Custom Benchmarks

```python
from llmds.utils import Timer, MetricsCollector

collector = MetricsCollector()

for i in range(100):
    with Timer() as t:
        # Your operation here
        pass
    collector.record_latency(t.elapsed * 1000)

metrics = collector.get_metrics()
print(f"P95: {metrics.latency_p95:.2f} ms")
```

## Integration Examples

### RAG Pipeline

```python
from llmds import RetrievalPipeline
import numpy as np

# Initialize
pipeline = RetrievalPipeline(embedding_dim=384)

# Index documents
documents = ["doc1", "doc2", "doc3"]
embeddings = [np.random.randn(384) for _ in documents]
for doc_id, (text, emb) in enumerate(zip(documents, embeddings)):
    emb = emb / np.linalg.norm(emb)
    pipeline.add_document(doc_id=doc_id, text=text, embedding=emb)

# Query
query_emb = np.random.randn(384)
query_emb = query_emb / np.linalg.norm(query_emb)
results = pipeline.search("query", query_embedding=query_emb, top_k=5)
```

### LLM Inference with KV Cache

```python
from llmds import KVCache, Scheduler, TokenLRU

# Setup
kv_cache = KVCache()
scheduler = Scheduler()
token_cache = TokenLRU(token_budget=100000, token_of=lambda x: len(str(x)))

# Process request
seq_id = 1
prompt_tokens = [1, 2, 3, 4, 5]
kv_tokens = generate_kv_cache(prompt_tokens)  # Your function

kv_cache.attach(seq_id=seq_id, kv_tokens=kv_tokens, prefix_tokens=prompt_tokens)

# Use cached KV for generation
cached_kv = kv_cache.get(seq_id)
# ... generate tokens using cached KV ...

# Cleanup
kv_cache.detach(seq_id)
```

