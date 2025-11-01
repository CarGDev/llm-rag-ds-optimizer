# API Reference

## Core Modules

### `llmds.paged_allocator.PagedAllocator`

Paged memory allocator with slab allocation.

**Methods:**
- `alloc(num_pages: int) -> list[int]`: Allocate pages
- `free(page_ids: list[int]) -> None`: Free pages
- `stats() -> PageStats`: Get allocation statistics
- `defragment() -> None`: Defragment pages

**Complexity:** O(1) alloc/free, O(n) defragment

### `llmds.kv_cache.KVCache`

KV cache with prefix sharing and deduplication. Implements copy-on-write (COW) for safe prefix sharing.

**Parameters:**
- `page_size: int = 512` - Size of each KV cache page in tokens
- `max_pages: int = 10000` - Maximum number of pages to allocate
- `enable_prefix_sharing: bool = True` - Enable prefix sharing optimization

**Methods:**
- `attach(seq_id: int, kv_tokens: list, prefix_tokens: Optional[list] = None) -> None` - Attach KV cache for a sequence. Uses COW for shared pages.
- `detach(seq_id: int) -> None` - Detach and free KV cache, with reference counting for shared pages
- `get(seq_id: int) -> Optional[list]` - Get KV cache (returns deep copy to prevent external modification)
- `stats() -> dict` - Get cache statistics including shared pages count and reference counts

**Complexity:** O(1) attach/get, O(k) detach where k = pages

**Copy-on-Write Semantics:**
- Shared pages (from prefix sharing) are read-only until written
- Writes to shared pages trigger lazy copying (COW)
- Reference counting ensures shared pages are only freed when all references are released
- `get()` returns deep copies to prevent external corruption of shared pages

**Safety:** All shared page operations are protected against data corruption through COW and defensive copying.

### `llmds.utils.MemoryProfiler`

Memory profiler for measuring peak RSS (Resident Set Size) during benchmarks.

**Methods:**
- `start() -> None`: Start memory profiling
- `sample() -> int`: Sample current RSS and update peak
- `get_peak_rss_mb() -> float`: Get peak RSS in megabytes
- `get_peak_rss_bytes() -> int`: Get peak RSS in bytes
- `get_current_rss_mb() -> float`: Get current RSS in megabytes
- `get_memory_delta_mb() -> float`: Get memory delta from initial RSS in megabytes

**Context Manager:**
- `memory_profiler() -> Iterator[MemoryProfiler]`: Context manager for automatic profiling

**Usage:**
```python
from llmds.utils import memory_profiler

with memory_profiler() as profiler:
    # Your code here
    profiler.sample()  # Optional: sample at specific points
peak_rss_mb = profiler.get_peak_rss_mb()
```

**Complexity:** O(1) for all operations

### `llmds.token_lru.TokenLRU`

Token-aware LRU cache with eviction until budget.

**Methods:**
- `put(key: K, value: V) -> None`
- `get(key: K) -> Optional[V]`
- `evict_until_budget(target_budget: int) -> list[tuple[K, V]]`
- `total_tokens() -> int`

**Complexity:** O(1) put/get, O(n) evict_until_budget

### `llmds.indexed_heap.IndexedHeap`

Indexed binary heap with decrease/increase-key operations. Supports both min-heap and max-heap modes.

**Parameters:**
- `max_heap: bool = False` - If True, use max-heap (largest score at top), otherwise min-heap

**Methods:**
- `push(key_id: int, score: float) -> None` - Add item to heap
- `pop() -> tuple[float, int]` - Remove and return top element
- `decrease_key(key_id: int, new_score: float) -> None` - Decrease key value (bubbles down for max-heap, up for min-heap)
- `increase_key(key_id: int, new_score: float) -> None` - Increase key value (bubbles up for max-heap, down for min-heap)
- `delete(key_id: int) -> tuple[float, int]` - Remove specific item
- `get_score(key_id: int) -> Optional[float]` - Get score for key_id
- `peek() -> Optional[tuple[float, int]]` - View top element without removing
- `size() -> int` - Get number of elements
- `is_empty() -> bool` - Check if heap is empty

**Complexity:** O(log n) for all operations

**Note:** Fixed max-heap bubble directions (v0.1.0) - `decrease_key` bubbles down and `increase_key` bubbles up for max-heap.

### `llmds.scheduler.Scheduler`

Dynamic micro-batching scheduler.

**Methods:**
- `submit(tokens: int, slo_ms: Optional[float] = None) -> int`
- `get_batch(force: bool = False) -> Optional[list[int]]`
- `complete_batch(request_ids: list[int]) -> None`
- `update_priority(request_id: int, new_tokens: int) -> None`

**Complexity:** O(log n) submit, O(k log n) get_batch where k = batch_size

### `llmds.admissions.AdmissionController`

Admission controller with rate limiting.

**Methods:**
- `should_admit(estimated_tokens: int = 0) -> tuple[bool, str]`
- `record_request(tokens: int) -> None`
- `stats() -> dict`: Get admission statistics

**Complexity:** O(1) should_admit

### `llmds.inverted_index.InvertedIndex`

Compressed inverted index with BM25 scoring.

**Methods:**
- `add_document(doc_id: int, text: str) -> None`
- `search(query: str, top_k: int = 10) -> list[tuple[int, float]]`
- `get_term_frequency(term: str, doc_id: int) -> int`
- `get_document_frequency(term: str) -> int`

**Complexity:** O(|doc|) add_document, O(|query| × avg_doc_freq) search

### `llmds.hnsw.HNSW`

Hierarchical Navigable Small World graph for approximate nearest neighbor search.

**Parameters:**
- `dim: int` - Dimension of vectors
- `M: int = 16` - Maximum number of connections per node
- `ef_construction: int = 200` - Size of candidate set during construction
- `ef_search: int = 50` - Size of candidate set during search
- `ml: float = 1.0 / log(2.0)` - Normalization factor for level assignment
- `seed: Optional[int] = None` - Random seed for reproducible graph structure

**Methods:**
- `add(vec: np.ndarray, vec_id: int) -> None` - Add vector to index
- `search(query: np.ndarray, k: int) -> list[tuple[int, float]]` - Search for k nearest neighbors. Returns list of (vector_id, distance) tuples
- `stats() -> dict` - Get index statistics (num_vectors, num_layers, entry_point, etc.)

**Complexity:** O(log n) search, O(log n × efConstruction) add

**Reproducibility:** When `seed` is provided, each HNSW instance uses its own `random.Random(seed)` state for level assignments, ensuring identical graph structures across runs with the same seed.

### `llmds.cmsketch.CountMinSketch`

Count-Min Sketch for frequency estimation.

**Methods:**
- `add(item: str, count: int = 1) -> None`
- `estimate(item: str) -> int`
- `is_hot(item: str, threshold: int) -> bool`
- `get_error_bound() -> float`

**Complexity:** O(depth) add/estimate

### `llmds.retrieval_pipeline.RetrievalPipeline`

End-to-end retrieval pipeline.

**Methods:**
- `add_document(doc_id: int, text: str, embedding: Optional[np.ndarray] = None) -> None`
- `search(query: str, query_embedding: Optional[np.ndarray] = None, top_k: int = 10, fusion_weight: float = 0.5) -> list[tuple[int, float]]`
- `stats() -> dict`: Get pipeline statistics

**Complexity:** O(log n) search (HNSW) + O(|query| × avg_doc_freq) (BM25)

