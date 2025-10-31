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

KV cache with prefix sharing and deduplication.

**Methods:**
- `attach(seq_id: int, kv_tokens: list, prefix_tokens: Optional[list] = None) -> None`
- `detach(seq_id: int) -> None`
- `get(seq_id: int) -> Optional[list]`
- `stats() -> dict`: Get cache statistics

**Complexity:** O(1) attach/get, O(k) detach where k = pages

### `llmds.token_lru.TokenLRU`

Token-aware LRU cache with eviction until budget.

**Methods:**
- `put(key: K, value: V) -> None`
- `get(key: K) -> Optional[V]`
- `evict_until_budget(target_budget: int) -> list[tuple[K, V]]`
- `total_tokens() -> int`

**Complexity:** O(1) put/get, O(n) evict_until_budget

### `llmds.indexed_heap.IndexedHeap`

Indexed binary heap with decrease/increase-key.

**Methods:**
- `push(key_id: int, score: float) -> None`
- `pop() -> tuple[float, int]`
- `decrease_key(key_id: int, new_score: float) -> None`
- `increase_key(key_id: int, new_score: float) -> None`
- `delete(key_id: int) -> tuple[float, int]`
- `get_score(key_id: int) -> Optional[float]`

**Complexity:** O(log n) for all operations

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

Hierarchical Navigable Small World graph.

**Methods:**
- `add(vec: np.ndarray, vec_id: int) -> None`
- `search(query: np.ndarray, k: int) -> list[tuple[int, float]]`
- `stats() -> dict`: Get index statistics

**Complexity:** O(log n) search, O(log n × efConstruction) add

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

