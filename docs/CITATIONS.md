# Research Citations and Implementation Mapping

This document maps research papers to their implementations in the codebase.

## HNSW (Hierarchical Navigable Small World)

**Implementation:** `llmds/hnsw.py`

**Primary Citation:**
- Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4), 824-836.

**Related Papers:**
- Efficient Vector Search on Disaggregated Memory with d-HNSW (for memory-efficient variations)

**Techniques Implemented:**
- Hierarchical multi-layer graph structure (`_layers`)
- Greedy search algorithm (`_search_layer`)
- Level assignment with exponential distribution (`_random_level`)
- Entry point selection and navigation
- Dynamic connection management (M parameter)
- efConstruction and efSearch parameters for quality/speed trade-offs

**Code References:**
- `HNSW` class: Main implementation
- `_random_level()`: Level assignment following exponential distribution
- `_search_layer()`: Greedy search in a single layer
- `add()`: Vector insertion with connection management
- `search()`: Multi-layer search from top to bottom

## KV Cache with Prefix Sharing

**Implementation:** `llmds/kv_cache.py`, `llmds/paged_allocator.py`

**Primary Citation:**
- Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation (specific paper on KV cache optimization for RAG)

**Techniques Implemented:**
- Paged allocation with fixed-size pages (`PagedAllocator`)
- Prefix/prompt sharing with copy-on-write semantics (`KVCache._copy_if_shared`)
- Hash-based deduplication (`_hash_prefix`)
- Reference counting for shared pages (`_page_refs`)
- Defensive copying to prevent corruption (`get()` returns deep copies)

**Code References:**
- `KVCache` class: Main KV cache implementation
- `PagedAllocator` class: Page-based memory management
- `_copy_if_shared()`: Copy-on-write implementation
- `_hash_prefix()`: SHA256-based prefix hashing
- `attach()` / `detach()`: Sequence management with reference counting

## Count-Min Sketch

**Implementation:** `llmds/cmsketch.py`

**Primary Citations:**
- Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: the count-min sketch and its applications. Journal of Algorithms, 55(1), 58-75.
- Fair-Count-Min: Frequency Estimation under Equal Group-wise Approximation Factor

**Techniques Implemented:**
- Count-Min Sketch with multiple hash functions (`depth` parameter)
- Conservative update strategy to reduce overestimation bias
- Error bound calculation (`get_error_bound()`)
- Hot item detection (`is_hot()`)

**Code References:**
- `CountMinSketch` class: Main sketch implementation
- `add()`: Conservative update algorithm
- `estimate()`: Minimum across all hash rows
- `get_error_bound()`: Theoretical error bound calculation
- Uses MurmurHash3 for hash functions

## BM25 Inverted Index

**Implementation:** `llmds/inverted_index.py`

**Primary Citation:**
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.

**Techniques Implemented:**
- BM25 scoring formula with k1 and b parameters
- Inverted index structure with compressed postings
- Varint encoding for integer compression (`_encode_varint`)
- Zigzag encoding for signed integers (`_zigzag_encode`)
- Term frequency and document frequency tracking

**Code References:**
- `InvertedIndex` class: Main inverted index implementation
- `_bm25_score()`: BM25 scoring function
- `add_document()`: Index construction
- `search()`: BM25 retrieval

## Hybrid Retrieval (Dense + Sparse)

**Implementation:** `llmds/retrieval_pipeline.py`

**Primary Citation:**
- Survey of Filtered Approximate Nearest Neighbor Search over the Vector-Scalar Hybrid Data

**Techniques Implemented:**
- Hybrid dense (HNSW) + sparse (BM25) retrieval
- Score fusion with configurable weights (`fusion_weight` parameter)
- Top-K maintenance using indexed heap
- Hot query caching using Count-Min Sketch

**Code References:**
- `RetrievalPipeline` class: End-to-end hybrid retrieval
- `search()`: Combines HNSW and BM25 with score fusion
- Uses `IndexedHeap` for efficient top-K maintenance

## Indexed Heap

**Implementation:** `llmds/indexed_heap.py`

**Techniques Implemented:**
- Indexed binary heap for O(log n) priority updates
- Support for both min-heap and max-heap
- O(1) key lookup via position map (`_pos` dictionary)
- Decrease/increase key operations with correct bubble directions

**Code References:**
- `IndexedHeap` class: Indexed heap implementation
- `decrease_key()` / `increase_key()`: Key update operations
- `_bubble_up()` / `_bubble_down()`: Heap property maintenance

## Scheduler and Batching

**Implementation:** `llmds/scheduler.py`, `llmds/admissions.py`

**Techniques Implemented:**
- Dynamic micro-batching with configurable wait time
- Priority queue using indexed heap
- Admission control with QPS and token rate limiting
- Moving window average for rate tracking

**Code References:**
- `Scheduler` class: Batching scheduler
- `AdmissionController` class: Rate limiting and admission control
- Uses `IndexedHeap` for priority queue

## Token-Aware LRU

**Implementation:** `llmds/token_lru.py`

**Techniques Implemented:**
- LRU eviction with token-aware budgeting
- Cumulative token tracking across cache entries
- Eviction based on token count rather than entry count

**Code References:**
- `TokenLRU` class: Token-aware LRU cache
- `total_tokens()`: Cumulative token tracking
- `put()`: Token-aware insertion with eviction

---

## How to Cite

When using this codebase in research, please cite the relevant papers:

1. **HNSW**: Cite Malkov & Yashunin (2018) for HNSW algorithm
2. **KV Cache**: Cite Cache-Craft paper for prefix sharing techniques
3. **Count-Min Sketch**: Cite Cormode & Muthukrishnan (2005) for Count-Min Sketch
4. **BM25**: Cite Robertson & Zaragoza (2009) for BM25 scoring
5. **Hybrid Retrieval**: Cite survey paper for hybrid dense+sparse approaches

## Additional References

- Papers in `papers/` directory contain full citations and implementation details
- See `README.md` for usage examples and performance benchmarks

