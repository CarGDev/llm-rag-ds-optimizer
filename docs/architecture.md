# Architecture Overview

## System Architecture

The LLM Data Structures Optimizer is organized into several key subsystems:

### 1. KV Cache System

```
┌─────────────────────────────────────────┐
│         KVCache                         │
│  ┌───────────────────────────────────┐  │
│  │  Prefix Hash Map                  │  │
│  │  (SHA256-based deduplication)     │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Sequence → Page Mapping          │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  PagedAllocator                   │  │
│  │  - Fixed-size pages               │  │
│  │  - Free-list management           │  │
│  │  - Defragmentation                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Key Features:**
- **Copy-on-write (COW)** for prefix sharing - shared pages are read-only until modified, then lazily copied
- **Reference counting** - shared pages are tracked and only freed when all references are released
- **Hash-based deduplication** - identical prefixes are automatically detected and shared
- **Page-level allocation granularity** - efficient memory management with fixed-size pages
- **Defensive copying** - `get()` returns deep copies to prevent external modification of shared data

### 2. Scheduler & Batching

```
┌─────────────────────────────────────────┐
│         Scheduler                       │
│  ┌───────────────────────────────────┐  │
│  │  IndexedHeap (Max-Heap Priority Queue) │  │
│  │  - O(log n) decrease/increase-key     │  │
│  │  - Priority by remaining tokens       │  │
│  │  - Fixed bubble directions (v0.1.0)   │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  AdmissionController              │  │
│  │  - QPS limiting                   │  │
│  │  - Token rate limiting            │  │
│  │  - Moving window average          │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Key Features:**
- Dynamic micro-batching with configurable wait time
- SLO-aware prioritization
- Rate limiting and admission control

### 3. Retrieval Pipeline

```
┌─────────────────────────────────────────┐
│      RetrievalPipeline                  │
│  ┌───────────────────────────────────┐  │
│  │  HNSW (Dense Search)               │  │
│  │  - Hierarchical graph              │  │
│  │  - Approximate nearest neighbor    │  │
│  │  - Reproducible via seed parameter  │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  InvertedIndex (BM25)             │  │
│  │  - Compressed postings             │  │
│  │  - Varint/zigzag encoding          │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Score Fusion                     │  │
│  │  - Weighted combination            │  │
│  │  - Top-K heap maintenance         │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  CountMinSketch                   │  │
│  │  - Hot query detection             │  │
│  │  - Cache priming                   │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Key Features:**
- Hybrid dense + sparse retrieval
- Score fusion with configurable weights
- Hot query caching

## Data Flow

### KV Cache Flow

1. **Attach Sequence**: Allocate pages, hash prefix, check for sharing
2. **Get Sequence**: Retrieve pages, reconstruct KV tokens
3. **Detach Sequence**: Free pages, update statistics

### Scheduler Flow

1. **Submit Request**: Add to priority queue, update admission stats
2. **Get Batch**: Check wait time, pop top-k requests
3. **Complete Batch**: Remove from queue, update metrics

### Retrieval Flow

1. **Index Building**: Add documents to HNSW and inverted index
2. **Query Processing**: 
   - Dense search (HNSW)
   - Sparse search (BM25)
   - Score fusion
   - Top-K selection
3. **Caching**: Check CMS for hot queries, cache results

## Memory Management

### Token Budgeting

- Global token budget manager tracks:
  - KV cache tokens
  - Prompt tokens
  - Context window tokens

### Page Allocation

- Fixed-size pages reduce fragmentation
- Free-list management for O(1) allocation
- Periodic defragmentation for compaction

## Performance Characteristics

### Time Complexities

- **KV Cache**: O(1) attach/get, O(k) detach (k = pages)
- **Indexed Heap**: O(log n) push/pop/update
- **HNSW Search**: O(log n) approximate nearest neighbor
- **BM25 Search**: O(|query| × avg_doc_freq)

### Space Complexities

- **KV Cache**: O(sequences × tokens_per_seq)
- **HNSW**: O(n × M) where M = max connections
- **Inverted Index**: O(|vocab| × avg_postings)

## Trade-offs

### Page Size
- **Small pages**: Better memory utilization, higher overhead
- **Large pages**: Lower overhead, more fragmentation

### Batch Size
- **Small batches**: Lower latency, lower throughput
- **Large batches**: Higher throughput, higher latency

### HNSW Parameters
- **M (connections)**: Higher = better recall, more memory
- **efSearch**: Higher = better recall, slower search

