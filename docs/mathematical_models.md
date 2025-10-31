# Mathematical Models

This document describes the mathematical formulations and algorithms used throughout the LLM RAG Data Structures Optimizer.

## Table of Contents

- [BM25 Ranking Function](#bm25-ranking-function)
- [HNSW Distance Metrics](#hnsw-distance-metrics)
- [Count-Min Sketch Error Bounds](#count-min-sketch-error-bounds)
- [Score Fusion](#score-fusion)
- [KV Cache Memory Calculation](#kv-cache-memory-calculation)
- [Token-Aware LRU Eviction](#token-aware-lru-eviction)
- [Admission Control Rate Limiting](#admission-control-rate-limiting)

---

## BM25 Ranking Function

BM25 (Best Matching 25) is a probabilistic ranking function used for information retrieval. It scores documents based on term frequency and inverse document frequency.

### Formula

For a query \( Q = \{q_1, q_2, \ldots, q_n\} \) and document \( D \), the BM25 score is:

\[
\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
\]

Where:
- \( f(q_i, D) \) = frequency of term \( q_i \) in document \( D \)
- \( |D| \) = length of document \( D \) (number of terms)
- \( \text{avgdl} \) = average document length in the collection
- \( k_1 \) = term frequency saturation parameter (typically 1.2-2.0)
- \( b \) = length normalization parameter (typically 0.75)

### Inverse Document Frequency (IDF)

\[
\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}
\]

Where:
- \( N \) = total number of documents in the collection
- \( n(q_i) \) = number of documents containing term \( q_i \)

The 0.5 smoothing factor prevents division by zero and handles terms that appear in all documents.

### Implementation Defaults

In our implementation:
- \( k_1 = 1.5 \) (default)
- \( b = 0.75 \) (default)

---

## HNSW Distance Metrics

Hierarchical Navigable Small World (HNSW) uses distance metrics to measure similarity between vectors. The default distance metric is **L2 (Euclidean) distance**.

### L2 Distance (Euclidean)

For vectors \( \vec{u} = (u_1, u_2, \ldots, u_d) \) and \( \vec{v} = (v_1, v_2, \ldots, v_d) \):

\[
d_{\text{L2}}(\vec{u}, \vec{v}) = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}
\]

In practice, we often use squared L2 distance for efficiency (monotonic with L2):

\[
d_{\text{L2}}^2(\vec{u}, \vec{v}) = \sum_{i=1}^{d} (u_i - v_i)^2
\]

### Cosine Similarity (Alternative)

For normalized vectors, cosine similarity is often preferred:

\[
\text{cosine}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{||\vec{u}|| \cdot ||\vec{v}||} = \frac{\sum_{i=1}^{d} u_i \cdot v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \cdot \sqrt{\sum_{i=1}^{d} v_i^2}}
\]

For normalized vectors where \( ||\vec{u}|| = ||\vec{v}|| = 1 \):

\[
\text{cosine}(\vec{u}, \vec{v}) = \vec{u} \cdot \vec{v} = \sum_{i=1}^{d} u_i \cdot v_i
\]

**Note**: Cosine similarity can be converted to distance: \( d_{\text{cosine}} = 1 - \text{cosine}(\vec{u}, \vec{v}) \)

### HNSW Graph Properties

The HNSW graph has logarithmic search complexity:

- **Search complexity**: \( O(\log N) \) where \( N \) is the number of vectors
- **Construction complexity**: \( O(N \log N) \)
- **Memory complexity**: \( O(N \cdot M) \) where \( M \) is the maximum connections per node

**Return Format**: The `search()` and `_search_layer()` methods return results as `(node_id, distance)` tuples, where:
- `node_id`: Integer identifier of the vector in the index
- `distance`: Float representing the L2 distance from the query vector

---

## Count-Min Sketch Error Bounds

Count-Min Sketch is a probabilistic data structure for frequency estimation with bounded error.

### Structure

A Count-Min Sketch has width \( w \) and depth \( d \), creating a \( d \times w \) table of counters.

### Update Operation

For item \( x \) with count \( c \), update all \( d \) rows:

\[
\text{CM}[i][h_i(x)] \leftarrow \text{CM}[i][h_i(x)] + c, \quad \forall i \in \{1, 2, \ldots, d\}
\]

Where \( h_i(x) \) is a hash function for row \( i \).

### Estimation

The estimated frequency is the minimum across all rows:

\[
\hat{f}(x) = \min_{i \in \{1, \ldots, d\}} \text{CM}[i][h_i(x)]
\]

### Error Bound

With probability at least \( 1 - \delta \), the error is bounded by:

\[
\hat{f}(x) - f(x) \leq \epsilon \cdot ||\mathbf{f}||_1
\]

Where:
- \( f(x) \) = true frequency of \( x \)
- \( ||\mathbf{f}||_1 \) = total count of all items (L1 norm)
- \( \epsilon = \frac{e}{w} \) (where \( e \approx 2.71828 \))
- \( \delta = \left(\frac{1}{2}\right)^d \)

### Parameter Selection

To achieve error bound \( \epsilon \) with probability \( 1 - \delta \):

\[
w = \left\lceil \frac{e}{\epsilon} \right\rceil
\]
\[
d = \left\lceil \ln \frac{1}{\delta} \right\rceil
\]

**Default parameters** in our implementation:
- \( w = 2048 \) → \( \epsilon \approx 0.0013 \)
- \( d = 4 \) → \( \delta = 0.0625 \) (6.25% error probability)

---

## Score Fusion

Hybrid search combines scores from multiple retrieval methods (dense vectors and sparse keywords).

### Weighted Linear Combination

\[
S_{\text{fused}}(d, q) = \alpha \cdot S_{\text{dense}}(d, q) + \beta \cdot S_{\text{sparse}}(d, q)
\]

Where:
- \( S_{\text{dense}}(d, q) \) = normalized vector similarity score
- \( S_{\text{sparse}}(d, q) \) = normalized BM25 score
- \( \alpha + \beta = 1 \) (typically \( \alpha = 0.7 \), \( \beta = 0.3 \))

### Score Normalization

Before fusion, scores are normalized to [0, 1] range:

\[
S_{\text{norm}}(d, q) = \frac{S(d, q) - S_{\min}}{S_{\max} - S_{\min}}
\]

Where \( S_{\min} \) and \( S_{\max} \) are the minimum and maximum scores in the candidate set.

### Reciprocal Rank Fusion (Alternative)

\[
S_{\text{RRF}}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
\]

Where:
- \( R \) = set of ranked lists to fuse
- \( \text{rank}_r(d) \) = rank of document \( d \) in ranked list \( r \)
- \( k \) = smoothing parameter (typically 60)

---

## KV Cache Memory Calculation

The KV cache memory usage depends on the number of cached tokens and the model dimensions.

### Per-Sequence Memory

For a sequence with \( T \) tokens and model with hidden dimension \( d \):

\[
M_{\text{sequence}} = 2 \cdot T \cdot d \cdot \text{bytes\_per\_element}
\]

Where:
- Factor of 2 accounts for both key and value tensors
- \( \text{bytes\_per\_element} = 4 \) for float32, \( 2 \) for float16

### Paged Allocation

With page size \( P \) pages and page capacity \( C \) tokens per page:

\[
M_{\text{paged}} = \left\lceil \frac{T}{C} \right\rceil \cdot P \cdot \text{page\_overhead}
\]

Where \( \text{page\_overhead} \) includes page metadata.

### Prefix Sharing Memory Savings

If \( N \) sequences share a prefix of length \( L \):

\[
M_{\text{shared}} = L \cdot d \cdot \text{bytes\_per\_element}
\]
\[
M_{\text{without\_sharing}} = N \cdot L \cdot d \cdot \text{bytes\_per\_element}
\]

Memory savings:

\[
\text{Savings} = (N - 1) \cdot L \cdot d \cdot \text{bytes\_per\_element}
\]

Savings ratio:

\[
\text{Savings Ratio} = \frac{N - 1}{N} = 1 - \frac{1}{N}
\]

For large \( N \), this approaches 100% savings on shared prefixes.

---

## Token-Aware LRU Eviction

Token-aware LRU maintains a cumulative token budget while evicting least recently used items.

### Eviction Criterion

Evict item \( i \) with minimum value of:

\[
\text{priority}(i) = \frac{\text{access\_count}(i)}{\text{token\_count}(i)}
\]

Or use recency-weighted:

\[
\text{priority}(i) = \frac{\text{last\_access\_time}(i)}{\text{token\_count}(i)}
\]

### Token Budget Constraint

Maintain total tokens below budget \( B \):

\[
\sum_{i \in \text{cache}} \text{token\_count}(i) \leq B
\]

When adding item \( j \) with \( t_j \) tokens:

1. If \( \sum_{i} t_i + t_j \leq B \): add item
2. Else: evict items until \( \sum_{i} t_i + t_j \leq B \)

### Eviction Algorithm

```
while total_tokens + new_tokens > budget:
    item = item_with_min_priority()
    total_tokens -= token_count(item)
    evict(item)
```

---

## Admission Control Rate Limiting

The admission controller uses an exponentially weighted moving average (EWMA) to track request rate.

### Moving Average Update

\[
\bar{r}_{t} = \alpha \cdot r_t + (1 - \alpha) \cdot \bar{r}_{t-1}
\]

Where:
- \( r_t \) = current request rate
- \( \bar{r}_t \) = smoothed average rate
- \( \alpha \) = smoothing factor (typically 0.1-0.3)

### Admission Decision

Admit request if:

\[
\bar{r}_t + \text{margin} \leq R_{\max}
\]

Where:
- \( R_{\max} \) = maximum allowed rate (QPS limit)
- \( \text{margin} \) = safety margin to account for burstiness

### Token Bucket (Alternative)

The token bucket algorithm allows bursty traffic:

\[
\text{tokens}(t) = \min(B, \text{tokens}(t-1) + R \cdot \Delta t)
\]

Where:
- \( B \) = bucket capacity (burst limit)
- \( R \) = token refill rate (sustainable rate)
- \( \Delta t \) = time since last update

Request is admitted if \( \text{tokens}(t) \geq 1 \), then \( \text{tokens}(t) \leftarrow \text{tokens}(t) - 1 \).

---

## Indexed Binary Heap

The indexed heap supports O(log n) priority updates.

### Heap Properties

For a min-heap with \( n \) elements:

- Parent of node \( i \): \( \lfloor (i-1)/2 \rfloor \)
- Left child of node \( i \): \( 2i + 1 \)
- Right child of node \( i \): \( 2i + 2 \)

### Heap Invariant

\[
\text{priority}(\text{parent}(i)) \leq \text{priority}(i), \quad \forall i > 0
\]

### Complexity

- Insert: \( O(\log n) \)
- Extract min: \( O(\log n) \)
- Update key: \( O(\log n) \) (with index mapping)
- Decrease key: \( O(\log n) \)

---

## References

1. **BM25**: Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

2. **HNSW**: Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE transactions on pattern analysis and machine intelligence*, 42(4), 824-836.

3. **Count-Min Sketch**: Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: the count-min sketch and its applications. *Journal of Algorithms*, 55(1), 58-75.

4. **Score Fusion**: Cormack, G. V., Clarke, C. L., & Büttcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *Proceedings of the 32nd international ACM SIGIR conference*.

---

**Last Updated**: 2024-10-30

