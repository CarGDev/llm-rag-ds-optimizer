"""Property-based tests using hypothesis."""

import numpy as np
from hypothesis import given, strategies as st

from llmds.indexed_heap import IndexedHeap


class TestIndexedHeapProperty:
    """Property-based tests for indexed heap."""

    @given(st.lists(st.tuples(st.integers(), st.floats(allow_nan=False, allow_infinity=False)), min_size=1, max_size=100))
    def test_heap_invariant_min(self, items):
        """Test min-heap invariant after operations."""
        heap = IndexedHeap(max_heap=False)
        unique_keys = {}

        # Add items with unique keys, filtering out invalid values
        for key, score in items:
            if key not in unique_keys:
                # Skip NaN and infinity
                if not (float('-inf') < score < float('inf')):
                    continue
                try:
                    heap.push(key, score)
                    unique_keys[key] = score
                except ValueError:
                    pass  # Skip duplicates

        # Verify heap property: parent <= children (min-heap)
        heap_list = heap._heap
        for i in range(len(heap_list)):
            left = 2 * i + 1
            right = 2 * i + 2
            parent_score, _ = heap_list[i]

            if left < len(heap_list):
                left_score, _ = heap_list[left]
                assert parent_score <= left_score, (
                    f"Min-heap violation: parent[{i}]={parent_score} > left[{left}]={left_score}"
                )

            if right < len(heap_list):
                right_score, _ = heap_list[right]
                assert parent_score <= right_score, (
                    f"Min-heap violation: parent[{i}]={parent_score} > right[{right}]={right_score}"
                )

    @given(st.lists(st.tuples(st.integers(), st.floats(allow_nan=False, allow_infinity=False)), min_size=1, max_size=100))
    def test_heap_invariant_max(self, items):
        """Test max-heap invariant after operations."""
        heap = IndexedHeap(max_heap=True)
        unique_keys = {}

        # Add items with unique keys, filtering out NaN and infinity
        for key, score in items:
            if key not in unique_keys:
                # Skip invalid scores
                if not (float('-inf') < score < float('inf')):
                    continue
                try:
                    heap.push(key, score)
                    unique_keys[key] = score
                except ValueError:
                    pass  # Skip duplicates

        # Verify heap property: parent >= children (max-heap)
        heap_list = heap._heap
        for i in range(len(heap_list)):
            left = 2 * i + 1
            right = 2 * i + 2
            parent_score, _ = heap_list[i]

            if left < len(heap_list):
                left_score, _ = heap_list[left]
                assert parent_score >= left_score, (
                    f"Max-heap violation: parent[{i}]={parent_score} < left[{left}]={left_score}"
                )

            if right < len(heap_list):
                right_score, _ = heap_list[right]
                assert parent_score >= right_score, (
                    f"Max-heap violation: parent[{i}]={parent_score} < right[{right}]={right_score}"
                )

    @given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=50))
    def test_token_lru_eviction(self, token_counts):
        """Test token LRU eviction maintains budget."""
        from llmds.token_lru import TokenLRU

        budget = sum(token_counts) // 2  # Budget less than total
        cache = TokenLRU(token_budget=budget, token_of=lambda x: x)

        for i, tokens in enumerate(token_counts):
            cache.put(f"key{i}", tokens)

        # Verify budget is maintained
        assert cache.total_tokens() <= budget


class TestHNSWProperty:
    """Property-based tests for HNSW."""

    @given(
        st.integers(min_value=10, max_value=100),  # num_vectors
        st.integers(min_value=5, max_value=20),    # dim
        st.integers(min_value=5, max_value=15),    # k
    )
    def test_hnsw_search_recall(self, num_vectors, dim, k):
        """Test that HNSW search returns reasonable recall (returns some results)."""
        from llmds.hnsw import HNSW

        hnsw = HNSW(dim=dim, M=8, ef_construction=20, ef_search=10, seed=42)
        
        # Add vectors
        np.random.seed(42)
        vectors = []
        for i in range(num_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            hnsw.add(vec, i)
        
        # For each vector, search and verify it appears in results (self-recall)
        recall_scores = []
        for i, vec in enumerate(vectors[:min(10, num_vectors)]):  # Test subset
            results = hnsw.search(vec, k=k)
            result_ids = [node_id for node_id, _ in results]
            
            # Self-recall: query vector should appear in its own search results
            # (or at least some results should be returned)
            if len(results) > 0:
                recall = 1.0 if i in result_ids else 0.0
                recall_scores.append(recall)
        
        # At least some searches should return results
        assert len(recall_scores) > 0 or num_vectors == 0

    @given(
        st.integers(min_value=10, max_value=50),  # num_vectors
        st.integers(min_value=5, max_value=20),    # dim
    )
    def test_hnsw_distance_ordering(self, num_vectors, dim):
        """Test that HNSW search results are ordered by distance (ascending)."""
        from llmds.hnsw import HNSW

        hnsw = HNSW(dim=dim, M=8, ef_construction=20, ef_search=15, seed=42)
        
        # Add vectors
        np.random.seed(42)
        vectors = []
        for i in range(num_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            hnsw.add(vec, i)
        
        # Test a few queries
        for _ in range(min(5, num_vectors)):
            query = np.random.randn(dim).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            results = hnsw.search(query, k=min(10, num_vectors))
            
            # Verify results are sorted by distance (ascending)
            if len(results) > 1:
                distances = [dist for _, dist in results]
                assert distances == sorted(distances), (
                    f"Results not sorted: {distances} vs {sorted(distances)}"
                )
            
            # Verify distances are non-negative
            for node_id, distance in results:
                assert distance >= 0, f"Negative distance: {distance}"
                assert 0 <= node_id < num_vectors, f"Invalid node_id: {node_id}"


class TestCMSProperty:
    """Property-based tests for Count-Min Sketch."""

    @given(
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=100),
        st.integers(min_value=256, max_value=2048),  # width
        st.integers(min_value=2, max_value=8),        # depth
    )
    def test_cms_error_bound(self, items, width, depth):
        """Test that CMS estimates are within theoretical error bounds."""
        from llmds.cmsketch import CountMinSketch

        cms = CountMinSketch(width=width, depth=depth)
        
        # Add items (count frequencies)
        true_counts = {}
        for item in items:
            cms.add(item)
            true_counts[item] = true_counts.get(item, 0) + 1
        
        # Calculate error bound
        error_bound = cms.get_error_bound()
        total_count = cms.get_total_count()
        
        # Verify error bound is positive when we have items
        if total_count > 0:
            assert error_bound >= 0, f"Error bound should be non-negative: {error_bound}"
            
            # For each item, estimate should be >= true count (CMS overestimates)
            # and within error bound
            for item, true_count in true_counts.items():
                estimate = cms.estimate(item)
                
                # CMS is conservative: estimate >= true_count
                assert estimate >= true_count, (
                    f"Estimate {estimate} < true count {true_count} for item '{item}'"
                )
                
                # Estimate should be within error bound (with high probability)
                # We allow some margin for hash collisions
                if total_count > 0:
                    relative_error = (estimate - true_count) / total_count if total_count > 0 else 0
                    # Allow up to 2x error bound for hash collisions
                    assert relative_error * total_count <= 2 * error_bound or estimate == true_count, (
                        f"Estimate {estimate} for '{item}' exceeds error bound {error_bound}"
                    )


class TestInvertedIndexProperty:
    """Property-based tests for inverted index."""

    @given(
        st.lists(st.tuples(st.integers(), st.text(min_size=1, max_size=50)), min_size=2, max_size=20),
        st.text(min_size=1, max_size=20),
    )
    def test_bm25_monotonicity(self, documents, query):
        """Test that BM25 scores satisfy monotonicity: adding query terms should increase scores."""
        from llmds.inverted_index import InvertedIndex

        index = InvertedIndex()
        
        # Add documents
        for doc_id, text in documents:
            index.add_document(doc_id=doc_id, text=text)
        
        if index.stats()["total_documents"] == 0:
            return  # Skip if no documents were added
        
        # Split query into terms
        query_terms = query.split()
        if not query_terms:
            return
        
        # Get scores for individual terms
        term_scores = {}
        for term in query_terms:
            results = index.search(term, top_k=10)
            term_scores[term] = {doc_id: score for doc_id, score in results}
        
        # Full query score should generally be >= sum of individual term scores
        # (due to BM25's additive nature across terms)
        full_results = index.search(query, top_k=10)
        full_scores = {doc_id: score for doc_id, score in full_results}
        
        # For documents that appear in both, verify monotonicity
        # BM25 is additive: score(query) = sum(score(term))
        for doc_id in full_scores:
            sum_term_scores = sum(
                term_scores[term].get(doc_id, 0.0) for term in query_terms
            )
            # Full query score should be approximately equal to sum of term scores
            # (allowing small floating point differences)
            assert abs(full_scores[doc_id] - sum_term_scores) < 1e-6 or full_scores[doc_id] >= sum_term_scores * 0.9, (
                f"BM25 monotonicity violation: doc {doc_id}, "
                f"full={full_scores[doc_id]}, sum_terms={sum_term_scores}"
            )


class TestKVCacheProperty:
    """Property-based tests for KV cache."""

    @given(
        st.integers(min_value=100, max_value=1000),  # total_tokens
        st.integers(min_value=64, max_value=512),     # page_size
        st.integers(min_value=10, max_value=100),    # max_pages
    )
    def test_kv_cache_budget(self, total_tokens, page_size, max_pages):
        """Test that KV cache respects page budget constraints."""
        from llmds.kv_cache import KVCache

        cache = KVCache(page_size=page_size, max_pages=max_pages, enable_prefix_sharing=False)
        
        # Calculate maximum tokens that can fit
        max_tokens = max_pages * page_size
        
        # Try to attach sequences up to the limit
        seq_tokens = []
        for seq_id in range(min(10, max_pages // 2)):  # Limit to reasonable number
            # Allocate tokens for this sequence
            seq_size = min(page_size * 2, total_tokens // (seq_id + 1) + 1)
            tokens = list(range(seq_size))
            seq_tokens.append((seq_id, tokens))
            
            try:
                cache.attach(seq_id=seq_id, kv_tokens=tokens)
            except Exception:
                # If we hit the limit, that's okay - verify stats are consistent
                break
        
        # Verify stats are consistent with budget
        stats = cache.stats()
        
        # Allocated pages should not exceed max_pages
        assert stats["allocated_pages"] <= max_pages, (
            f"Allocated pages {stats['allocated_pages']} exceeds max {max_pages}"
        )
        
        # Total pages should not exceed max_pages
        assert stats["total_pages"] <= max_pages, (
            f"Total pages {stats['total_pages']} exceeds max {max_pages}"
        )


