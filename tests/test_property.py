"""Property-based tests using hypothesis."""

from hypothesis import given, strategies as st

from llmds.indexed_heap import IndexedHeap


class TestIndexedHeapProperty:
    """Property-based tests for indexed heap."""

    @given(st.lists(st.tuples(st.integers(), st.floats()), min_size=1, max_size=100))
    def test_heap_invariant_min(self, items):
        """Test min-heap invariant after operations."""
        heap = IndexedHeap(max_heap=False)
        unique_keys = {}

        # Add items with unique keys
        for key, score in items:
            if key not in unique_keys:
                try:
                    heap.push(key, score)
                    unique_keys[key] = score
                except ValueError:
                    pass  # Skip duplicates

        # Verify heap property: parent <= children
        heap_list = heap._heap
        for i in range(len(heap_list)):
            left = 2 * i + 1
            right = 2 * i + 2
            parent_score, _ = heap_list[i]

            if left < len(heap_list):
                left_score, _ = heap_list[left]
                assert parent_score <= left_score

            if right < len(heap_list):
                right_score, _ = heap_list[right]
                assert parent_score <= right_score

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


