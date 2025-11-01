"""Tests for KV cache."""

import copy

import pytest

from llmds.kv_cache import KVCache


class TestKVCache:
    """Test KV cache functionality."""

    def test_attach_detach(self):
        """Test attach and detach operations."""
        cache = KVCache(page_size=512, max_pages=100)
        kv_tokens = [1, 2, 3, 4, 5] * 100  # 500 tokens

        cache.attach(seq_id=1, kv_tokens=kv_tokens)
        assert cache.get(seq_id=1) is not None

        cache.detach(seq_id=1)
        assert cache.get(seq_id=1) is None

    def test_prefix_sharing(self):
        """Test prefix sharing functionality."""
        cache = KVCache(page_size=512, max_pages=100, enable_prefix_sharing=True)
        prefix = [1, 2, 3, 4, 5]
        kv_tokens1 = prefix + [6, 7, 8] * 100
        kv_tokens2 = prefix + [9, 10, 11] * 100

        cache.attach(seq_id=1, kv_tokens=kv_tokens1, prefix_tokens=prefix)
        cache.attach(seq_id=2, kv_tokens=kv_tokens2, prefix_tokens=prefix)

        stats = cache.stats()
        assert stats["prefix_shares"] >= 1  # Should detect sharing
        assert stats["shared_pages_count"] > 0  # Should have shared pages

    def test_prefix_sharing_no_corruption(self):
        """Test that prefix sharing doesn't cause data corruption."""
        cache = KVCache(page_size=10, max_pages=100, enable_prefix_sharing=True)
        
        # Create two sequences with same prefix
        prefix = [1, 2, 3, 4, 5]
        kv_tokens1 = prefix + [10, 20, 30] * 10  # Different suffix
        kv_tokens2 = prefix + [40, 50, 60] * 10  # Different suffix
        
        cache.attach(seq_id=1, kv_tokens=kv_tokens1, prefix_tokens=prefix)
        cache.attach(seq_id=2, kv_tokens=kv_tokens2, prefix_tokens=prefix)
        
        # Get both sequences
        result1 = cache.get(seq_id=1)
        result2 = cache.get(seq_id=2)
        
        # Verify prefix is identical in both
        assert result1[:len(prefix)] == prefix
        assert result2[:len(prefix)] == prefix
        assert result1[:len(prefix)] == result2[:len(prefix)]
        
        # Verify suffixes are different (no corruption)
        assert result1[len(prefix):] != result2[len(prefix):]
        assert result1[len(prefix):] == [10, 20, 30] * 10
        assert result2[len(prefix):] == [40, 50, 60] * 10

    def test_copy_on_write_behavior(self):
        """Test that modifying shared pages triggers copy-on-write."""
        cache = KVCache(page_size=10, max_pages=100, enable_prefix_sharing=True)
        
        prefix = [1, 2, 3]
        kv_tokens1 = prefix + [10, 11, 12] * 5
        
        # First sequence - establishes shared prefix
        cache.attach(seq_id=1, kv_tokens=kv_tokens1, prefix_tokens=prefix)
        initial_stats = cache.stats()
        initial_shared = initial_stats["shared_pages_count"]
        
        # Second sequence - should share prefix, then overwrite triggers COW
        kv_tokens2 = prefix + [20, 21, 22] * 5
        cache.attach(seq_id=2, kv_tokens=kv_tokens2, prefix_tokens=prefix)
        
        # After second attach, shared pages should be copied if needed
        # Since we're writing different data, COW should trigger
        stats_after = cache.stats()
        
        # Verify no corruption
        result1 = cache.get(seq_id=1)
        result2 = cache.get(seq_id=2)
        
        assert result1[:len(prefix)] == prefix
        assert result2[:len(prefix)] == prefix
        assert result1[len(prefix):] == [10, 11, 12] * 5
        assert result2[len(prefix):] == [20, 21, 22] * 5

    def test_shared_pages_reference_counting(self):
        """Test that shared pages maintain correct reference counts."""
        cache = KVCache(page_size=10, max_pages=100, enable_prefix_sharing=True)
        
        prefix = [1, 2, 3]
        kv_tokens = prefix + [10] * 10
        
        # Attach multiple sequences with same prefix
        cache.attach(seq_id=1, kv_tokens=kv_tokens, prefix_tokens=prefix)
        stats1 = cache.stats()
        assert stats1["shared_pages_count"] == 0  # First one, no sharing yet
        
        cache.attach(seq_id=2, kv_tokens=kv_tokens, prefix_tokens=prefix)
        stats2 = cache.stats()
        assert stats2["shared_pages_count"] > 0  # Now sharing
        
        cache.attach(seq_id=3, kv_tokens=kv_tokens, prefix_tokens=prefix)
        stats3 = cache.stats()
        # Shared pages should still exist (reference count > 0)
        
        # Detach one sequence - shared pages should still exist
        cache.detach(seq_id=1)
        stats_after_detach = cache.stats()
        assert stats_after_detach["total_sequences"] == 2
        
        # Detach all - shared pages should be freed
        cache.detach(seq_id=2)
        cache.detach(seq_id=3)
        final_stats = cache.stats()
        assert final_stats["shared_pages_count"] == 0  # All freed

    def test_no_corruption_on_get(self):
        """Test that get() returns copies, preventing external corruption."""
        cache = KVCache(page_size=10, max_pages=100, enable_prefix_sharing=True)
        
        prefix = [1, 2, 3]
        kv_tokens1 = prefix + [10] * 10
        kv_tokens2 = prefix + [20] * 10
        
        cache.attach(seq_id=1, kv_tokens=kv_tokens1, prefix_tokens=prefix)
        cache.attach(seq_id=2, kv_tokens=kv_tokens2, prefix_tokens=prefix)
        
        # Get data and modify it externally
        result1 = cache.get(seq_id=1)
        result1.append(999)  # Modify the returned list
        
        # Get again - should not be modified (was a copy)
        result1_again = cache.get(seq_id=1)
        assert 999 not in result1_again
        
        # Verify other sequence unaffected
        result2 = cache.get(seq_id=2)
        assert result2[:len(prefix)] == prefix

    def test_prefix_sharing_disabled(self):
        """Test behavior when prefix sharing is disabled."""
        cache = KVCache(page_size=10, max_pages=100, enable_prefix_sharing=False)
        
        prefix = [1, 2, 3]
        kv_tokens1 = prefix + [10] * 10
        kv_tokens2 = prefix + [20] * 10
        
        cache.attach(seq_id=1, kv_tokens=kv_tokens1, prefix_tokens=prefix)
        cache.attach(seq_id=2, kv_tokens=kv_tokens2, prefix_tokens=prefix)
        
        stats = cache.stats()
        assert stats["prefix_shares"] == 0
        assert stats["shared_pages_count"] == 0

    def test_stats(self):
        """Test statistics."""
        cache = KVCache(page_size=512, max_pages=100)
        cache.attach(seq_id=1, kv_tokens=[1, 2, 3] * 100)

        stats = cache.stats()
        assert stats["total_sequences"] == 1
        assert stats["allocated_pages"] > 0
        assert "shared_pages_count" in stats
        assert "total_page_refs" in stats


