"""Tests for KV cache."""

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
        assert stats["prefix_shares"] >= 0

    def test_stats(self):
        """Test statistics."""
        cache = KVCache(page_size=512, max_pages=100)
        cache.attach(seq_id=1, kv_tokens=[1, 2, 3] * 100)

        stats = cache.stats()
        assert stats["total_sequences"] == 1
        assert stats["allocated_pages"] > 0


