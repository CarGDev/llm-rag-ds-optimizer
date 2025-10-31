"""Tests for token LRU cache."""

import pytest

from llmds.token_lru import TokenLRU


class TestTokenLRU:
    """Test token-aware LRU cache."""

    def test_basic_put_get(self):
        """Test basic put and get operations."""
        cache = TokenLRU(token_budget=100, token_of=lambda x: len(str(x)))

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.total_tokens() == 6  # "value1" has 6 chars

    def test_eviction(self):
        """Test LRU eviction when budget exceeded."""
        cache = TokenLRU(token_budget=10, token_of=lambda x: len(str(x)))

        cache.put("key1", "value1")  # 6 tokens
        cache.put("key2", "value2")  # 6 tokens, total 12, evicts key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.total_tokens() == 6

    def test_lru_order(self):
        """Test LRU ordering."""
        cache = TokenLRU(token_budget=20, token_of=lambda x: len(str(x)))

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Access key1, making it MRU

        cache.put("key3", "value3")  # Should evict key2 (LRU)

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_evict_until_budget(self):
        """Test evict until budget functionality."""
        cache = TokenLRU(token_budget=30, token_of=lambda x: len(str(x)))

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        evicted = cache.evict_until_budget(10)
        assert len(evicted) >= 2  # Should evict at least 2 items
        assert cache.total_tokens() <= 10

    def test_update_existing(self):
        """Test updating existing key."""
        cache = TokenLRU(token_budget=20, token_of=lambda x: len(str(x)))

        cache.put("key1", "value1")
        cache.put("key1", "value123")  # Update with larger value

        assert cache.get("key1") == "value123"
        assert cache.total_tokens() == 8  # "value123" has 8 chars


