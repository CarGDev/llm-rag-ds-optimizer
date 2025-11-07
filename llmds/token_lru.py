"""Token-aware LRU cache with eviction until budget."""

from collections import OrderedDict
from typing import Callable, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TokenLRU(Generic[K, V]):
    """
    Token-aware LRU cache that evicts items until budget is satisfied.

    Evicts least recently used items until the total token count
    fits within the specified budget.
    """

    def __init__(self, token_budget: int, token_of: Callable[[V], int]):
        """
        Initialize token-aware LRU cache.

        Args:
            token_budget: Maximum total tokens allowed
            token_of: Function to extract token count from a value
        """
        self.budget = token_budget
        self.token_of = token_of
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._total_tokens = 0

    def put(self, key: K, value: V) -> None:
        """
        Add or update an item in the cache.

        Evicts LRU items until budget is satisfied.

        Args:
            key: Cache key
            value: Cache value
        """
        token_count = self.token_of(value)

        # If key exists, remove old value first
        if key in self._cache:
            old_value = self._cache[key]
            self._total_tokens -= self.token_of(old_value)
            del self._cache[key]

        # Evict LRU items until we have space
        while self._total_tokens + token_count > self.budget and self._cache:
            self._evict_lru()

        # Add new item
        if self._total_tokens + token_count <= self.budget:
            self._cache[key] = value
            self._total_tokens += token_count
            # Move to end (most recently used)
            self._cache.move_to_end(key)

    def get(self, key: K) -> Optional[V]:
        """
        Get an item from the cache.

        Moves item to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key not in self._cache:
            return None

        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def _evict_lru(self) -> tuple[K, V]:
        """
        Evict the least recently used item.

        Returns:
            Tuple of (key, value) that was evicted
        """
        if not self._cache:
            raise RuntimeError("Cannot evict from empty cache")

        key, value = self._cache.popitem(last=False)
        self._total_tokens -= self.token_of(value)
        return key, value

    def evict_until_budget(self, target_budget: int) -> list[tuple[K, V]]:
        """
        Evict items until total tokens <= target_budget.

        Args:
            target_budget: Target token budget

        Returns:
            List of (key, value) tuples that were evicted
        """
        evicted = []
        while self._total_tokens > target_budget and self._cache:
            evicted.append(self._evict_lru())
        return evicted

    def total_tokens(self) -> int:
        """Get total tokens currently in cache."""
        return self._total_tokens

    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
        self._total_tokens = 0

