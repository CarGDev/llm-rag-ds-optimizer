"""Count-Min Sketch for hot query estimation and cache priming."""

import mmh3
from typing import Optional


class CountMinSketch:
    """
    Count-Min Sketch for frequency estimation with conservative update.

    Uses 4 hash functions (via MurmurHash3) and provides error bounds.
    """

    def __init__(self, width: int = 2048, depth: int = 4):
        """
        Initialize Count-Min Sketch.

        Args:
            width: Width of the sketch (number of counters per row)
            depth: Depth of the sketch (number of hash functions)
        """
        self.width = width
        self.depth = depth
        self._table: list[list[int]] = [[0] * width for _ in range(depth)]
        self._total_count = 0

    def _hash(self, item: str, seed: int) -> int:
        """Hash an item with a given seed."""
        return mmh3.hash(item, seed) % self.width

    def add(self, item: str, count: int = 1) -> None:
        """
        Add an item to the sketch.

        Args:
            item: Item to add
            count: Count to add (default 1)
        """
        self._total_count += count
        min_val = float("inf")

        # Find minimum count across all rows
        for i in range(self.depth):
            idx = self._hash(item, i)
            self._table[i][idx] += count
            min_val = min(min_val, self._table[i][idx])

        # Conservative update: only increment if current count < min
        # This reduces overestimation bias
        for i in range(self.depth):
            idx = self._hash(item, i)
            if self._table[i][idx] > min_val:
                self._table[i][idx] = min_val

    def estimate(self, item: str) -> int:
        """
        Estimate the frequency of an item.

        Args:
            item: Item to estimate

        Returns:
            Estimated frequency (minimum across all rows)
        """
        min_count = float("inf")
        for i in range(self.depth):
            idx = self._hash(item, i)
            min_count = min(min_count, self._table[i][idx])
        return int(min_count)

    def get_error_bound(self) -> float:
        """
        Get theoretical error bound (with high probability).

        Returns:
            Error bound as a fraction of total count
        """
        # With probability 1 - delta, error <= epsilon * total_count
        # where epsilon = e / width and delta = (1/2)^depth
        epsilon = 2.71828 / self.width
        return epsilon * self._total_count

    def get_total_count(self) -> int:
        """Get total count of all items."""
        return self._total_count

    def is_hot(self, item: str, threshold: int) -> bool:
        """
        Check if an item is "hot" (above threshold).

        Args:
            item: Item to check
            threshold: Frequency threshold

        Returns:
            True if estimated frequency >= threshold
        """
        return self.estimate(item) >= threshold

    def reset(self) -> None:
        """Reset all counters."""
        self._table = [[0] * self.width for _ in range(self.depth)]
        self._total_count = 0

