"""Indexed binary heap with decrease/increase-key operations."""

from typing import Optional


class IndexedHeap:
    """
    Indexed binary heap supporting O(log n) decrease/increase-key operations.

    Maintains a heap of (score, id) pairs with an index map for O(1) lookup
    and O(log n) updates.
    """

    def __init__(self, max_heap: bool = False):
        """
        Initialize indexed heap.

        Args:
            max_heap: If True, use max-heap (largest score at top),
                     otherwise min-heap (smallest score at top)
        """
        self._heap: list[tuple[float, int]] = []  # (score, id)
        self._pos: dict[int, int] = {}  # id -> index in heap
        self._max_heap = max_heap

    def _compare(self, a: float, b: float) -> bool:
        """Compare two scores based on heap type."""
        if self._max_heap:
            return a > b
        return a < b

    def _swap(self, i: int, j: int) -> None:
        """Swap elements at indices i and j, updating position map."""
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        _, id_i = self._heap[i]
        _, id_j = self._heap[j]
        self._pos[id_i] = i
        self._pos[id_j] = j

    def _bubble_up(self, idx: int) -> None:
        """Bubble up element at idx to maintain heap property."""
        while idx > 0:
            parent = (idx - 1) // 2
            score_curr, _ = self._heap[idx]
            score_parent, _ = self._heap[parent]

            if self._compare(score_curr, score_parent):
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _bubble_down(self, idx: int) -> None:
        """Bubble down element at idx to maintain heap property."""
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            best = idx

            if left < len(self._heap):
                score_best, _ = self._heap[best]
                score_left, _ = self._heap[left]
                if self._compare(score_left, score_best):
                    best = left

            if right < len(self._heap):
                score_best, _ = self._heap[best]
                score_right, _ = self._heap[right]
                if self._compare(score_right, score_best):
                    best = right

            if best != idx:
                self._swap(idx, best)
                idx = best
            else:
                break

    def push(self, key_id: int, score: float) -> None:
        """
        Push an item onto the heap.

        Args:
            key_id: Unique identifier for the item
            score: Score/priority value
        """
        if key_id in self._pos:
            raise ValueError(f"Key {key_id} already exists in heap")

        idx = len(self._heap)
        self._heap.append((score, key_id))
        self._pos[key_id] = idx
        self._bubble_up(idx)

    def pop(self) -> tuple[float, int]:
        """
        Pop the top element from the heap.

        Returns:
            Tuple of (score, id)

        Raises:
            IndexError: If heap is empty
        """
        if not self._heap:
            raise IndexError("Cannot pop from empty heap")

        if len(self._heap) == 1:
            score, key_id = self._heap.pop()
            del self._pos[key_id]
            return score, key_id

        # Swap root with last element
        self._swap(0, len(self._heap) - 1)
        score, key_id = self._heap.pop()
        del self._pos[key_id]

        if self._heap:
            self._bubble_down(0)

        return score, key_id

    def decrease_key(self, key_id: int, new_score: float) -> None:
        """
        Decrease the key value for an item.

        For min-heap: new_score must be < old_score (bubble up).
        For max-heap: new_score must be < old_score (bubble down).

        Args:
            key_id: Item identifier
            new_score: New score value

        Raises:
            KeyError: If key_id not found
            ValueError: If new_score doesn't satisfy heap property
        """
        if key_id not in self._pos:
            raise KeyError(f"Key {key_id} not found in heap")

        idx = self._pos[key_id]
        old_score, _ = self._heap[idx]

        # Validate direction - both heap types decrease when new < old
        if new_score >= old_score:
            heap_type = "max-heap" if self._max_heap else "min-heap"
            raise ValueError(f"For {heap_type}, new_score must be < old_score")

        self._heap[idx] = (new_score, key_id)
        
        # Bubble direction depends on heap type
        if self._max_heap:
            # Max-heap: decreasing score means lower priority -> bubble down
            self._bubble_down(idx)
        else:
            # Min-heap: decreasing score means higher priority -> bubble up
            self._bubble_up(idx)

    def increase_key(self, key_id: int, new_score: float) -> None:
        """
        Increase the key value for an item.

        For min-heap: new_score must be > old_score (bubble down).
        For max-heap: new_score must be > old_score (bubble up).

        Args:
            key_id: Item identifier
            new_score: New score value

        Raises:
            KeyError: If key_id not found
            ValueError: If new_score doesn't satisfy heap property
        """
        if key_id not in self._pos:
            raise KeyError(f"Key {key_id} not found in heap")

        idx = self._pos[key_id]
        old_score, _ = self._heap[idx]

        # Validate direction - both heap types increase when new > old
        if new_score <= old_score:
            heap_type = "max-heap" if self._max_heap else "min-heap"
            raise ValueError(f"For {heap_type}, new_score must be > old_score")

        self._heap[idx] = (new_score, key_id)
        
        # Bubble direction depends on heap type
        if self._max_heap:
            # Max-heap: increasing score means higher priority -> bubble up
            self._bubble_up(idx)
        else:
            # Min-heap: increasing score means lower priority -> bubble down
            self._bubble_down(idx)

    def delete(self, key_id: int) -> tuple[float, int]:
        """
        Delete an item from the heap.

        Args:
            key_id: Item identifier

        Returns:
            Tuple of (score, id) that was deleted

        Raises:
            KeyError: If key_id not found
        """
        if key_id not in self._pos:
            raise KeyError(f"Key {key_id} not found in heap")

        idx = self._pos[key_id]
        score, _ = self._heap[idx]

        # Swap with last element
        self._swap(idx, len(self._heap) - 1)
        self._heap.pop()
        del self._pos[key_id]

        # Restore heap property
        if idx < len(self._heap):
            # Try bubbling up first (might be smaller/bigger than parent)
            parent = (idx - 1) // 2
            if idx > 0:
                score_curr, _ = self._heap[idx]
                score_parent, _ = self._heap[parent]
                if self._compare(score_curr, score_parent):
                    self._bubble_up(idx)
                    return score, key_id

            # Otherwise bubble down
            self._bubble_down(idx)

        return score, key_id

    def peek(self) -> Optional[tuple[float, int]]:
        """
        Peek at the top element without removing it.

        Returns:
            Tuple of (score, id) or None if empty
        """
        if not self._heap:
            return None
        return self._heap[0]

    def get_score(self, key_id: int) -> Optional[float]:
        """
        Get the score for a given key_id.

        Args:
            key_id: Item identifier

        Returns:
            Score value or None if not found
        """
        if key_id not in self._pos:
            return None
        idx = self._pos[key_id]
        score, _ = self._heap[idx]
        return score

    def size(self) -> int:
        """Get the number of elements in the heap."""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Check if heap is empty."""
        return len(self._heap) == 0

    def contains(self, key_id: int) -> bool:
        """Check if key_id exists in heap."""
        return key_id in self._pos

