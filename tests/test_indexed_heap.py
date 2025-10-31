"""Tests for indexed heap."""

import pytest

from llmds.indexed_heap import IndexedHeap


class TestIndexedHeap:
    """Test indexed heap functionality."""

    def test_basic_push_pop(self):
        """Test basic push and pop operations."""
        heap = IndexedHeap(max_heap=False)  # Min heap

        heap.push(1, 10.0)
        heap.push(2, 5.0)
        heap.push(3, 15.0)

        score, key_id = heap.pop()
        assert score == 5.0
        assert key_id == 2

        score, key_id = heap.pop()
        assert score == 10.0
        assert key_id == 1

    def test_decrease_key(self):
        """Test decrease key operation."""
        heap = IndexedHeap(max_heap=False)  # Min heap

        heap.push(1, 10.0)
        heap.push(2, 5.0)
        heap.push(3, 15.0)

        heap.decrease_key(3, 3.0)  # Decrease 15 to 3

        score, key_id = heap.pop()
        assert score == 3.0
        assert key_id == 3

    def test_increase_key(self):
        """Test increase key operation."""
        heap = IndexedHeap(max_heap=False)  # Min heap

        heap.push(1, 10.0)
        heap.push(2, 5.0)

        heap.increase_key(2, 12.0)  # Increase 5 to 12

        score, key_id = heap.pop()
        assert score == 10.0
        assert key_id == 1

    def test_delete(self):
        """Test delete operation."""
        heap = IndexedHeap(max_heap=False)

        heap.push(1, 10.0)
        heap.push(2, 5.0)
        heap.push(3, 15.0)

        score, key_id = heap.delete(2)
        assert score == 5.0
        assert key_id == 2

        assert not heap.contains(2)
        assert heap.size() == 2

    def test_max_heap(self):
        """Test max heap behavior."""
        heap = IndexedHeap(max_heap=True)

        heap.push(1, 10.0)
        heap.push(2, 5.0)
        heap.push(3, 15.0)

        score, key_id = heap.pop()
        assert score == 15.0
        assert key_id == 3

    def test_duplicate_key_error(self):
        """Test error when pushing duplicate key."""
        heap = IndexedHeap()
        heap.push(1, 10.0)
        with pytest.raises(ValueError):
            heap.push(1, 5.0)

    def test_get_score(self):
        """Test get score operation."""
        heap = IndexedHeap()
        heap.push(1, 10.0)
        heap.push(2, 5.0)

        assert heap.get_score(1) == 10.0
        assert heap.get_score(2) == 5.0
        assert heap.get_score(99) is None

    def test_contains(self):
        """Test contains check."""
        heap = IndexedHeap()
        heap.push(1, 10.0)

        assert heap.contains(1)
        assert not heap.contains(2)


