"""Tests for Count-Min Sketch."""

import pytest

from llmds.cmsketch import CountMinSketch


class TestCountMinSketch:
    """Test Count-Min Sketch functionality."""

    def test_add_and_estimate(self):
        """Test adding items and estimating frequency."""
        cms = CountMinSketch(width=1024, depth=4)

        cms.add("item1")
        cms.add("item1")
        cms.add("item2")

        assert cms.estimate("item1") >= 2
        assert cms.estimate("item2") >= 1

    def test_error_bound(self):
        """Test error bound calculation."""
        cms = CountMinSketch(width=1024, depth=4)

        for i in range(100):
            cms.add(f"item{i % 10}")

        error_bound = cms.get_error_bound()
        assert error_bound >= 0

    def test_is_hot(self):
        """Test hot item detection."""
        cms = CountMinSketch(width=1024, depth=4)

        for _ in range(10):
            cms.add("hot_item")

        assert cms.is_hot("hot_item", threshold=5) is True
        assert cms.is_hot("cold_item", threshold=5) is False

    def test_reset(self):
        """Test reset functionality."""
        cms = CountMinSketch(width=1024, depth=4)
        cms.add("item1")
        assert cms.get_total_count() > 0

        cms.reset()
        assert cms.get_total_count() == 0
        assert cms.estimate("item1") == 0


