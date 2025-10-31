"""Tests for HNSW."""

import numpy as np
import pytest

from llmds.hnsw import HNSW


class TestHNSW:
    """Test HNSW functionality."""

    def test_add_and_search(self):
        """Test adding vectors and searching."""
        hnsw = HNSW(dim=10, M=4, ef_construction=10, ef_search=5)

        # Add some vectors
        for i in range(20):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            hnsw.add(vec, i)

        # Search
        query = np.random.randn(10).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = hnsw.search(query, k=5)
        assert len(results) <= 5
        assert len(results) > 0

    def test_dimension_mismatch(self):
        """Test dimension mismatch error."""
        hnsw = HNSW(dim=10)
        vec = np.random.randn(5).astype(np.float32)

        with pytest.raises(ValueError):
            hnsw.add(vec, 1)

    def test_duplicate_id(self):
        """Test duplicate ID error."""
        hnsw = HNSW(dim=10)
        vec = np.random.randn(10).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        hnsw.add(vec, 1)
        with pytest.raises(ValueError):
            hnsw.add(vec, 1)

    def test_stats(self):
        """Test statistics."""
        hnsw = HNSW(dim=10, M=4)
        vec = np.random.randn(10).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        hnsw.add(vec, 1)

        stats = hnsw.stats()
        assert stats["num_vectors"] == 1
        assert stats["num_layers"] > 0


