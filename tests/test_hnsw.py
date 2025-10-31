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

    def test_hnsw_search_returns_results(self):
        """Regression test: Verify search returns results in correct format (node_id, distance)."""
        hnsw = HNSW(dim=10, M=4, ef_construction=10, ef_search=5)

        # Add multiple vectors
        vectors = []
        for i in range(20):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            hnsw.add(vec, i)

        # Search for nearest neighbors
        query = np.random.randn(10).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = hnsw.search(query, k=5)

        # Verify results are not empty
        assert len(results) > 0, "Search should return results"
        assert len(results) <= 5, "Should return at most k results"

        # Verify format: (node_id, distance)
        for node_id, distance in results:
            assert isinstance(node_id, (int, np.integer)), f"First element should be node_id (int), got {type(node_id)}"
            assert isinstance(distance, (float, np.floating)), f"Second element should be distance (float), got {type(distance)}"
            assert 0 <= node_id < 20, f"Node ID should be valid: {node_id}"
            assert distance >= 0, f"Distance should be non-negative: {distance}"

        # Verify results are sorted by distance (ascending)
        distances = [dist for _, dist in results]
        assert distances == sorted(distances), "Results should be sorted by distance"

        # Verify all returned node IDs are valid
        node_ids = [node_id for node_id, _ in results]
        assert all(0 <= nid < 20 for nid in node_ids), "All node IDs should be valid"


