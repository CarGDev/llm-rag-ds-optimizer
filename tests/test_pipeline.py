"""Tests for retrieval pipeline."""

import numpy as np
import pytest

from llmds.retrieval_pipeline import RetrievalPipeline


class TestRetrievalPipeline:
    """Test retrieval pipeline functionality."""

    def test_add_document(self):
        """Test adding documents."""
        pipeline = RetrievalPipeline(embedding_dim=10)

        embedding = np.random.randn(10).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        pipeline.add_document(doc_id=1, text="the quick brown fox", embedding=embedding)

        stats = pipeline.stats()
        assert stats["inverted_index"]["total_documents"] == 1
        assert stats["hnsw"]["num_vectors"] == 1

    def test_search(self):
        """Test search functionality."""
        pipeline = RetrievalPipeline(embedding_dim=10)

        # Add documents
        for i in range(5):
            embedding = np.random.randn(10).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            pipeline.add_document(
                doc_id=i, text=f"document {i} with some text", embedding=embedding
            )

        # Search
        query_embedding = np.random.randn(10).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = pipeline.search("document", query_embedding=query_embedding, top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_cache(self):
        """Test result caching."""
        pipeline = RetrievalPipeline(embedding_dim=10)

        embedding = np.random.randn(10).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        pipeline.add_document(doc_id=1, text="test document", embedding=embedding)

        query_embedding = np.random.randn(10).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results1 = pipeline.search("test", query_embedding=query_embedding, top_k=1)
        results2 = pipeline.search("test", query_embedding=query_embedding, top_k=1)

        # Second search should use cache
        assert len(results1) == len(results2)


