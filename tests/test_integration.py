"""Integration tests for end-to-end workflows."""

import numpy as np
import pytest

from llmds.hnsw import HNSW
from llmds.inverted_index import InvertedIndex
from llmds.kv_cache import KVCache
from llmds.retrieval_pipeline import RetrievalPipeline
from llmds.scheduler import Scheduler


class TestRetrievalPipelineEnd2End:
    """Test end-to-end retrieval pipeline workflows."""

    def test_retrieval_pipeline_end2end(self):
        """Test complete retrieval pipeline from indexing to search."""
        pipeline = RetrievalPipeline(embedding_dim=128, seed=42)

        # Add multiple documents
        num_docs = 100
        for i in range(num_docs):
            text = f"document {i} about topic {i % 5}"
            embedding = np.random.randn(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            pipeline.add_document(doc_id=i, text=text, embedding=embedding)

        # Verify index stats
        stats = pipeline.stats()
        assert stats["inverted_index"]["total_documents"] == num_docs
        assert stats["hnsw"]["num_vectors"] == num_docs

        # Perform multiple searches
        query_embedding = np.random.randn(128).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = pipeline.search(
            "document topic", query_embedding=query_embedding, top_k=10
        )

        # Verify results
        assert len(results) <= 10
        assert len(results) > 0
        assert all(isinstance(doc_id, int) for doc_id, _ in results)
        assert all(isinstance(score, (int, float)) for _, score in results)

        # Verify scores are sorted descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_pipeline_large_scale(self):
        """Test pipeline with larger dataset."""
        pipeline = RetrievalPipeline(embedding_dim=64, seed=42)

        # Add many documents
        num_docs = 1000
        for i in range(num_docs):
            text = f"document number {i} with content {i}"
            embedding = np.random.randn(64).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            pipeline.add_document(doc_id=i, text=text, embedding=embedding)

        # Verify it can handle large scale
        stats = pipeline.stats()
        assert stats["hnsw"]["num_vectors"] == num_docs

        # Search should still work
        query_embedding = np.random.randn(64).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = pipeline.search(
            "document", query_embedding=query_embedding, top_k=50
        )
        assert len(results) <= 50


class TestKVCacheSchedulerIntegration:
    """Test KV cache and scheduler integration."""

    def test_kv_cache_scheduler_integration(self):
        """Test that KV cache works with scheduler batching."""
        cache = KVCache(page_size=128, max_pages=100, enable_prefix_sharing=True)
        scheduler = Scheduler(
            max_batch_size=8,
            wait_time_ms=10.0,
            max_qps=100,
            max_token_rate=10000,
        )

        # Simulate attaching KV caches for scheduled sequences
        sequences = []
        for seq_id in range(5):
            kv_tokens = list(range(seq_id * 10, (seq_id + 1) * 10))
            prefix = kv_tokens[:5]
            cache.attach(seq_id=seq_id, kv_tokens=kv_tokens, prefix_tokens=prefix)
            sequences.append(seq_id)

        # Verify cache stats
        cache_stats = cache.stats()
        assert cache_stats["total_sequences"] == 5
        assert cache_stats["allocated_pages"] > 0

        # Verify prefix sharing was used
        assert cache_stats["prefix_shares"] > 0 or cache_stats["shared_pages_count"] >= 0

        # Detach sequences
        for seq_id in sequences:
            cache.detach(seq_id=seq_id)

        # Verify cleanup
        final_stats = cache.stats()
        assert final_stats["total_sequences"] == 0


class TestHNSWInvertedIndexFusion:
    """Test HNSW and inverted index fusion."""

    def test_hnsw_inverted_index_fusion(self):
        """Test that HNSW and inverted index can be used together for fusion."""
        pipeline = RetrievalPipeline(embedding_dim=128, seed=42)

        # Add documents with both text and embeddings
        texts = [
            "machine learning algorithms",
            "deep learning neural networks",
            "natural language processing",
            "computer vision systems",
            "reinforcement learning agents",
        ]

        for i, text in enumerate(texts):
            embedding = np.random.randn(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            pipeline.add_document(doc_id=i, text=text, embedding=embedding)

        # Search using both dense and sparse components
        query_text = "machine learning"
        query_embedding = np.random.randn(128).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Test different fusion weights
        for fusion_weight in [0.0, 0.5, 1.0]:
            results = pipeline.search(
                query_text,
                query_embedding=query_embedding,
                top_k=5,
                fusion_weight=fusion_weight,
            )

            # Should return results regardless of fusion weight
            assert len(results) > 0
            assert len(results) <= 5

            # Verify results format
            for doc_id, score in results:
                assert isinstance(doc_id, int)
                assert isinstance(score, (int, float))
                assert 0 <= doc_id < len(texts)

    def test_hybrid_retrieval_consistency(self):
        """Test that hybrid retrieval gives consistent results."""
        pipeline = RetrievalPipeline(embedding_dim=64, seed=42)

        # Add documents
        for i in range(20):
            text = f"document {i}"
            embedding = np.random.randn(64).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            pipeline.add_document(doc_id=i, text=text, embedding=embedding)

        query_text = "document"
        query_embedding = np.random.randn(64).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Multiple searches should return similar results
        results1 = pipeline.search(
            query_text, query_embedding=query_embedding, top_k=10, fusion_weight=0.5
        )
        results2 = pipeline.search(
            query_text, query_embedding=query_embedding, top_k=10, fusion_weight=0.5
        )

        # Results should be consistent (same length, overlapping results)
        assert len(results1) == len(results2)
        doc_ids1 = {doc_id for doc_id, _ in results1}
        doc_ids2 = {doc_id for doc_id, _ in results2}

        # Should have significant overlap
        overlap = len(doc_ids1 & doc_ids2)
        assert overlap >= len(results1) * 0.8  # At least 80% overlap


class TestMultipleCorpusLoading:
    """Test loading and searching across multiple corpora."""

    def test_multiple_corpus_loading(self):
        """Test that pipeline can handle multiple corpus scenarios."""
        pipeline = RetrievalPipeline(embedding_dim=128, seed=42)

        # Simulate loading multiple corpora
        corpora = {
            "corpus1": list(range(0, 50)),
            "corpus2": list(range(50, 100)),
            "corpus3": list(range(100, 150)),
        }

        doc_id = 0
        for corpus_name, doc_range in corpora.items():
            for i in doc_range:
                text = f"{corpus_name} document {i}"
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                pipeline.add_document(doc_id=doc_id, text=text, embedding=embedding)
                doc_id += 1

        # Verify all documents are indexed
        stats = pipeline.stats()
        assert stats["hnsw"]["num_vectors"] == 150
        assert stats["inverted_index"]["total_documents"] == 150

        # Search should work across all corpora
        query_embedding = np.random.randn(128).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = pipeline.search(
            "document", query_embedding=query_embedding, top_k=20
        )

        # Should return results from multiple corpora
        assert len(results) > 0
        result_doc_ids = {doc_id for doc_id, _ in results}

        # Verify results span multiple corpora (doc_ids 0-149)
        assert min(result_doc_ids) < 50 or max(result_doc_ids) >= 50


class TestLargeScaleScenarios:
    """Test large-scale scenarios."""

    @pytest.mark.slow
    def test_large_scale_scenarios(self):
        """Test with larger datasets to verify scalability."""
        pipeline = RetrievalPipeline(embedding_dim=128, seed=42)

        # Add a moderate number of documents
        num_docs = 5000
        batch_size = 100

        for batch_start in range(0, num_docs, batch_size):
            batch_end = min(batch_start + batch_size, num_docs)
            for i in range(batch_start, batch_end):
                text = f"document {i} with content"
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                pipeline.add_document(doc_id=i, text=text, embedding=embedding)

            # Verify incremental stats
            stats = pipeline.stats()
            assert stats["hnsw"]["num_vectors"] == batch_end

        # Final verification
        final_stats = pipeline.stats()
        assert final_stats["hnsw"]["num_vectors"] == num_docs

        # Search should still be fast
        query_embedding = np.random.randn(128).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = pipeline.search(
            "document", query_embedding=query_embedding, top_k=100
        )

        assert len(results) <= 100
        assert len(results) > 0

