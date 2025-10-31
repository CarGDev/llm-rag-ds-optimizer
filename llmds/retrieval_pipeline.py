"""Retrieval pipeline combining ANN, lexical search, and fusion."""

from typing import Optional

import numpy as np

from llmds.cmsketch import CountMinSketch
from llmds.hnsw import HNSW
from llmds.indexed_heap import IndexedHeap
from llmds.inverted_index import InvertedIndex
from llmds.token_lru import TokenLRU
from llmds.tokenizer import Tokenizer


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline combining ANN, lexical search, and fusion.

    Combines HNSW for dense embeddings, inverted index for BM25,
    and score fusion with top-K maintenance using indexed heap.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        token_budget: int = 100000,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            embedding_dim: Dimension of embedding vectors
            hnsw_M: HNSW M parameter
            hnsw_ef_construction: HNSW efConstruction parameter
            hnsw_ef_search: HNSW efSearch parameter
            token_budget: Token budget for cache
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.hnsw = HNSW(
            dim=embedding_dim,
            M=hnsw_M,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
        )
        self.inverted_index = InvertedIndex(tokenizer=self.tokenizer)
        self.cmsketch = CountMinSketch(width=2048, depth=4)
        self.token_cache = TokenLRU(
            token_budget=token_budget,
            token_of=lambda text: self.tokenizer.count_tokens(text),
        )

    def add_document(
        self,
        doc_id: int,
        text: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a document to both indices.

        Args:
            doc_id: Document identifier
            text: Document text
            embedding: Optional embedding vector (if None, generates random)
        """
        # Add to inverted index
        self.inverted_index.add_document(doc_id, text)

        # Add to HNSW if embedding provided
        if embedding is not None:
            if embedding.shape != (self.hnsw.dim,):
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.hnsw.dim}, "
                    f"got {embedding.shape[0]}"
                )
            self.hnsw.add(embedding, doc_id)
        else:
            # Generate random embedding for testing
            random_embedding = np.random.randn(self.hnsw.dim).astype(np.float32)
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            self.hnsw.add(random_embedding, doc_id)

    def search(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        fusion_weight: float = 0.5,
    ) -> list[tuple[int, float]]:
        """
        Search with hybrid retrieval and score fusion.

        Args:
            query: Query text
            query_embedding: Optional query embedding vector
            top_k: Number of results to return
            fusion_weight: Weight for dense search (1-fusion_weight for BM25)

        Returns:
            List of (doc_id, fused_score) tuples
        """
        # Check cache
        cached = self.token_cache.get(query)
        if cached:
            self.cmsketch.add(query)
            # Parse cached string back to list of tuples
            import ast
            try:
                return ast.literal_eval(cached)
            except:
                return results  # Return computed results if parsing fails

        # BM25 search
        bm25_results = self.inverted_index.search(query, top_k=top_k * 2)

        # Dense search (if embedding provided)
        dense_results = []
        if query_embedding is not None:
            dense_results = self.hnsw.search(query_embedding, k=top_k * 2)

        # Normalize scores
        bm25_scores: dict[int, float] = {doc_id: score for doc_id, score in bm25_results}
        dense_scores: dict[int, float] = {}

        if dense_results:
            max_dense = max(dist for _, dist in dense_results) if dense_results else 1.0
            min_dense = min(dist for _, dist in dense_results) if dense_results else 0.0
            dense_range = max_dense - min_dense if max_dense > min_dense else 1.0

            for doc_id, dist in dense_results:  # HNSW.search returns (node_id, distance)
                # Convert distance to similarity (inverse)
                normalized = 1.0 - (dist - min_dense) / dense_range if dense_range > 0 else 1.0
                dense_scores[doc_id] = normalized

        # Normalize BM25 scores
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            min_bm25 = min(bm25_scores.values())
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

            for doc_id in bm25_scores:
                bm25_scores[doc_id] = (
                    (bm25_scores[doc_id] - min_bm25) / bm25_range if bm25_range > 0 else 1.0
                )

        # Fuse scores using indexed heap
        fused_scores: dict[int, float] = {}
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)

            # Weighted fusion
            fused_score = fusion_weight * dense_score + (1 - fusion_weight) * bm25_score
            fused_scores[doc_id] = fused_score

        # Top-K using indexed heap
        heap = IndexedHeap(max_heap=True)
        for doc_id, score in fused_scores.items():
            if heap.size() < top_k:
                heap.push(doc_id, score)
            else:
                min_score, _ = heap.peek()
                if min_score is not None and score > min_score:
                    heap.pop()
                    heap.push(doc_id, score)

        # Extract results
        results = []
        while not heap.is_empty():
            score, doc_id = heap.pop()
            results.append((doc_id, score))

        results.reverse()  # Highest score first

        # Cache results (store as string representation for token counting)
        results_str = str(results)
        self.token_cache.put(query, results_str)
        self.cmsketch.add(query)

        return results

    def stats(self) -> dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        hnsw_stats = self.hnsw.stats()
        index_stats = self.inverted_index.stats()

        return {
            "hnsw": hnsw_stats,
            "inverted_index": index_stats,
            "cmsketch_total_count": self.cmsketch.get_total_count(),
            "cache_size": self.token_cache.size(),
            "cache_tokens": self.token_cache.total_tokens(),
        }

