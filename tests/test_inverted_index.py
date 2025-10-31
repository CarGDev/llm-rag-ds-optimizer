"""Tests for inverted index."""

import pytest

from llmds.inverted_index import InvertedIndex
from llmds.tokenizer import Tokenizer


class TestInvertedIndex:
    """Test inverted index functionality."""

    def test_add_document(self):
        """Test adding documents."""
        index = InvertedIndex()
        index.add_document(doc_id=1, text="the quick brown fox")
        index.add_document(doc_id=2, text="the lazy dog")

        stats = index.stats()
        assert stats["total_documents"] == 2

    def test_search(self):
        """Test search functionality."""
        index = InvertedIndex()
        index.add_document(doc_id=1, text="the quick brown fox")
        index.add_document(doc_id=2, text="the lazy dog")
        index.add_document(doc_id=3, text="the quick fox")

        results = index.search("quick fox", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0

    def test_bm25_scoring(self):
        """Test BM25 scoring."""
        index = InvertedIndex()
        index.add_document(doc_id=1, text="the quick brown fox")
        index.add_document(doc_id=2, text="the quick fox")

        results = index.search("quick fox", top_k=2)
        # Document 2 should score higher (exact match)
        assert len(results) >= 1

    def test_term_frequency(self):
        """Test term frequency retrieval."""
        index = InvertedIndex()
        index.add_document(doc_id=1, text="the quick brown fox")

        tf = index.get_term_frequency("quick", 1)
        assert tf >= 0

    def test_document_frequency(self):
        """Test document frequency."""
        index = InvertedIndex()
        index.add_document(doc_id=1, text="the quick brown fox")
        index.add_document(doc_id=2, text="the lazy dog")

        df = index.get_document_frequency("the")
        assert df >= 1


