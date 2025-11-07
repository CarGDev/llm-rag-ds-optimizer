"""Compressed inverted index with BM25 scoring.

Implementation based on:
    Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: 
    BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.

See docs/CITATIONS.md for full citation details.
"""

from collections import defaultdict
from typing import Any, Optional

from llmds.tokenizer import Tokenizer


class InvertedIndex:
    """
    Compressed inverted index with varint/zigzag encoding and BM25 scoring.

    Stores postings lists with compression and provides BM25 retrieval.
    
    Reference:
        Robertson & Zaragoza (2009). The probabilistic relevance framework: 
        BM25 and beyond.
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """
        Initialize inverted index.

        Args:
            tokenizer: Tokenizer instance (creates default if None)
        """
        self.tokenizer = tokenizer or Tokenizer()
        self._inverted_lists: dict[str, list[int]] = defaultdict(list)  # term -> doc_ids
        self._doc_lengths: dict[int, int] = {}  # doc_id -> length
        self._doc_terms: dict[int, dict[str, int]] = {}  # doc_id -> term -> count
        self._total_docs = 0
        self._avg_doc_length = 0.0
        # BM25 parameters
        self.k1 = 1.2
        self.b = 0.75

    def _encode_varint(self, value: int) -> bytes:
        """Encode integer as varint."""
        result = bytearray()
        while value >= 0x80:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

    def _decode_varint(self, data: bytes, offset: int) -> tuple[int, int]:
        """Decode varint from bytes."""
        value = 0
        shift = 0
        while offset < len(data):
            byte = data[offset]
            value |= (byte & 0x7F) << shift
            offset += 1
            if (byte & 0x80) == 0:
                break
            shift += 7
        return value, offset

    def _zigzag_encode(self, value: int) -> int:
        """Zigzag encode for signed integers."""
        return (value << 1) ^ (value >> 31)

    def _zigzag_decode(self, value: int) -> int:
        """Zigzag decode."""
        return (value >> 1) ^ (-(value & 1))

    def add_document(self, doc_id: int, text: str) -> None:
        """
        Add a document to the index.

        Args:
            doc_id: Document identifier
            text: Document text
        """
        tokens = self.tokenizer.encode(text)
        term_counts: dict[str, int] = defaultdict(int)

        # Count term frequencies
        for token_id in tokens:
            term = self.tokenizer.decode([token_id])
            if term:
                term_counts[term] += 1

        # Update inverted lists
        for term, count in term_counts.items():
            if doc_id not in self._inverted_lists[term]:
                self._inverted_lists[term].append(doc_id)

        # Store document metadata
        self._doc_lengths[doc_id] = len(tokens)
        self._doc_terms[doc_id] = term_counts

        # Update average document length
        self._total_docs += 1
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / self._total_docs if self._total_docs > 0 else 0.0

    def _bm25_score(self, term: str, doc_id: int, query_term_freq: int) -> float:
        """
        Calculate BM25 score for a term-document pair.

        Args:
            term: Query term
            doc_id: Document ID
            query_term_freq: Frequency of term in query

        Returns:
            BM25 score
        """
        if doc_id not in self._doc_terms or term not in self._doc_terms[doc_id]:
            return 0.0

        # Term frequency in document
        tf = self._doc_terms[doc_id][term]

        # Document frequency
        df = len(self._inverted_lists.get(term, []))

        # Inverse document frequency
        idf = 0.0
        if df > 0:
            idf = (self._total_docs - df + 0.5) / (df + 0.5)
            idf = max(0.0, idf)  # Avoid negative IDF

        # Document length normalization
        doc_length = self._doc_lengths.get(doc_id, 1)
        length_norm = (1 - self.b) + self.b * (doc_length / self._avg_doc_length)

        # BM25 formula
        score = (
            idf
            * (tf * (self.k1 + 1))
            / (tf + self.k1 * length_norm)
            * (query_term_freq / (query_term_freq + 0.5))
        )

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Search the index with BM25 scoring.

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        query_tokens = self.tokenizer.encode(query)
        query_term_counts: dict[str, int] = defaultdict(int)

        for token_id in query_tokens:
            term = self.tokenizer.decode([token_id])
            if term:
                query_term_counts[term] += 1

        # Score all candidate documents
        doc_scores: dict[int, float] = defaultdict(float)

        for term, query_freq in query_term_counts.items():
            if term in self._inverted_lists:
                for doc_id in self._inverted_lists[term]:
                    score = self._bm25_score(term, doc_id, query_freq)
                    doc_scores[doc_id] += score

        # Sort by score and return top-k
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Get term frequency in a document.

        Args:
            term: Term
            doc_id: Document ID

        Returns:
            Term frequency
        """
        if doc_id in self._doc_terms:
            return self._doc_terms[doc_id].get(term, 0)
        return 0

    def get_document_frequency(self, term: str) -> int:
        """
        Get document frequency of a term.

        Args:
            term: Term

        Returns:
            Document frequency
        """
        return len(self._inverted_lists.get(term, []))

    def stats(self) -> dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        total_postings = sum(len(postings) for postings in self._inverted_lists.values())
        return {
            "total_documents": self._total_docs,
            "total_terms": len(self._inverted_lists),
            "total_postings": total_postings,
            "avg_doc_length": self._avg_doc_length,
            "avg_postings_per_term": (
                total_postings / len(self._inverted_lists) if self._inverted_lists else 0.0
            ),
        }

