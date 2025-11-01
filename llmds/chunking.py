"""Text chunking utilities for RAG."""

from typing import Any, Iterator, Optional


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    tokenizer: Optional[Any] = None,
) -> Iterator[str]:
    """
    Chunk text into overlapping segments.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens/characters
        overlap: Overlap between chunks
        tokenizer: Optional tokenizer (if None, uses character-based)

    Yields:
        Text chunks
    """
    if tokenizer is not None:
        # Token-based chunking
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            yield tokenizer.decode(chunk_tokens)
    else:
        # Character-based chunking (simple fallback)
        for i in range(0, len(text), chunk_size - overlap):
            yield text[i:i + chunk_size]


def chunk_documents(
    documents: Iterator[dict[str, Any]],
    chunk_size: int = 512,
    overlap: int = 50,
    tokenizer: Optional[Any] = None,
) -> Iterator[dict[str, Any]]:
    """
    Chunk documents into smaller segments.

    Args:
        documents: Iterator of document dicts with 'id', 'text', 'meta'
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        tokenizer: Optional tokenizer

    Yields:
        Chunk dictionaries with 'id', 'text', 'meta', 'chunk_idx'
    """
    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]
        meta = doc.get("meta", {})
        
        chunks = list(chunk_text(text, chunk_size, overlap, tokenizer))
        
        for chunk_idx, chunk_text_seg in enumerate(chunks):
            yield {
                "id": f"{doc_id}_chunk_{chunk_idx}",
                "text": chunk_text_seg,
                "meta": {
                    **meta,
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                }
            }

