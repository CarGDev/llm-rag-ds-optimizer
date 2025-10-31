"""KV cache with paged allocation and prefix sharing."""

import hashlib
from typing import Any, Optional

from llmds.paged_allocator import PagedAllocator


class KVCache:
    """
    KV cache with paged allocation, prefix sharing, and deduplication.

    Supports copy-on-write for prefix sharing and hash-based deduplication
    of repeated system prompts.
    """

    def __init__(
        self,
        page_size: int = 512,
        max_pages: int = 10000,
        enable_prefix_sharing: bool = True,
    ):
        """
        Initialize KV cache.

        Args:
            page_size: Size of each KV cache page in tokens
            max_pages: Maximum number of pages to allocate
            enable_prefix_sharing: Enable prefix sharing optimization
        """
        self.allocator = PagedAllocator(page_size, max_pages)
        self.page_size = page_size
        self._sequences: dict[int, list[int]] = {}  # seq_id -> list[page_ids]
        self._kv_data: dict[int, Any] = {}  # page_id -> KV data
        self._prefix_map: dict[str, list[int]] = {}  # hash -> page_ids
        self._enable_prefix_sharing = enable_prefix_sharing
        self._seq_counter = 0
        self._prefix_shares = 0

    def _hash_prefix(self, prefix: list[int]) -> str:
        """Compute hash of prefix tokens."""
        prefix_str = ",".join(map(str, prefix[:100]))  # Limit length
        return hashlib.sha256(prefix_str.encode()).hexdigest()

    def attach(
        self,
        seq_id: int,
        kv_tokens: list[Any],
        prefix_tokens: Optional[list[int]] = None,
    ) -> None:
        """
        Attach KV cache for a sequence.

        Args:
            seq_id: Sequence identifier
            kv_tokens: KV tokens to cache
            prefix_tokens: Optional prefix tokens for sharing
        """
        if seq_id in self._sequences:
            self.detach(seq_id)

        pages_needed = (len(kv_tokens) + self.page_size - 1) // self.page_size
        page_ids = self.allocator.alloc(pages_needed)

        # Try prefix sharing if enabled
        if self._enable_prefix_sharing and prefix_tokens:
            prefix_hash = self._hash_prefix(prefix_tokens)
            if prefix_hash in self._prefix_map:
                shared_pages = self._prefix_map[prefix_hash]
                # Copy-on-write: reference shared pages
                page_ids = shared_pages + page_ids[len(shared_pages) :]
                self._prefix_shares += 1
            else:
                self._prefix_map[prefix_hash] = page_ids[: len(prefix_tokens) // self.page_size + 1]

        # Store KV data
        for i, page_id in enumerate(page_ids):
            start = i * self.page_size
            end = min(start + self.page_size, len(kv_tokens))
            self._kv_data[page_id] = kv_tokens[start:end]

        self._sequences[seq_id] = page_ids

    def detach(self, seq_id: int) -> None:
        """
        Detach and free KV cache for a sequence.

        Args:
            seq_id: Sequence identifier
        """
        if seq_id not in self._sequences:
            return

        page_ids = self._sequences[seq_id]
        # Free pages (allocator handles shared pages)
        self.allocator.free(page_ids)

        # Remove KV data
        for page_id in page_ids:
            if page_id in self._kv_data:
                del self._kv_data[page_id]

        del self._sequences[seq_id]

    def get(self, seq_id: int) -> Optional[list[Any]]:
        """
        Get KV cache for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            List of KV tokens or None if not found
        """
        if seq_id not in self._sequences:
            return None

        page_ids = self._sequences[seq_id]
        kv_tokens = []
        for page_id in page_ids:
            if page_id in self._kv_data:
                kv_tokens.extend(self._kv_data[page_id])

        return kv_tokens

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        alloc_stats = self.allocator.stats()
        return {
            "total_sequences": len(self._sequences),
            "total_pages": alloc_stats.total_pages,
            "allocated_pages": alloc_stats.allocated_pages,
            "free_pages": alloc_stats.free_pages,
            "prefix_shares": self._prefix_shares,
            "prefix_map_size": len(self._prefix_map),
        }

    def hook_speculative_decode(self, seq_id: int, draft_tokens: list[int]) -> None:
        """
        Hook for speculative decoding compatibility.

        Placeholder API for future implementation.

        Args:
            seq_id: Sequence identifier
            draft_tokens: Draft tokens from speculative decoding
        """
        # Placeholder for speculative decoding integration
        pass

