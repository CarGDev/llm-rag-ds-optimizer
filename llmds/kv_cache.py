"""KV cache with paged allocation and prefix sharing.

Implementation based on techniques from:
    Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation.

See docs/CITATIONS.md for full citation details.
"""

import copy
import hashlib
from typing import Any, Optional

from llmds.paged_allocator import PagedAllocator


class KVCache:
    """
    KV cache with paged allocation, prefix sharing, and deduplication.

    Implements copy-on-write (COW) for prefix sharing: shared pages are
    read-only until a write occurs, at which point they are copied.
    
    Reference:
        Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation.

    **Copy-on-Write Semantics:**
    - Shared pages (from prefix sharing) are read-only
    - Attempts to modify shared pages trigger lazy copying
    - Each sequence maintains its own copy of modified pages
    - Original shared pages remain unchanged for other sequences

    Supports hash-based deduplication of repeated system prompts.
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
        self._page_refs: dict[int, int] = {}  # page_id -> reference count
        self._shared_pages: set[int] = set()  # page_ids that are shared (read-only)
        self._enable_prefix_sharing = enable_prefix_sharing
        self._seq_counter = 0
        self._prefix_shares = 0

    def _hash_prefix(self, prefix: list[int]) -> str:
        """Compute hash of prefix tokens."""
        prefix_str = ",".join(map(str, prefix[:100]))  # Limit length
        return hashlib.sha256(prefix_str.encode()).hexdigest()

    def _copy_if_shared(self, page_id: int, seq_id: int) -> int:
        """
        Copy-on-write: if page is shared, create a new copy.

        Args:
            page_id: Original page ID (may be shared)
            seq_id: Sequence ID requesting the copy

        Returns:
            New page_id if copied, original page_id if not shared
        """
        if page_id not in self._shared_pages:
            return page_id

        # Page is shared - need to copy
        new_page_id = self.allocator.alloc(1)[0]
        
        # Copy the data
        if page_id in self._kv_data:
            self._kv_data[new_page_id] = copy.deepcopy(self._kv_data[page_id])
        else:
            # Empty page
            self._kv_data[new_page_id] = []
        
        # Decrement reference count of original
        self._page_refs[page_id] = self._page_refs.get(page_id, 1) - 1
        if self._page_refs[page_id] <= 0:
            self._shared_pages.discard(page_id)
            if page_id in self._page_refs:
                del self._page_refs[page_id]
        
        # New page is not shared (single owner)
        self._page_refs[new_page_id] = 1
        
        return new_page_id

    def attach(
        self,
        seq_id: int,
        kv_tokens: list[Any],
        prefix_tokens: Optional[list[int]] = None,
    ) -> None:
        """
        Attach KV cache for a sequence.

        Implements copy-on-write: if prefix sharing is used, shared pages
        are referenced but will be copied on first write.

        Args:
            seq_id: Sequence identifier
            kv_tokens: KV tokens to cache
            prefix_tokens: Optional prefix tokens for sharing
        """
        if seq_id in self._sequences:
            self.detach(seq_id)

        pages_needed = (len(kv_tokens) + self.page_size - 1) // self.page_size
        new_page_ids = self.allocator.alloc(pages_needed)
        page_ids: list[int] = []

        # Try prefix sharing if enabled
        shared_prefix_pages: list[int] = []
        if self._enable_prefix_sharing and prefix_tokens:
            prefix_hash = self._hash_prefix(prefix_tokens)
            if prefix_hash in self._prefix_map:
                shared_prefix_pages = self._prefix_map[prefix_hash]
                # Reference shared pages (will be copied on write if needed)
                num_prefix_pages = min(len(shared_prefix_pages), pages_needed)
                page_ids.extend(shared_prefix_pages[:num_prefix_pages])
                
                # Update reference counts for shared pages
                for shared_page_id in shared_prefix_pages[:num_prefix_pages]:
                    self._page_refs[shared_page_id] = self._page_refs.get(shared_page_id, 0) + 1
                    self._shared_pages.add(shared_page_id)
                
                # Use remaining allocated pages for non-shared suffix
                page_ids.extend(new_page_ids[num_prefix_pages:])
                self._prefix_shares += 1
            else:
                # First time seeing this prefix - mark these pages as potential shared
                num_prefix_pages = min(
                    (len(prefix_tokens) + self.page_size - 1) // self.page_size,
                    pages_needed
                )
                self._prefix_map[prefix_hash] = new_page_ids[:num_prefix_pages]
                page_ids = new_page_ids
        else:
            page_ids = new_page_ids

        # Store KV data with copy-on-write semantics
        # For shared pages: if data differs, trigger COW; otherwise, reference existing
        for i, page_id in enumerate(page_ids):
            start = i * self.page_size
            end = min(start + self.page_size, len(kv_tokens))
            page_data = kv_tokens[start:end]
            
            # Check if this page is shared
            if page_id in self._shared_pages:
                # Page is shared - check if data matches
                existing_data = self._kv_data.get(page_id, [])
                if existing_data != page_data:
                    # Data differs - trigger copy-on-write
                    page_id = self._copy_if_shared(page_id, seq_id)
                    page_ids[i] = page_id  # Update the page_id in our list
                    # Now safe to write (page is not shared)
                    self._kv_data[page_id] = page_data
                    if page_id not in self._page_refs:
                        self._page_refs[page_id] = 1
                # If data matches, no need to copy or write - just reference the shared page
            else:
                # Non-shared page - safe to write directly
                self._kv_data[page_id] = page_data
                if page_id not in self._page_refs:
                    self._page_refs[page_id] = 1

        self._sequences[seq_id] = page_ids

    def detach(self, seq_id: int) -> None:
        """
        Detach and free KV cache for a sequence.

        Decrements reference counts for shared pages. Pages are only freed
        when their reference count reaches zero.

        Args:
            seq_id: Sequence identifier
        """
        if seq_id not in self._sequences:
            return

        page_ids = self._sequences[seq_id]
        
        # Update reference counts and free pages
        pages_to_free: list[int] = []
        for page_id in page_ids:
            if page_id in self._shared_pages:
                # Shared page - decrement reference count
                self._page_refs[page_id] = self._page_refs.get(page_id, 1) - 1
                if self._page_refs[page_id] <= 0:
                    # No more references - can free
                    self._shared_pages.discard(page_id)
                    if page_id in self._kv_data:
                        del self._kv_data[page_id]
                    if page_id in self._page_refs:
                        del self._page_refs[page_id]
                    pages_to_free.append(page_id)
            else:
                # Non-shared page - free immediately
                if page_id in self._kv_data:
                    del self._kv_data[page_id]
                if page_id in self._page_refs:
                    del self._page_refs[page_id]
                pages_to_free.append(page_id)
        
        # Free pages via allocator
        if pages_to_free:
            self.allocator.free(pages_to_free)

        del self._sequences[seq_id]

    def get(self, seq_id: int) -> Optional[list[Any]]:
        """
        Get KV cache for a sequence.

        Returns a copy of the data to prevent external modifications
        from affecting shared pages.

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
                # Return copy to prevent external modification of shared pages
                kv_tokens.extend(copy.deepcopy(self._kv_data[page_id]))

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
            "shared_pages_count": len(self._shared_pages),
            "total_page_refs": sum(self._page_refs.values()),
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

