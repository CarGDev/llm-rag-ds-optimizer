"""Paged memory allocator with slab allocation for KV cache."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PageStats:
    """Statistics for page allocation."""

    total_pages: int
    allocated_pages: int
    free_pages: int
    fragmentation_ratio: float
    allocation_count: int
    free_count: int


class PagedAllocator:
    """
    Paged memory allocator with fixed-size pages and freelist management.

    Uses a slab allocator approach with freelists for efficient allocation
    and deallocation of fixed-size page blocks.
    """

    def __init__(self, page_size: int, max_pages: int):
        """
        Initialize the paged allocator.

        Args:
            page_size: Size of each page in tokens/bytes
            max_pages: Maximum number of pages to allocate
        """
        self.page_size = page_size
        self.max_pages = max_pages
        self._pages: list[Optional[bool]] = [None] * max_pages  # None=free, True=allocated
        self._free_list: list[int] = list(range(max_pages))
        self._allocation_count = 0
        self._free_count = 0

    def alloc(self, num_pages: int) -> list[int]:
        """
        Allocate a contiguous block of pages.

        Args:
            num_pages: Number of pages to allocate

        Returns:
            List of page IDs (indices)

        Raises:
            ValueError: If insufficient pages available
        """
        if len(self._free_list) < num_pages:
            raise ValueError(f"Insufficient pages: requested {num_pages}, available {len(self._free_list)}")

        allocated = []
        for _ in range(num_pages):
            page_id = self._free_list.pop(0)
            self._pages[page_id] = True
            allocated.append(page_id)
            self._allocation_count += 1

        return allocated

    def free(self, page_ids: list[int]) -> None:
        """
        Free a list of pages.

        Args:
            page_ids: List of page IDs to free
        """
        for page_id in page_ids:
            if 0 <= page_id < self.max_pages and self._pages[page_id] is True:
                self._pages[page_id] = None
                self._free_list.append(page_id)
                self._free_count += 1

    def stats(self) -> PageStats:
        """
        Get allocation statistics.

        Returns:
            PageStats object with current statistics
        """
        allocated = sum(1 for p in self._pages if p is True)
        free = len(self._free_list)
        fragmentation = 1.0 - (free / self.max_pages) if self.max_pages > 0 else 0.0

        return PageStats(
            total_pages=self.max_pages,
            allocated_pages=allocated,
            free_pages=free,
            fragmentation_ratio=fragmentation,
            allocation_count=self._allocation_count,
            free_count=self._free_count,
        )

    def defragment(self) -> None:
        """
        Defragment pages by compacting allocated pages.

        This is a simple implementation that moves allocated pages
        to the front. More sophisticated strategies could be implemented.
        """
        allocated_indices = [i for i, p in enumerate(self._pages) if p is True]
        free_indices = [i for i, p in enumerate(self._pages) if p is None]

        # Simple compaction: move allocated pages to front
        new_pages: list[bool | None] = [None] * self.max_pages
        for i, idx in enumerate(allocated_indices):
            new_pages[i] = True

        self._pages = new_pages
        self._free_list = list(range(len(allocated_indices), self.max_pages))

