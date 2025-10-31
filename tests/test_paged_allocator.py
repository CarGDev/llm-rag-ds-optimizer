"""Tests for paged allocator."""

import pytest

from llmds.paged_allocator import PagedAllocator


class TestPagedAllocator:
    """Test paged allocator functionality."""

    def test_alloc_free(self):
        """Test basic allocation and deallocation."""
        allocator = PagedAllocator(page_size=512, max_pages=10)
        pages = allocator.alloc(5)
        assert len(pages) == 5
        assert all(isinstance(p, int) for p in pages)

        stats = allocator.stats()
        assert stats.allocated_pages == 5
        assert stats.free_pages == 5

        allocator.free(pages)
        stats = allocator.stats()
        assert stats.allocated_pages == 0
        assert stats.free_pages == 10

    def test_insufficient_pages(self):
        """Test error when insufficient pages."""
        allocator = PagedAllocator(page_size=512, max_pages=5)
        with pytest.raises(ValueError):
            allocator.alloc(10)

    def test_fragmentation(self):
        """Test fragmentation tracking."""
        allocator = PagedAllocator(page_size=512, max_pages=10)
        pages1 = allocator.alloc(3)
        pages2 = allocator.alloc(4)

        stats = allocator.stats()
        assert stats.allocated_pages == 7
        assert stats.fragmentation_ratio > 0

        allocator.free(pages1)
        allocator.free(pages2)

    def test_defragment(self):
        """Test defragmentation."""
        allocator = PagedAllocator(page_size=512, max_pages=10)
        pages1 = allocator.alloc(3)
        pages2 = allocator.alloc(4)
        allocator.free(pages1)

        allocator.defragment()
        stats = allocator.stats()
        assert stats.free_pages == 3


