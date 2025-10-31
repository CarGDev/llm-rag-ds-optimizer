"""Tests for scheduler."""

import time

import pytest

from llmds.scheduler import Request, Scheduler


class TestScheduler:
    """Test scheduler functionality."""

    def test_submit_and_get_batch(self):
        """Test submitting requests and getting batches."""
        scheduler = Scheduler(max_batch_size=4, max_wait_ms=10.0)

        req_id1 = scheduler.submit(tokens=100)
        req_id2 = scheduler.submit(tokens=200)

        # Force batch
        batch = scheduler.get_batch(force=True)
        assert batch is not None
        assert len(batch) == 2
        assert req_id1 in batch
        assert req_id2 in batch

    def test_batch_size_limit(self):
        """Test batch size limit."""
        scheduler = Scheduler(max_batch_size=3, max_wait_ms=10.0)

        for i in range(5):
            scheduler.submit(tokens=100)

        batch = scheduler.get_batch(force=True)
        assert batch is not None
        assert len(batch) == 3  # Limited by max_batch_size

    def test_wait_time(self):
        """Test waiting time before batching."""
        scheduler = Scheduler(max_batch_size=4, max_wait_ms=50.0)

        scheduler.submit(tokens=100)

        # Should not return batch immediately
        batch = scheduler.get_batch(force=False)
        assert batch is None

        # Wait and force
        time.sleep(0.06)
        batch = scheduler.get_batch(force=True)
        assert batch is not None

    def test_complete_batch(self):
        """Test completing a batch."""
        scheduler = Scheduler(max_batch_size=4, max_wait_ms=10.0)

        req_id1 = scheduler.submit(tokens=100)
        req_id2 = scheduler.submit(tokens=200)

        batch = scheduler.get_batch(force=True)
        assert batch is not None

        scheduler.complete_batch(batch)

        stats = scheduler.stats()
        assert stats["queue_size"] == 0

    def test_update_priority(self):
        """Test updating request priority."""
        scheduler = Scheduler(max_batch_size=4, max_wait_ms=10.0)

        req_id = scheduler.submit(tokens=100)
        scheduler.update_priority(req_id, new_tokens=50)

        assert scheduler._requests[req_id].tokens == 50


