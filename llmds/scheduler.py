"""Dynamic micro-batching scheduler with priority queue."""

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from llmds.indexed_heap import IndexedHeap


@dataclass
class Request:
    """Represents a request in the scheduler."""

    request_id: int
    tokens: int
    priority: float  # Higher = more priority
    created_at: float
    slo_ms: Optional[float] = None  # Service level objective in milliseconds


class Scheduler:
    """
    Dynamic micro-batching scheduler with priority-based queuing.

    Uses an indexed heap to prioritize sequences by remaining length or SLO.
    Supports dynamic batching with configurable waiting time vs. throughput trade-offs.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        priority_fn: Optional[Callable[[Request], float]] = None,
    ):
        """
        Initialize scheduler.

        Args:
            max_batch_size: Maximum batch size
            max_wait_ms: Maximum wait time in milliseconds before batching
            priority_fn: Optional function to compute priority from request.
                        Default: prioritize by remaining tokens (inverse)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._heap = IndexedHeap(max_heap=True)  # Max heap for priority
        self._requests: dict[int, Request] = {}
        self._priority_fn = priority_fn or self._default_priority_fn
        self._request_counter = 0
        self._batch_count = 0
        self._total_processed = 0

    def _default_priority_fn(self, req: Request) -> float:
        """Default priority: higher priority for shorter sequences (inverse of tokens)."""
        return 1.0 / (req.tokens + 1.0)

    def _slo_priority_fn(self, req: Request) -> float:
        """Priority based on SLO deadline."""
        if req.slo_ms is None:
            return self._default_priority_fn(req)

        elapsed_ms = (time.time() - req.created_at) * 1000
        remaining_ms = req.slo_ms - elapsed_ms
        if remaining_ms <= 0:
            return float("inf")  # Urgent: past deadline
        return 1.0 / (remaining_ms + 1.0)

    def submit(self, tokens: int, slo_ms: Optional[float] = None) -> int:
        """
        Submit a request to the scheduler.

        Args:
            tokens: Estimated token count for the request
            slo_ms: Optional SLO deadline in milliseconds

        Returns:
            Request ID
        """
        req_id = self._request_counter
        self._request_counter += 1

        req = Request(
            request_id=req_id,
            tokens=tokens,
            priority=self._priority_fn(
                Request(
                    request_id=req_id,
                    tokens=tokens,
                    priority=0.0,
                    created_at=time.time(),
                    slo_ms=slo_ms,
                )
            ),
            created_at=time.time(),
            slo_ms=slo_ms,
        )

        self._requests[req_id] = req
        self._heap.push(req_id, req.priority)

        return req_id

    def get_batch(self, force: bool = False) -> Optional[list[int]]:
        """
        Get next batch of requests to process.

        Args:
            force: If True, return batch even if not full

        Returns:
            List of request IDs or None if no batch ready
        """
        if self._heap.is_empty():
            return None

        # Check if oldest request exceeds max wait time
        oldest_req_id = None
        oldest_time = float("inf")

        for req_id in self._requests:
            if self._requests[req_id].created_at < oldest_time:
                oldest_time = self._requests[req_id].created_at
                oldest_req_id = req_id

        if oldest_req_id:
            wait_time_ms = (time.time() - oldest_time) * 1000
            if not force and wait_time_ms < self.max_wait_ms:
                return None

        # Build batch from heap
        batch: list[int] = []
        temp_heap = IndexedHeap(max_heap=True)

        # Pop top requests
        while len(batch) < self.max_batch_size and not self._heap.is_empty():
            _, req_id = self._heap.pop()
            if req_id in self._requests:
                batch.append(req_id)
            else:
                temp_heap.push(req_id, self._requests[req_id].priority)

        # Restore heap (add back any that weren't used)
        while not temp_heap.is_empty():
            _, req_id = temp_heap.pop()
            self._heap.push(req_id, self._requests[req_id].priority)

        if batch:
            self._batch_count += 1
            self._total_processed += len(batch)
            return batch

        return None

    def complete_batch(self, request_ids: list[int]) -> None:
        """
        Mark a batch as completed and remove requests.

        Args:
            request_ids: List of completed request IDs
        """
        for req_id in request_ids:
            if req_id in self._requests:
                # Try to remove from heap if present
                if self._heap.contains(req_id):
                    try:
                        self._heap.delete(req_id)
                    except KeyError:
                        pass
                del self._requests[req_id]

    def update_priority(self, request_id: int, new_tokens: int) -> None:
        """
        Update priority for a request (e.g., after partial processing).

        Args:
            request_id: Request identifier
            new_tokens: Updated token count
        """
        if request_id not in self._requests:
            return

        req = self._requests[request_id]
        req.tokens = new_tokens
        new_priority = self._priority_fn(req)

        if self._heap.contains(request_id):
            old_priority = self._heap.get_score(request_id)
            if old_priority is not None:
                if new_priority > old_priority:
                    self._heap.increase_key(request_id, new_priority)
                else:
                    self._heap.decrease_key(request_id, new_priority)
        else:
            self._heap.push(request_id, new_priority)

    def stats(self) -> dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        return {
            "queue_size": len(self._requests),
            "batch_count": self._batch_count,
            "total_processed": self._total_processed,
            "avg_batch_size": (
                self._total_processed / self._batch_count if self._batch_count > 0 else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear all pending requests."""
        self._heap = IndexedHeap(max_heap=True)
        self._requests.clear()

