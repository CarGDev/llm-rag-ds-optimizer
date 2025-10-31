"""Admission controller with rate limiting and QPS tracking."""

import time
from collections import deque
from typing import Optional


class AdmissionController:
    """
    Admission controller with token-rate limiting and moving-average QPS.

    Controls admission based on token budget and QPS targets.
    """

    def __init__(
        self,
        qps_target: float = 10.0,
        token_rate_limit: int = 10000,
        window_size: int = 10,
    ):
        """
        Initialize admission controller.

        Args:
            qps_target: Target queries per second
            token_rate_limit: Maximum tokens per second
            window_size: Size of moving average window in seconds
        """
        self.qps_target = qps_target
        self.token_rate_limit = token_rate_limit
        self.window_size = window_size
        self._request_times: deque[float] = deque()
        self._token_history: deque[tuple[float, int]] = deque()  # (time, tokens)
        self._admitted_requests = 0
        self._rejected_requests = 0

    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove requests outside the time window."""
        while self._request_times and current_time - self._request_times[0] > self.window_size:
            self._request_times.popleft()

        while self._token_history and current_time - self._token_history[0][0] > self.window_size:
            self._token_history.popleft()

    def _get_current_qps(self, current_time: float) -> float:
        """Calculate current QPS over the window."""
        self._cleanup_old_requests(current_time)
        if not self._request_times:
            return 0.0
        return len(self._request_times) / self.window_size

    def _get_current_token_rate(self, current_time: float) -> float:
        """Calculate current token rate over the window."""
        self._cleanup_old_requests(current_time)
        if not self._token_history:
            return 0.0

        total_tokens = sum(tokens for _, tokens in self._token_history)
        return total_tokens / self.window_size

    def should_admit(self, estimated_tokens: int = 0) -> tuple[bool, str]:
        """
        Check if a request should be admitted.

        Args:
            estimated_tokens: Estimated tokens for this request

        Returns:
            Tuple of (should_admit, reason)
        """
        current_time = time.time()
        current_qps = self._get_current_qps(current_time)
        current_token_rate = self._get_current_token_rate(current_time)

        # Check QPS limit
        if current_qps >= self.qps_target:
            self._rejected_requests += 1
            return False, f"QPS limit exceeded: {current_qps:.2f} >= {self.qps_target}"

        # Check token rate limit
        if current_token_rate + estimated_tokens / self.window_size > self.token_rate_limit:
            self._rejected_requests += 1
            return False, f"Token rate limit exceeded"

        # Admit request
        self._request_times.append(current_time)
        if estimated_tokens > 0:
            self._token_history.append((current_time, estimated_tokens))
        self._admitted_requests += 1

        return True, "admitted"

    def record_request(self, tokens: int) -> None:
        """
        Record a completed request with token count.

        Args:
            tokens: Number of tokens processed
        """
        current_time = time.time()
        self._token_history.append((current_time, tokens))

    def stats(self) -> dict[str, float]:
        """
        Get admission statistics.

        Returns:
            Dictionary with admission statistics
        """
        current_time = time.time()
        current_qps = self._get_current_qps(current_time)
        current_token_rate = self._get_current_token_rate(current_time)

        total_requests = self._admitted_requests + self._rejected_requests
        rejection_rate = (
            self._rejected_requests / total_requests if total_requests > 0 else 0.0
        )

        return {
            "current_qps": current_qps,
            "target_qps": self.qps_target,
            "current_token_rate": current_token_rate,
            "token_rate_limit": self.token_rate_limit,
            "admitted_requests": self._admitted_requests,
            "rejected_requests": self._rejected_requests,
            "rejection_rate": rejection_rate,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._request_times.clear()
        self._token_history.clear()
        self._admitted_requests = 0
        self._rejected_requests = 0

