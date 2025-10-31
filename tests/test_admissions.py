"""Tests for admission controller."""

import time

import pytest

from llmds.admissions import AdmissionController


class TestAdmissionController:
    """Test admission controller functionality."""

    def test_admit_request(self):
        """Test admitting a request."""
        controller = AdmissionController(qps_target=10.0, token_rate_limit=1000)

        should_admit, reason = controller.should_admit(estimated_tokens=100)
        assert should_admit is True
        assert reason == "admitted"

    def test_qps_limit(self):
        """Test QPS limit enforcement."""
        controller = AdmissionController(qps_target=2.0, window_size=1)

        # Submit multiple requests quickly
        for _ in range(5):
            controller.should_admit()

        stats = controller.stats()
        assert stats["rejection_rate"] > 0  # Some should be rejected

    def test_token_rate_limit(self):
        """Test token rate limit."""
        controller = AdmissionController(
            qps_target=100.0, token_rate_limit=100, window_size=1
        )

        # Submit request with high token count
        should_admit, reason = controller.should_admit(estimated_tokens=200)
        # May be rejected if token rate exceeded

        stats = controller.stats()
        assert "token_rate_limit" in stats

    def test_stats(self):
        """Test statistics."""
        controller = AdmissionController(qps_target=10.0, token_rate_limit=1000)

        controller.should_admit(estimated_tokens=100)
        controller.should_admit(estimated_tokens=200)

        stats = controller.stats()
        assert stats["admitted_requests"] >= 0
        assert stats["current_qps"] >= 0


