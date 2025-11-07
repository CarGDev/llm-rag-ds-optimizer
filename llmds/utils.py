"""Utility functions."""

import time
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Optional

import numpy as np

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class Timer:
    """Simple timer context manager."""
    
    def __init__(self) -> None:
        self.start: float | None = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> Literal[False]:
        if self.start is not None:
            self.elapsed = time.perf_counter() - self.start
        return False


class MemoryProfiler:
    """
    Memory profiler for measuring peak RSS (Resident Set Size).
    
    Tracks memory usage during benchmark execution and reports peak RSS.
    """
    
    def __init__(self) -> None:
        """Initialize memory profiler."""
        if not _PSUTIL_AVAILABLE:
            raise ImportError("psutil is required for memory profiling. Install with: pip install psutil")
        
        self.process = psutil.Process()
        self.initial_rss: Optional[int] = None
        self.peak_rss: int = 0
        self.current_rss: int = 0
        
    def start(self) -> None:
        """Start memory profiling."""
        self.initial_rss = self.process.memory_info().rss
        self.peak_rss = self.initial_rss
        self.current_rss = self.initial_rss
    
    def sample(self) -> int:
        """
        Sample current RSS and update peak.
        
        Returns:
            Current RSS in bytes
        """
        if not _PSUTIL_AVAILABLE:
            return 0
        
        self.current_rss = self.process.memory_info().rss
        if self.current_rss > self.peak_rss:
            self.peak_rss = self.current_rss
        return self.current_rss
    
    def get_peak_rss_mb(self) -> float:
        """
        Get peak RSS in megabytes.
        
        Returns:
            Peak RSS in MB
        """
        return self.peak_rss / (1024 * 1024)
    
    def get_peak_rss_bytes(self) -> int:
        """
        Get peak RSS in bytes.
        
        Returns:
            Peak RSS in bytes
        """
        return self.peak_rss
    
    def get_current_rss_mb(self) -> float:
        """
        Get current RSS in megabytes.
        
        Returns:
            Current RSS in MB
        """
        return self.current_rss / (1024 * 1024)
    
    def get_memory_delta_mb(self) -> float:
        """
        Get memory delta from initial RSS in megabytes.
        
        Returns:
            Memory delta in MB (peak - initial)
        """
        if self.initial_rss is None:
            return 0.0
        return (self.peak_rss - self.initial_rss) / (1024 * 1024)


@contextmanager
def memory_profiler() -> Iterator[MemoryProfiler]:
    """
    Context manager for memory profiling.
    
    Usage:
        with memory_profiler() as profiler:
            # Your code here
            profiler.sample()  # Optional: sample at specific points
        peak_rss_mb = profiler.get_peak_rss_mb()
    
    Yields:
        MemoryProfiler instance
    """
    if not _PSUTIL_AVAILABLE:
        # Return dummy profiler if psutil not available
        class DummyProfiler:
            def start(self) -> None: pass
            def sample(self) -> int: return 0
            def get_peak_rss_mb(self) -> float: return 0.0
            def get_peak_rss_bytes(self) -> int: return 0
            def get_current_rss_mb(self) -> float: return 0.0
            def get_memory_delta_mb(self) -> float: return 0.0
        
        profiler = DummyProfiler()  # type: ignore
        profiler.start()
        yield profiler
        return
    
    profiler = MemoryProfiler()
    profiler.start()
    try:
        yield profiler
        # Final sample to capture any last-minute allocations
        profiler.sample()
    finally:
        pass


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """
    Compute P50, P95, P99 percentiles from a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with p50, p95, p99 keys
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    return {
        "p50": sorted_values[n // 2],
        "p95": sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0],
        "p99": sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
    }


def calculate_statistics(values: list[float], confidence_level: float = 0.95) -> dict[str, Any]:
    """
    Calculate statistical summary for a list of values.
    
    Args:
        values: List of numeric values
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Dictionary with mean, std, min, max, percentiles, and confidence intervals
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "cv": 0.0,  # Coefficient of variation
        }
    
    values_array = np.array(values)
    mean = float(np.mean(values_array))
    std = float(np.std(values_array, ddof=1))  # Sample std dev (ddof=1)
    min_val = float(np.min(values_array))
    max_val = float(np.max(values_array))
    
    # Percentiles
    p50 = float(np.percentile(values_array, 50))
    p95 = float(np.percentile(values_array, 95))
    p99 = float(np.percentile(values_array, 99))
    
    # Confidence interval (t-distribution for small samples)
    n = len(values)
    if n > 1:
        alpha = 1 - confidence_level
        if HAS_SCIPY:
            # Use t-distribution for small samples
            t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_critical * (std / np.sqrt(n))
        else:
            # Fallback: use normal distribution approximation (z-score)
            # For 95% CI: z = 1.96, for 90% CI: z = 1.645
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_critical = z_scores.get(confidence_level, 1.96)
            margin = z_critical * (std / np.sqrt(n))
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = mean
        ci_upper = mean
    
    # Coefficient of variation (relative standard deviation)
    cv = (std / mean * 100) if mean > 0 else 0.0
    
    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cv": cv,  # Coefficient of variation (%)
        "count": n,
    }
