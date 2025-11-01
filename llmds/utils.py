"""Utility functions."""

import time
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore


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
