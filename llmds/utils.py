"""Utility functions."""

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def Timer():
    """Context manager for timing code execution."""
    start = time.perf_counter()
    try:
        yield
    finally:
        pass
    
    @property
    def elapsed(self):
        """Elapsed time in seconds."""
        return time.perf_counter() - start
    
    # Make it work as context manager with attribute access
    timer = type("Timer", (), {"elapsed": time.perf_counter() - start})()
    timer.elapsed = time.perf_counter() - start
    return timer


class Timer:
    """Simple timer context manager."""
    
    def __init__(self):
        self.start = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        return False
