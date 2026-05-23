"""Utility functions for token counting and benchmarking."""

import time
from functools import wraps
from typing import Callable

from ..core.compressor import estimate_tokens


def count_tokens_batch(texts: list[str]) -> list[int]:
    """Batch token estimation for multiple texts."""
    return [estimate_tokens(t) for t in texts]


def token_reduction_report(
    original: str, compressed: str
) -> dict:
    """Generate a human-readable reduction report."""
    orig_tokens = estimate_tokens(original)
    comp_tokens = estimate_tokens(compressed)
    saved = orig_tokens - comp_tokens
    pct = (saved / orig_tokens * 100) if orig_tokens > 0 else 0
    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "tokens_saved": saved,
        "reduction_pct": round(pct, 1),
    }


def benchmark(func: Callable) -> Callable:
    """Decorator to measure compression speed."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[ContextFlow] {func.__name__}: {elapsed_ms:.1f}ms")
        return result

    return wrapper
