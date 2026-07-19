"""Deterministic benchmark registry and diagnostic suites."""

from saliencegate.benchmarks.registry import (
    BenchmarkDefinition,
    BenchmarkNotFoundError,
    available_benchmarks,
    get_benchmark,
)

__all__ = [
    "BenchmarkDefinition",
    "BenchmarkNotFoundError",
    "available_benchmarks",
    "get_benchmark",
]
