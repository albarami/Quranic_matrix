"""
Evaluation Harness for QBM Brain

Runs the 200-question benchmark suite against capability engines.
Generates reports with pass/fail metrics by section.
"""

from .harness import EvaluationHarness, EvaluationResult
from .runner import run_benchmark_suite

__all__ = [
    "EvaluationHarness",
    "EvaluationResult",
    "run_benchmark_suite",
]
