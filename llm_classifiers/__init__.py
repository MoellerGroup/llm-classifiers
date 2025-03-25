"""
Framework to build production-grade LLM-based classifiers.

This library provides tools to create, evaluate, and deploy
LLM-based classifiers with structured outputs.
"""

from llm_classifiers.classifier import Classifier, Example
from llm_classifiers.evaluation import (
    ClassifierEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    EvaluationFormat
)

__all__ = [
    "Classifier",
    "Example",
    "ClassifierEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "EvaluationFormat"
]