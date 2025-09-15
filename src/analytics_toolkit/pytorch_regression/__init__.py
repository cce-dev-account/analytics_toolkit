"""
PyTorch-based statistical regression module.

This module provides PyTorch implementations of regression models with
comprehensive statistical inference capabilities, similar to statsmodels
but with the computational efficiency of PyTorch.
"""

from .base import BaseRegression
from .linear import LinearRegression
from .logistic import LogisticRegression
from .stats import (
    compute_confidence_intervals,
    compute_model_statistics,
    compute_standard_errors,
    format_summary_table,
)
from .utils import calculate_vif, detect_categorical_columns

__all__ = [
    "BaseRegression",
    "LinearRegression",
    "LogisticRegression",
    "calculate_vif",
    "detect_categorical_columns",
    "compute_standard_errors",
    "compute_confidence_intervals",
    "compute_model_statistics",
    "format_summary_table",
]

__version__ = "0.1.0"
