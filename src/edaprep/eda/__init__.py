"""Exploratory data analysis."""

from .analyzer import EDA, EDAReport
from .categorical import categorical_summary, category_frequencies
from .correlation import correlation_matrix, top_correlated_pairs, variance_inflation
from .numerical import numerical_summary
from .outliers import outlier_summary
from .target import target_relationships, target_summary

__all__ = [
    "EDA",
    "EDAReport",
    "numerical_summary",
    "categorical_summary",
    "category_frequencies",
    "correlation_matrix",
    "top_correlated_pairs",
    "variance_inflation",
    "outlier_summary",
    "target_summary",
    "target_relationships",
]
