"""Numerical summaries.

Replaces the usual per-column loop::

    for col in num_cols:
        print(f"{col} | Skew: {df[col].skew():.2f} | Kurtosis: {df[col].kurt():.2f}")

with a single frame, computed from the profile (so the statistics are shared with the
planner rather than recomputed) and annotated with what each number implies.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Thresholds
from ..profiling.profiler import DatasetProfile
from ..types import SemanticType

__all__ = ["numerical_summary", "distribution_flags"]


def distribution_flags(skew: float, thresholds: Thresholds) -> str:
    """A short human label for a skewness value."""
    if not np.isfinite(skew):
        return ""
    magnitude = abs(skew)
    direction = "right" if skew > 0 else "left"
    if magnitude >= thresholds.skew_heavy:
        return f"heavily {direction}-skewed"
    if magnitude >= thresholds.skew_moderate:
        return f"moderately {direction}-skewed"
    return "approximately symmetric"


def numerical_summary(
    profile: DatasetProfile, thresholds: Optional[Thresholds] = None
) -> pd.DataFrame:
    """One row per numeric column: moments, quantiles, and what they imply."""
    thresholds = thresholds or Thresholds()
    rows: List[dict] = []

    for name in profile.column_order:
        cp = profile.columns[name]
        if cp.numeric is None or cp.semantic not in (
            SemanticType.NUMERIC,
            SemanticType.ORDINAL,
        ):
            continue
        stats = cp.numeric
        rows.append(
            {
                "column": name,
                "count": stats.count,
                "missing_%": round(cp.missing_fraction * 100, 2),
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.minimum,
                "q1": stats.q1,
                "median": stats.median,
                "q3": stats.q3,
                "max": stats.maximum,
                "iqr": stats.iqr,
                "skew": stats.skew,
                "kurtosis": stats.kurtosis,
                "n_zeros": stats.n_zeros,
                "n_negative": stats.n_negative,
                "distribution": distribution_flags(stats.skew, thresholds),
                "target_assoc": cp.target_association,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Most-skewed first: those are the columns a reader most needs to look at.
    frame["_sort"] = frame["skew"].abs().fillna(-1)
    frame = frame.sort_values("_sort", ascending=False, ignore_index=True).drop(
        columns="_sort"
    )
    numeric_cols = frame.select_dtypes(include="number").columns
    frame[numeric_cols] = frame[numeric_cols].round(4)
    return frame
