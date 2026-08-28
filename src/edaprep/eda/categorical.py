"""Categorical summaries.

Replaces the usual per-column ``value_counts`` / ``countplot`` loop with one frame
covering cardinality, concentration, rare levels, and the encoding implication -- the
last being the number a reader actually needs before deciding how to encode.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..config import Config
from ..profiling.profiler import DatasetProfile
from ..types import SemanticType

__all__ = ["categorical_summary", "category_frequencies"]


def categorical_summary(
    data: pd.DataFrame, profile: DatasetProfile, config: Optional[Config] = None
) -> pd.DataFrame:
    """One row per categorical column."""
    config = config or Config()
    rare_threshold = config.effective_rare_threshold
    high_cardinality = config.effective_high_cardinality
    n_rows = profile.n_rows
    floor = max(1, int(rare_threshold * n_rows))

    rows: List[dict] = []
    for name in profile.column_order:
        cp = profile.columns[name]
        if cp.semantic not in (
            SemanticType.CATEGORICAL,
            SemanticType.BINARY,
            SemanticType.ORDINAL,
        ):
            continue

        n_rare = 0
        if name in data.columns:
            try:
                counts = data[name].value_counts(dropna=True)
                n_rare = int((counts < floor).sum())
            except TypeError:
                n_rare = 0

        top = cp.top_values[:3]
        rows.append(
            {
                "column": name,
                "semantic": str(cp.semantic),
                "n_unique": cp.n_unique,
                "missing_%": round(cp.missing_fraction * 100, 2),
                "modal_value": cp.modal_value,
                "modal_%": round(cp.modal_frequency * 100, 2),
                "n_rare_levels": n_rare,
                "top_values": ", ".join(f"{v} ({c})" for v, c in top),
                "onehot_columns": cp.n_unique if cp.n_unique <= high_cardinality else None,
                "note": _note(cp, high_cardinality, n_rare),
                "target_assoc": cp.target_association,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("n_unique", ascending=False, ignore_index=True)
    return frame


def _note(cp, high_cardinality: int, n_rare: int) -> str:
    notes = []
    if cp.n_unique > high_cardinality:
        notes.append(f"high cardinality: one-hot would add {cp.n_unique} columns")
    if cp.is_near_constant:
        notes.append(f"near-constant ({cp.modal_frequency:.1%} one value)")
    if n_rare:
        notes.append(f"{n_rare} rare level(s)")
    return "; ".join(notes)


def category_frequencies(
    data: pd.DataFrame, column: str, top: int = 20, dropna: bool = False
) -> pd.DataFrame:
    """Frequency table for one column, including the missing count.

    ``dropna=False`` by default because "how often is this missing" is usually part of
    the question being asked of a frequency table.
    """
    counts = data[column].value_counts(dropna=dropna)
    total = int(counts.sum())
    frame = counts.head(top).rename("count").to_frame()
    frame.index.name = "value"
    frame = frame.reset_index()
    frame["percent"] = (frame["count"] / total * 100).round(2) if total else 0.0
    frame["cumulative_%"] = frame["percent"].cumsum().round(2)
    return frame
