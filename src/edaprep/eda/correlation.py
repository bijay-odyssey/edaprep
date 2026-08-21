"""Correlation and multicollinearity.

Notebook practice computes ``df[num_cols].sample(10_000).corr('spearman')`` and plots a
heatmap.  Two things are added:

* the sampling is principled and *recorded*, rather than an ad-hoc ``.sample()`` whose
  presence a reader of the notebook has to notice;
* VIF is reimplemented in NumPy.  Notebook practice calls ``statsmodels``'
  ``variance_inflation_factor`` in a loop, which refits a regression per feature and is
  O(p) full least-squares solves; inverting the correlation matrix once gives the same
  numbers in one step, and drops the statsmodels dependency entirely.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..profiling.profiler import DatasetProfile
from ..types import SemanticType

__all__ = ["correlation_matrix", "top_correlated_pairs", "variance_inflation"]


def _numeric_columns(profile: DatasetProfile, data: pd.DataFrame) -> List[str]:
    return [
        name
        for name in profile.column_order
        if name in data.columns
        and profile.columns[name].semantic in (SemanticType.NUMERIC, SemanticType.ORDINAL)
        and not profile.columns[name].is_constant
        and pd.api.types.is_numeric_dtype(data[name].dtype)
    ]


def correlation_matrix(
    data: pd.DataFrame,
    profile: DatasetProfile,
    method: str = "spearman",
    config: Optional[Config] = None,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """Correlation matrix over numeric columns, sampled on large frames.

    Returns ``None`` when the frame is too wide and ``force`` is False: a 500x500
    matrix is 250,000 numbers that nobody reads, and computing it is the single most
    expensive part of a "standard" analysis.  ``level="deep"`` sets ``force``.
    """
    config = config or Config()
    columns = _numeric_columns(profile, data)
    if len(columns) < 2:
        return None
    if not force and len(columns) > config.thresholds.correlation_max_columns:
        return None

    frame = data[columns]
    limit = config.effective_sample_size
    if len(frame) > limit:
        seed = config.random_state if config.random_state is not None else 0
        frame = frame.sample(n=limit, random_state=seed)

    return frame.corr(method=method, numeric_only=True)


def top_correlated_pairs(
    corr: pd.DataFrame, threshold: float = 0.7, top: int = 30
) -> pd.DataFrame:
    """Column pairs whose absolute correlation is at or above ``threshold``."""
    if corr is None or corr.empty:
        return pd.DataFrame(columns=["column_a", "column_b", "correlation"])
    values = corr.to_numpy()
    names = list(corr.columns)
    # Upper triangle only: the matrix is symmetric, so the lower half is the same pairs.
    rows, cols = np.triu_indices_from(values, k=1)
    magnitudes = np.abs(values[rows, cols])
    keep = np.isfinite(magnitudes) & (magnitudes >= threshold)
    order = np.argsort(-magnitudes[keep])
    selected_rows = rows[keep][order][:top]
    selected_cols = cols[keep][order][:top]
    return pd.DataFrame(
        {
            "column_a": [names[i] for i in selected_rows],
            "column_b": [names[j] for j in selected_cols],
            "correlation": np.round(values[selected_rows, selected_cols], 4),
        }
    )


def variance_inflation(
    data: pd.DataFrame,
    profile: DatasetProfile,
    config: Optional[Config] = None,
    max_columns: int = 100,
) -> Optional[pd.DataFrame]:
    """Variance inflation factors, from one matrix inversion.

    ``VIF_i = 1 / (1 - R_i^2)`` where ``R_i^2`` is from regressing feature *i* on the
    others.  The diagonal of the inverted correlation matrix equals exactly that, so
    the whole vector comes from a single inversion rather than *p* regressions.

    A singular correlation matrix means at least one feature is an exact linear
    combination of the others -- infinite VIF.  The pseudo-inverse is used so that the
    remaining, well-determined factors are still reported instead of the whole call
    failing, and the perfectly collinear columns are marked with ``inf``.
    """
    config = config or Config()
    columns = _numeric_columns(profile, data)
    if len(columns) < 2:
        return None
    if len(columns) > max_columns:
        columns = columns[:max_columns]

    frame = data[columns]
    limit = config.effective_sample_size
    if len(frame) > limit:
        seed = config.random_state if config.random_state is not None else 0
        frame = frame.sample(n=limit, random_state=seed)

    # VIF is undefined with missing values; dropping rows keeps the estimate honest,
    # whereas imputing first would deflate the correlations that VIF measures.
    frame = frame.dropna()
    if len(frame) <= len(columns):
        return pd.DataFrame(
            {
                "column": columns,
                "vif": [np.nan] * len(columns),
                "note": ["too few complete rows to estimate"] * len(columns),
            }
        )

    corr = frame.corr(method="pearson", numeric_only=True).to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    try:
        inverse = np.linalg.inv(corr)
        singular = False
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(corr)
        singular = True

    vif = np.diag(inverse).astype(float)
    vif = np.where(vif < 1.0, 1.0, vif)  # numerical noise can push it just below 1
    if singular:
        # A pseudo-inverse silently caps the diagonal, so perfect collinearity would
        # otherwise show as a merely large VIF rather than an infinite one.
        rank = np.linalg.matrix_rank(corr)
        if rank < len(columns):
            vif = np.where(vif > 1e10, np.inf, vif)

    frame_out = pd.DataFrame(
        {
            "column": columns,
            "vif": np.round(vif, 3),
            "note": [_vif_note(v) for v in vif],
        }
    )
    return frame_out.sort_values("vif", ascending=False, ignore_index=True)


def _vif_note(value: float) -> str:
    if not np.isfinite(value):
        return "perfectly collinear with other features"
    if value >= 10:
        return "severe multicollinearity"
    if value >= 5:
        return "moderate multicollinearity"
    return ""
