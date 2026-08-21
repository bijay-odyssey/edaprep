"""Outlier summary for EDA.

Reports what *would* be flagged under each fence, without changing anything.  Showing
the three methods side by side is deliberate: they disagree, often substantially, and
seeing the disagreement is what stops a reader treating "outlier" as an objective
property of a value rather than an artefact of the fence chosen.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..preprocessing.outliers import (
    IQRDetector,
    ModifiedZScoreDetector,
    PercentileDetector,
    ZScoreDetector,
)
from ..profiling.profiler import DatasetProfile
from ..types import SemanticType

__all__ = ["outlier_summary"]


def outlier_summary(
    data: pd.DataFrame, profile: DatasetProfile, config: Optional[Config] = None
) -> pd.DataFrame:
    """One row per numeric column, with counts under each fence.

    The ``recommended`` column names the method the planner would choose for that
    column, so the table doubles as a preview of the plan.
    """
    config = config or Config()
    thresholds = config.thresholds

    detectors = {
        "iqr": IQRDetector(k=thresholds.iqr_k),
        "iqr_k3": IQRDetector(k=thresholds.iqr_k_skewed),
        "zscore": ZScoreDetector(threshold=thresholds.zscore_threshold),
        "modified_z": ModifiedZScoreDetector(threshold=thresholds.modified_zscore_threshold),
        "percentile": PercentileDetector(*thresholds.percentile_bounds),
    }

    rows: List[dict] = []
    for name in profile.column_order:
        cp = profile.columns[name]
        if (
            cp.semantic is not SemanticType.NUMERIC
            or name not in data.columns
            or cp.n_unique <= 2
        ):
            continue
        values = pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=np.float64)
        n_finite = int(np.count_nonzero(np.isfinite(values)))
        if n_finite == 0:
            continue

        skew = round(float(cp.skew), 3) if np.isfinite(cp.skew) else None
        row = {"column": name, "skew": skew}
        for label, detector in detectors.items():
            bounds = detector(values)
            count = int(np.count_nonzero(bounds.mask(values)))
            row[f"n_{label}"] = count
            row[f"pct_{label}"] = round(count / n_finite * 100, 2)

        row["recommended"] = _recommended(cp.skew, thresholds)
        chosen = detectors[_detector_key(row["recommended"])](values)
        row["lower"] = round(float(chosen.lower), 4)
        row["upper"] = round(float(chosen.upper), 4)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("pct_iqr", ascending=False, ignore_index=True)


def _recommended(skew: float, thresholds) -> str:
    if not np.isfinite(skew):
        return "iqr"
    magnitude = abs(skew)
    if magnitude >= thresholds.skew_heavy:
        return "modified_zscore"
    if magnitude >= thresholds.skew_moderate:
        return "iqr (k=3)"
    return "zscore"


def _detector_key(recommended: str) -> str:
    return {
        "modified_zscore": "modified_z",
        "iqr (k=3)": "iqr_k3",
        "zscore": "zscore",
        "iqr": "iqr",
    }[recommended]
