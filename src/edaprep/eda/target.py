"""Target relationships.

Replaces the usual statistical-test block::

    if len(unique_val) == 2:
        stats, p = ttest_ind(group1, group2)
    else:
        stats, p = f_oneway(*groups)

with the same tests, applied to the right pairings, plus the association measures
already on the profile so that numeric and categorical features are comparable on one
scale.

One correction: it is common to run the tests on every categorical column and sort by
p-value, which with 400 columns guarantees small p-values by chance alone.  A
Benjamini-Hochberg adjusted q-value is reported alongside, so the ranking survives
being looked at.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..profiling.profiler import DatasetProfile
from ..types import NUMERIC_LIKE, SemanticType

__all__ = ["target_summary", "target_relationships", "benjamini_hochberg"]


def target_summary(
    data: pd.DataFrame, profile: DatasetProfile, target: str
) -> Dict[str, Any]:
    """Distribution of the target, plus the imbalance measurement."""
    series = data[target]
    out: Dict[str, Any] = {
        "column": target,
        "kind": profile.target_kind,
        "n_missing": int(series.isna().sum()),
    }
    if profile.target_kind == "classification":
        counts = series.value_counts(dropna=True)
        out["n_classes"] = int(len(counts))
        out["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["imbalance_ratio"] = (
            round(float(counts.min() / counts.max()), 6) if len(counts) > 1 else 1.0
        )
        out["majority_class"] = str(counts.index[0])
        out["minority_class"] = str(counts.index[-1])
        if out["imbalance_ratio"] < 0.2:
            out["note"] = (
                "Imbalanced. Handle it at model-fitting time with class weights or "
                "resampling, after the train/test split. edaprep reports imbalance but "
                "does not resample, because resampling is a modelling decision."
            )
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        out["mean"] = round(float(numeric.mean()), 6)
        out["std"] = round(float(numeric.std(ddof=1)), 6)
        out["min"] = float(numeric.min())
        out["max"] = float(numeric.max())
        cp = profile.columns.get(target)
        if cp is not None and cp.numeric is not None and np.isfinite(cp.numeric.skew):
            out["skew"] = round(float(cp.numeric.skew), 4)
            if abs(cp.numeric.skew) >= 1.0:
                out["note"] = (
                    f"The target is skewed ({cp.numeric.skew:.2f}). Consider modelling "
                    f"log1p(y) and inverting the prediction, rather than transforming "
                    f"the target column in place -- the latter is easy to forget to "
                    f"undo, and edaprep never transforms a target."
                )
    return out


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), controlling the FDR.

    Without this, testing 400 columns at alpha=0.05 produces ~20 "significant"
    features from pure noise, which is precisely the situation a wide dataset creates.
    """
    p = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if not finite.any():
        return out
    values = p[finite]
    n = values.size
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downwards, as the procedure requires.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    out[finite] = result
    return out


def target_relationships(
    data: pd.DataFrame,
    profile: DatasetProfile,
    target: str,
    deep: bool = False,
    max_groups: int = 50,
) -> pd.DataFrame:
    """One row per feature: association with the target, and a significance test."""
    rows: List[dict] = []
    target_series = data[target]
    is_classification = profile.target_kind == "classification"

    for name in profile.column_order:
        if name == target or name not in data.columns:
            continue
        cp = profile.columns[name]
        if cp.is_constant or cp.semantic in (SemanticType.TEXT, SemanticType.IDENTIFIER):
            continue

        row = {
            "column": name,
            "semantic": str(cp.semantic),
            "association": (
                round(cp.target_association, 4)
                if cp.target_association is not None
                else None
            ),
            "measure": cp.target_association_kind,
        }
        if deep:
            statistic, p_value, test = _test(
                data[name], target_series, cp.semantic, is_classification, max_groups
            )
            row["test"] = test
            row["statistic"] = round(statistic, 4) if statistic is not None else None
            row["p_value"] = p_value
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    if deep and "p_value" in frame:
        frame["q_value"] = np.round(benjamini_hochberg(frame["p_value"].to_numpy()), 6)
        frame["p_value"] = frame["p_value"].round(6)

    frame["_sort"] = frame["association"].fillna(-1)
    return frame.sort_values("_sort", ascending=False, ignore_index=True).drop(
        columns="_sort"
    )


def _test(
    feature: pd.Series,
    target: pd.Series,
    semantic: SemanticType,
    is_classification: bool,
    max_groups: int,
):
    """Pick and run the appropriate test.  Returns ``(statistic, p_value, name)``."""
    try:
        from scipy import stats as scipy_stats
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return None, None, "scipy unavailable"

    mask = feature.notna() & target.notna()
    if mask.sum() < 8:
        return None, None, "too few complete rows"
    x, y = feature[mask], target[mask]
    numeric_feature = semantic in NUMERIC_LIKE or semantic is SemanticType.BINARY

    try:
        if is_classification and numeric_feature:
            groups = [
                pd.to_numeric(x[y == level], errors="coerce").dropna().to_numpy()
                for level in y.unique()[:max_groups]
            ]
            groups = [g for g in groups if g.size > 1]
            if len(groups) < 2:
                return None, None, "not enough groups"
            if len(groups) == 2:
                # Welch's t-test: the two classes rarely have equal variance, and
                # assuming they do (the usual default) inflates significance.
                statistic, p = scipy_stats.ttest_ind(*groups, equal_var=False)
                return float(statistic), float(p), "welch t-test"
            statistic, p = scipy_stats.f_oneway(*groups)
            return float(statistic), float(p), "anova"

        if is_classification and not numeric_feature:
            table = pd.crosstab(x, y)
            if min(table.shape) < 2:
                return None, None, "degenerate contingency table"
            statistic, p, _, _ = scipy_stats.chi2_contingency(table)
            return float(statistic), float(p), "chi-squared"

        if numeric_feature:
            xv = pd.to_numeric(x, errors="coerce")
            keep = xv.notna()
            if keep.sum() < 8:
                return None, None, "too few complete rows"
            statistic, p = scipy_stats.spearmanr(xv[keep], y[keep])
            return float(statistic), float(p), "spearman"

        groups = [
            pd.to_numeric(y[x == level], errors="coerce").dropna().to_numpy()
            for level in x.unique()[:max_groups]
        ]
        groups = [g for g in groups if g.size > 1]
        if len(groups) < 2:
            return None, None, "not enough groups"
        statistic, p = scipy_stats.f_oneway(*groups)
        return float(statistic), float(p), "anova"
    except (ValueError, TypeError) as exc:
        return None, None, f"test failed: {type(exc).__name__}"
