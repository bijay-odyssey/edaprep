"""Data-quality detection: sentinels, co-missingness, and structured issue records.

Two of these checks are the kind normally written once, by hand, for a single dataset:

* the ``(df == '?').sum()`` scan in ``a census-income notebook`` becomes
  :func:`detect_sentinels`, generalised to a configurable vocabulary;
* the ``df[cols].isnull().astype(int).corr()`` co-missingness analysis in the same
  notebook becomes :func:`missingness_correlation`.

Everything here returns :class:`QualityIssue` records, which are deliberately
schema-shaped: they are the seam a future data-validation layer would build on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..types import Severity

__all__ = [
    "QualityIssue",
    "detect_sentinels",
    "detect_numeric_sentinels",
    "missingness_correlation",
    "detect_duplicate_columns",
    "detect_mixed_types",
    "detect_whitespace_issues",
    "detect_case_variants",
]


@dataclass(frozen=True)
class QualityIssue:
    """One machine-readable data-quality finding."""

    code: str
    severity: Severity
    message: str
    columns: Tuple[str, ...] = ()
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": str(self.severity),
            "message": self.message,
            "columns": list(self.columns),
            "details": _jsonable(self.details),
        }

    def __str__(self) -> str:
        marker = {"info": "i", "warning": "!", "error": "x"}[str(self.severity)]
        return f"[{marker}] {self.message}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def detect_sentinels(
    frame: pd.DataFrame,
    sentinels: Sequence[str],
    columns: Optional[Iterable[str]] = None,
    min_count: int = 1,
) -> Dict[str, Dict[str, int]]:
    """Find placeholder strings standing in for missing values.

    Returns ``{column: {sentinel: count}}`` for every column containing at least
    ``min_count`` occurrences of at least one sentinel.

    Matching is case-insensitive and whitespace-stripped, because ``" NA "``, ``"na"``
    and ``"NA"`` are the same defect.  Only object/string/category columns are scanned;
    numeric sentinels are handled separately by :func:`detect_numeric_sentinels`, which
    only ever reports, because replacing ``-999`` is not safe to do automatically.
    """
    wanted = {str(s).strip().lower() for s in sentinels}
    targets = list(columns) if columns is not None else list(frame.columns)
    found: Dict[str, Dict[str, int]] = {}

    for name in targets:
        if name not in frame.columns:
            continue
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            continue
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            continue

        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = series.cat.categories
            hits = [c for c in categories if str(c).strip().lower() in wanted]
            if not hits:
                continue
            counts = series.value_counts(dropna=True)
            per_sentinel = {
                str(c): int(counts.get(c, 0))
                for c in hits
                if int(counts.get(c, 0)) >= min_count
            }
        else:
            # Count first, then normalise the *distinct* values.  Normalising every
            # cell (`.astype(str).str.strip().str.lower()` over the whole column) costs
            # one Python-level call per row; a 100,000-row column with 500 distinct
            # values does 200x more work than it needs to, and this scan showed up as
            # 21% of profiling time in the benchmark.
            counts = series.value_counts(dropna=True)
            if counts.empty:
                continue
            per_sentinel: Dict[str, int] = {}
            for value, count in counts.items():
                if not isinstance(value, str):
                    continue
                normalised = value.strip().lower()
                if normalised in wanted and int(count) >= min_count:
                    per_sentinel[normalised] = per_sentinel.get(normalised, 0) + int(count)
            per_sentinel = {k: v for k, v in per_sentinel.items() if v >= min_count}

        if per_sentinel:
            found[str(name)] = per_sentinel
    return found


def detect_numeric_sentinels(
    frame: pd.DataFrame,
    sentinels: Sequence[float],
    columns: Optional[Iterable[str]] = None,
    min_fraction: float = 0.001,
) -> Dict[str, Dict[str, int]]:
    """Find suspicious numeric placeholders such as ``-999``.

    Reported, never replaced.  ``-999`` may be a legitimate reading, and the cost of
    getting that wrong silently is high.  A value must additionally sit outside the
    column's central 99% to be reported, so a ``-999`` inside a column that genuinely
    ranges over ``[-2000, 2000]`` does not trigger.
    """
    targets = list(columns) if columns is not None else list(frame.columns)
    found: Dict[str, Dict[str, int]] = {}
    wanted = np.asarray(sentinels, dtype=np.float64)

    for name in targets:
        if name not in frame.columns:
            continue
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            continue
        values = series.to_numpy(dtype="float64", na_value=np.nan, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        per_value: Dict[str, int] = {}
        for candidate in wanted:
            count = int(np.count_nonzero(finite == candidate))
            if count == 0:
                continue
            if count / finite.size < min_fraction:
                continue
            # The reference range must exclude the candidate itself.  A sentinel
            # repeated often enough occupies the tail it is being compared against,
            # so including it makes the column look as if -999 were part of its own
            # normal range and the check silently never fires.
            rest = finite[finite != candidate]
            if rest.size < 2:
                continue
            lo, hi = np.quantile(rest, [0.005, 0.995])
            if lo <= candidate <= hi:
                continue  # inside the bulk of the distribution: probably legitimate
            per_value[str(candidate)] = count
        if per_value:
            found[str(name)] = per_value
    return found


def missingness_correlation(
    frame: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    threshold: float = 0.5,
    max_columns: int = 200,
) -> List[Tuple[str, str, float]]:
    """Pairs of columns whose *missingness patterns* correlate.

    Generalises the ``a census-income notebook`` observation that ``workclass`` and
    ``occupation`` go missing together.  Co-missing columns usually share a cause (a
    failed join, an optional form section), which changes how they should be imputed
    and whether a single shared indicator is enough.

    Returns ``[(col_a, col_b, correlation)]`` sorted by descending absolute
    correlation, restricted to pairs above ``threshold``.
    """
    targets = list(columns if columns is not None else frame.columns)
    mask = frame[targets].isna()
    # Columns with no variation in missingness carry no information and would produce
    # NaN correlations.
    varying = [c for c in targets if 0 < int(mask[c].sum()) < len(frame)]
    if len(varying) < 2:
        return []
    if len(varying) > max_columns:
        counts = mask[varying].sum().sort_values(ascending=False)
        varying = list(counts.head(max_columns).index)

    indicator = mask[varying].astype(np.int8)
    corr = np.corrcoef(indicator.to_numpy(dtype=np.float64), rowvar=False)
    corr = np.atleast_2d(corr)

    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(varying)):
        for j in range(i + 1, len(varying)):
            value = corr[i, j]
            if np.isfinite(value) and abs(value) >= threshold:
                pairs.append((str(varying[i]), str(varying[j]), float(value)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs


def detect_duplicate_columns(
    frame: pd.DataFrame, columns: Optional[Iterable[str]] = None
) -> List[List[str]]:
    """Groups of columns holding identical values.

    Hashes each column once and only compares within hash buckets, so the cost is
    O(n * p) rather than the O(n * p^2) of pairwise comparison.  NaN positions must
    match too, which ``pandas.util.hash_pandas_object`` handles consistently.
    """
    targets = list(columns if columns is not None else frame.columns)
    if len(targets) < 2:
        return []

    buckets: Dict[Any, List[str]] = {}
    for name in targets:
        series = frame[name]
        try:
            digest = int(pd.util.hash_pandas_object(series, index=False).sum())
        except TypeError:  # unhashable objects (lists, dicts) in the column
            digest = ("unhashable", str(series.dtype), len(series))
        buckets.setdefault(digest, []).append(str(name))

    groups: List[List[str]] = []
    for names in buckets.values():
        if len(names) < 2:
            continue
        remaining = list(names)
        while remaining:
            head = remaining.pop(0)
            group = [head]
            still: List[str] = []
            for other in remaining:
                if frame[head].equals(frame[other]):
                    group.append(other)
                else:
                    still.append(other)
            remaining = still
            if len(group) > 1:
                groups.append(group)
    return groups


def detect_mixed_types(
    frame: pd.DataFrame, columns: Optional[Iterable[str]] = None, sample_size: int = 1000
) -> Dict[str, List[str]]:
    """Object columns holding more than one Python type.

    A column mixing ``str`` and ``float`` usually means a parse failure upstream, and
    it silently breaks sorting, grouping and comparison.
    """
    targets = list(columns if columns is not None else frame.columns)
    out: Dict[str, List[str]] = {}
    for name in targets:
        series = frame[name]
        if series.dtype != object:
            continue
        sample = series.dropna()
        if len(sample) > sample_size:
            sample = sample.iloc[:sample_size]
        kinds = {type(v).__name__ for v in sample}
        if len(kinds) > 1:
            out[str(name)] = sorted(kinds)
    return out


def detect_whitespace_issues(
    frame: pd.DataFrame, columns: Optional[Iterable[str]] = None
) -> Dict[str, int]:
    """String columns with leading/trailing whitespace.

    ``"Yes"`` and ``"Yes "`` become two categories, inflating cardinality and creating
    a category the model will never see again.
    """
    targets = list(columns if columns is not None else frame.columns)
    out: Dict[str, int] = {}
    for name in targets:
        series = frame[name]
        if not (series.dtype == object or isinstance(series.dtype, pd.StringDtype)):
            continue
        # As in detect_sentinels: work on the distinct values and weight by their
        # counts, rather than stripping every cell in the column.
        counts = series.value_counts(dropna=True)
        if counts.empty:
            continue
        n_bad = 0
        for value, count in counts.items():
            if isinstance(value, str) and value != value.strip():
                n_bad += int(count)
        if n_bad:
            out[str(name)] = n_bad
    return out


def detect_case_variants(
    frame: pd.DataFrame, columns: Optional[Iterable[str]] = None, max_cardinality: int = 5000
) -> Dict[str, List[List[str]]]:
    """Categories that differ only by case: ``["USA", "usa", "Usa"]``."""
    targets = list(columns if columns is not None else frame.columns)
    out: Dict[str, List[List[str]]] = {}
    for name in targets:
        series = frame[name]
        if not (
            series.dtype == object
            or isinstance(series.dtype, (pd.StringDtype, pd.CategoricalDtype))
        ):
            continue
        try:
            uniques = pd.Series(series.dropna().unique())
        except TypeError:
            continue  # unhashable cell values; case folding is meaningless here
        if uniques.empty or len(uniques) > max_cardinality:
            continue
        as_str = uniques.astype(str)
        folded = as_str.str.strip().str.lower()
        groups: Dict[str, List[str]] = {}
        for original, key in zip(as_str, folded):
            groups.setdefault(key, []).append(original)
        collisions = [sorted(v) for v in groups.values() if len(v) > 1]
        if collisions:
            out[str(name)] = collisions
    return out
