"""Dataset profiling: measurement only, no decisions.

``DatasetProfile`` is the sole input the planner receives about the data.  Keeping it a
frozen, serialisable value with no behaviour is what makes the planner testable without
fixtures and incapable of leaking: it cannot reach past the profile to the frame.

Cost control
------------
Cheap statistics (dtype, null count, min/max) run on the full frame -- they are single
vectorised passes.  Expensive statistics (moments, quantiles, distinct counts,
correlation) switch to a deterministic sample once the frame crosses
``Thresholds.sampling_row_threshold``.  Whether sampling happened, and with which seed,
is recorded on the profile, so no statistic is ever silently approximate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._version import __version__
from ..config import Config, Thresholds
from ..exceptions import EmptyDataError
from ..types import CATEGORICAL_LIKE, NUMERIC_LIKE, SemanticType, Severity
from . import quality as quality_mod
from .column_types import TypeInference, infer_semantic_type
from .statistics import NumericStats, estimate_memory, numeric_block_stats, top_categories

__all__ = ["ColumnProfile", "DatasetProfile", "profile"]


@dataclass(frozen=True)
class ColumnProfile:
    """Everything measured about one column."""

    name: str
    dtype: str
    semantic: SemanticType
    semantic_confidence: float
    semantic_alternatives: Tuple[SemanticType, ...]
    semantic_reasons: Tuple[str, ...]
    suggested_dtype: Optional[str]

    n_rows: int
    n_missing: int
    n_unique: int
    memory_bytes: int

    numeric: Optional[NumericStats] = None
    top_values: Tuple[Tuple[Any, int], ...] = ()
    modal_value: Any = None
    modal_frequency: float = 0.0

    is_constant: bool = False
    is_near_constant: bool = False
    is_possible_id: bool = False
    is_target: bool = False

    #: Absolute association with the target, when a target was supplied.  Pearson for
    #: numeric/numeric, correlation ratio (eta) for categorical/numeric, Cramer's V for
    #: categorical/categorical.  ``None`` when not computed.
    target_association: Optional[float] = None
    target_association_kind: Optional[str] = None

    @property
    def missing_fraction(self) -> float:
        return (self.n_missing / self.n_rows) if self.n_rows else 0.0

    @property
    def unique_ratio(self) -> float:
        present = self.n_rows - self.n_missing
        return (self.n_unique / present) if present else 0.0

    @property
    def is_numeric_like(self) -> bool:
        return self.semantic in NUMERIC_LIKE

    @property
    def is_categorical_like(self) -> bool:
        return self.semantic in CATEGORICAL_LIKE

    @property
    def skew(self) -> float:
        return self.numeric.skew if self.numeric else float("nan")

    @property
    def has_negative(self) -> bool:
        return bool(self.numeric and self.numeric.n_negative > 0)

    @property
    def has_zero(self) -> bool:
        return bool(self.numeric and self.numeric.n_zeros > 0)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "semantic": str(self.semantic),
            "semantic_confidence": round(self.semantic_confidence, 3),
            "semantic_alternatives": [str(a) for a in self.semantic_alternatives],
            "semantic_reasons": list(self.semantic_reasons),
            "suggested_dtype": self.suggested_dtype,
            "n_rows": self.n_rows,
            "n_missing": self.n_missing,
            "missing_fraction": round(self.missing_fraction, 6),
            "n_unique": self.n_unique,
            "unique_ratio": round(self.unique_ratio, 6),
            "memory_bytes": self.memory_bytes,
            "is_constant": self.is_constant,
            "is_near_constant": self.is_near_constant,
            "is_possible_id": self.is_possible_id,
            "is_target": self.is_target,
            "modal_value": quality_mod._jsonable(self.modal_value),
            "modal_frequency": round(self.modal_frequency, 6),
            "top_values": [
                [quality_mod._jsonable(v), c] for v, c in self.top_values
            ],
        }
        if self.numeric is not None:
            out["numeric"] = self.numeric.to_dict()
        if self.target_association is not None:
            out["target_association"] = round(self.target_association, 6)
            out["target_association_kind"] = self.target_association_kind
        return out

    def summary_line(self) -> str:
        bits = [f"{self.name:<28.28}", f"{str(self.semantic):<12}", f"{self.dtype:<12.12}"]
        bits.append(f"miss {self.missing_fraction:>6.1%}")
        bits.append(f"uniq {self.n_unique:>8,}")
        if self.numeric is not None and np.isfinite(self.numeric.skew):
            bits.append(f"skew {self.numeric.skew:>7.2f}")
        if self.semantic_confidence < 0.70:
            bits.append(f"(conf {self.semantic_confidence:.2f})")
        return "  ".join(bits)


@dataclass(frozen=True)
class DatasetProfile:
    """Measurements for a whole dataset."""

    n_rows: int
    n_columns: int
    memory_bytes: int
    columns: Dict[str, ColumnProfile]
    column_order: Tuple[str, ...]

    n_duplicate_rows: int = 0
    duplicate_row_fraction: float = 0.0
    total_missing_cells: int = 0

    target: Optional[str] = None
    target_kind: Optional[str] = None  # "classification" | "regression" | None
    target_classes: Optional[int] = None
    target_imbalance_ratio: Optional[float] = None

    sentinels: Dict[str, Dict[str, int]] = field(default_factory=dict)
    numeric_sentinels: Dict[str, Dict[str, int]] = field(default_factory=dict)
    duplicate_columns: Tuple[Tuple[str, ...], ...] = ()
    comissing_pairs: Tuple[Tuple[str, str, float], ...] = ()
    mixed_type_columns: Dict[str, List[str]] = field(default_factory=dict)
    whitespace_columns: Dict[str, int] = field(default_factory=dict)
    case_variant_columns: Dict[str, List[List[str]]] = field(default_factory=dict)

    issues: Tuple[quality_mod.QualityIssue, ...] = ()
    sampling: Dict[str, Any] = field(default_factory=dict)
    edaprep_version: str = __version__

    # -- convenience views -------------------------------------------------------

    def __getitem__(self, name: str) -> ColumnProfile:
        return self.columns[name]

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def __iter__(self):
        return iter(self.column_order)

    def of_type(self, *types: SemanticType) -> List[str]:
        wanted = {SemanticType.coerce(t) for t in types}
        return [c for c in self.column_order if self.columns[c].semantic in wanted]

    @property
    def numeric_columns(self) -> List[str]:
        return [c for c in self.column_order if self.columns[c].is_numeric_like]

    @property
    def categorical_columns(self) -> List[str]:
        return [c for c in self.column_order if self.columns[c].is_categorical_like]

    @property
    def datetime_columns(self) -> List[str]:
        return self.of_type(SemanticType.DATETIME)

    @property
    def text_columns(self) -> List[str]:
        return self.of_type(SemanticType.TEXT)

    @property
    def identifier_columns(self) -> List[str]:
        return self.of_type(SemanticType.IDENTIFIER)

    @property
    def constant_columns(self) -> List[str]:
        return [c for c in self.column_order if self.columns[c].is_constant]

    @property
    def feature_columns(self) -> List[str]:
        return [c for c in self.column_order if c != self.target]

    @property
    def uncertain_columns(self) -> List[str]:
        return [c for c in self.column_order if self.columns[c].semantic_confidence < 0.70]

    @property
    def missing_fraction(self) -> float:
        cells = self.n_rows * self.n_columns
        return (self.total_missing_cells / cells) if cells else 0.0

    def issues_of(self, severity: Severity) -> List[quality_mod.QualityIssue]:
        target = Severity.coerce(severity)
        return [i for i in self.issues if i.severity is target]

    # -- serialisation -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edaprep_version": self.edaprep_version,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "memory_bytes": self.memory_bytes,
            "n_duplicate_rows": self.n_duplicate_rows,
            "duplicate_row_fraction": round(self.duplicate_row_fraction, 6),
            "total_missing_cells": self.total_missing_cells,
            "missing_fraction": round(self.missing_fraction, 6),
            "target": self.target,
            "target_kind": self.target_kind,
            "target_classes": self.target_classes,
            "target_imbalance_ratio": (
                round(self.target_imbalance_ratio, 6)
                if self.target_imbalance_ratio is not None
                else None
            ),
            "sampling": self.sampling,
            "sentinels": self.sentinels,
            "numeric_sentinels": self.numeric_sentinels,
            "duplicate_columns": [list(g) for g in self.duplicate_columns],
            "comissing_pairs": [[a, b, round(c, 4)] for a, b, c in self.comissing_pairs],
            "mixed_type_columns": self.mixed_type_columns,
            "whitespace_columns": self.whitespace_columns,
            "case_variant_columns": self.case_variant_columns,
            "issues": [i.to_dict() for i in self.issues],
            "columns": {name: self.columns[name].to_dict() for name in self.column_order},
        }

    def summary(self, max_columns: int = 60) -> str:
        """Human-readable overview."""
        lines: List[str] = []
        lines.append("Dataset")
        lines.append(f"  {self.n_rows:,} rows x {self.n_columns:,} columns")
        lines.append(f"  {_human_bytes(self.memory_bytes)} in memory")
        if self.n_duplicate_rows:
            lines.append(
                f"  {self.n_duplicate_rows:,} duplicate rows "
                f"({self.duplicate_row_fraction:.1%})"
            )
        lines.append(
            f"  {self.total_missing_cells:,} missing cells ({self.missing_fraction:.2%})"
        )
        if self.sampling.get("used"):
            lines.append(
                f"  expensive statistics sampled: {self.sampling['n']:,} of "
                f"{self.sampling['of']:,} rows (random_state="
                f"{self.sampling.get('random_state')})"
            )
        if self.target:
            kind = self.target_kind or "unknown"
            extra = ""
            if self.target_classes:
                extra = f", {self.target_classes} classes"
            if self.target_imbalance_ratio is not None:
                extra += f", minority/majority ratio {self.target_imbalance_ratio:.3f}"
            lines.append(f"  target: {self.target} ({kind}{extra})")

        counts: Dict[str, int] = {}
        for name in self.column_order:
            key = str(self.columns[name].semantic)
            counts[key] = counts.get(key, 0) + 1
        lines.append("")
        lines.append("Semantic types")
        for key in sorted(counts, key=lambda k: -counts[k]):
            lines.append(f"  {key:<14} {counts[key]:>4}")

        lines.append("")
        lines.append("Columns")
        for name in self.column_order[:max_columns]:
            lines.append("  " + self.columns[name].summary_line())
        if len(self.column_order) > max_columns:
            lines.append(f"  ... {len(self.column_order) - max_columns} more")

        if self.issues:
            lines.append("")
            lines.append("Data-quality findings")
            for issue in sorted(self.issues, key=lambda i: -i.severity.rank):
                lines.append("  " + str(issue))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DatasetProfile(n_rows={self.n_rows:,}, n_columns={self.n_columns}, "
            f"target={self.target!r}, issues={len(self.issues)})"
        )


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value):,} B"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover


def _make_sample(
    frame: pd.DataFrame, thresholds: Thresholds, config: Config
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Deterministic sample for expensive statistics."""
    n_rows = len(frame)
    limit = config.effective_sample_size
    if n_rows <= thresholds.sampling_row_threshold or n_rows <= limit:
        return frame, {"used": False, "n": n_rows, "of": n_rows}
    seed = config.random_state if config.random_state is not None else 0
    sample = frame.sample(n=limit, random_state=seed)
    return sample, {
        "used": True,
        "n": limit,
        "of": n_rows,
        "random_state": seed,
        "statistics": [
            "moments",
            "quantiles",
            "distinct counts",
            "modal frequency",
            "target association",
        ],
    }


def _target_kind(series: pd.Series, thresholds: Thresholds) -> Tuple[str, Optional[int], Optional[float]]:
    """Classify the target and, for classification, measure imbalance."""
    clean = series.dropna()
    if clean.empty:
        return "unknown", None, None
    n_unique = int(clean.nunique())
    numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    )
    if numeric and n_unique > thresholds.classification_max_classes:
        return "regression", None, None
    counts = clean.value_counts()
    ratio = float(counts.min() / counts.max()) if len(counts) > 1 else 1.0
    return "classification", n_unique, ratio


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: the association between a categorical grouping and a numeric variable.

    ``eta^2`` is the fraction of the numeric variance explained by group membership, so
    ``eta`` is directly comparable in magnitude to a Pearson correlation.  Using it,
    instead of the usual habit of eyeballing a boxplot, gives categorical columns a
    target association on the same scale as numeric ones.
    """
    mask = categories.notna() & values.notna()
    if mask.sum() < 2:
        return float("nan")
    cats = categories[mask]
    vals = values[mask].astype("float64")
    total_var = float(vals.var(ddof=0))
    if not np.isfinite(total_var) or total_var == 0.0:
        return 0.0
    group_means = vals.groupby(cats, observed=True).mean()
    group_sizes = vals.groupby(cats, observed=True).size()
    grand_mean = float(vals.mean())
    between = float((group_sizes * (group_means - grand_mean) ** 2).sum() / len(vals))
    eta_sq = between / total_var
    return float(np.sqrt(max(0.0, min(1.0, eta_sq))))


def _target_kind_is_categorical(target_kind: Optional[str]) -> bool:
    return target_kind == "classification"


def _correlation_ratio_batch(
    target: pd.Series, values: pd.DataFrame
) -> Dict[str, float]:
    """Eta for every numeric column against one categorical target, in k passes.

    The per-column :func:`_correlation_ratio` runs a pandas ``groupby`` per feature.
    On a 300-column frame that is 300 groupbys over the same grouping, and it measured
    as 36% of total profiling time.  Since the grouping never changes, the sums and
    counts for all columns can be accumulated with one masked pass per *class* --
    typically 2 to 20 passes rather than one per column.

    Returns ``{column: eta}``.  Identical to the per-column function to floating-point
    tolerance, which the tests assert.
    """
    names = [str(c) for c in values.columns]
    if not names:
        return {}
    codes, levels = pd.factorize(target, use_na_sentinel=True)
    n_levels = len(levels)
    if n_levels < 2 or len(values) == 0:
        return {name: 0.0 for name in names}

    # Materialise in column chunks: the whole frame as one float64 array is 48 MiB on
    # a 20,000 x 300 input, and each per-class mask copies a slice of it again.  A
    # 16 MiB budget keeps the working set bounded without changing the arithmetic.
    per_column_bytes = max(len(values) * 8, 1)
    chunk = max(1, min(len(names), (16 * 1024 * 1024) // per_column_bytes))
    if chunk < len(names):
        out: Dict[str, float] = {}
        for start in range(0, len(names), chunk):
            out.update(
                _correlation_ratio_batch(target, values.iloc[:, start : start + chunk])
            )
        return out

    matrix = values.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    width = matrix.shape[1]

    sums = np.zeros((n_levels, width), dtype=np.float64)
    counts = np.zeros((n_levels, width), dtype=np.float64)
    with np.errstate(invalid="ignore"):
        for level in range(n_levels):
            rows = matrix[codes == level]
            if rows.size == 0:
                continue
            present = ~np.isnan(rows)
            counts[level] = present.sum(axis=0)
            sums[level] = np.where(present, rows, 0.0).sum(axis=0)

        total = counts.sum(axis=0)
        with np.errstate(divide="ignore"):
            grand_mean = np.where(total > 0, sums.sum(axis=0) / total, np.nan)
            group_means = np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), 0.0)

        deviations = np.where(counts > 0, (group_means - grand_mean) ** 2, 0.0)
        between = (counts * deviations).sum(axis=0) / np.where(total > 0, total, 1.0)

        # Population variance over the rows that have a non-missing target, which is
        # the same denominator the per-column version uses.
        usable = matrix[codes >= 0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            total_var = np.nanvar(usable, axis=0)

        eta_sq = np.where(total_var > 0, between / np.where(total_var > 0, total_var, 1.0), 0.0)
        eta = np.sqrt(np.clip(eta_sq, 0.0, 1.0))
    return {name: float(eta[i]) for i, name in enumerate(names)}


def _cramers_v(a: pd.Series, b: pd.Series, max_cells: int = 1_000_000) -> float:
    """Cramer's V with the Bergsma-Wicher bias correction.

    The correction matters: on a 2 x 50 table with modest n, the uncorrected statistic
    is inflated enough to make an unrelated high-cardinality column look predictive,
    which is precisely the false-positive the usual raw correlation heatmaps invite.
    """
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan")
    x, y = a[mask], b[mask]
    # factorize + bincount rather than pd.crosstab: crosstab routes through
    # pivot_table, which sorts, groups and builds a labelled frame we immediately throw
    # away.  This was 17% of profiling time in the benchmark on a frame with one
    # 500-level column.  The result is the same contingency table.
    x_codes, x_levels = pd.factorize(x, use_na_sentinel=True)
    y_codes, y_levels = pd.factorize(y, use_na_sentinel=True)
    n_x, n_y = len(x_levels), len(y_levels)
    if n_x * n_y > max_cells or n_x < 2 or n_y < 2:
        return float("nan")
    flat = x_codes.astype(np.int64) * n_y + y_codes.astype(np.int64)
    table = np.bincount(flat, minlength=n_x * n_y).reshape(n_x, n_y).astype(np.float64)
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def _target_association(
    column: pd.Series,
    column_semantic: SemanticType,
    target: pd.Series,
    target_kind: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Association between one feature and the target, on a 0-1 scale."""
    col_numeric = column_semantic in NUMERIC_LIKE or column_semantic is SemanticType.BINARY
    tgt_numeric = target_kind == "regression"

    try:
        if col_numeric and tgt_numeric:
            x = pd.to_numeric(column, errors="coerce")
            y = pd.to_numeric(target, errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                return None, None
            xv, yv = x[mask].to_numpy(), y[mask].to_numpy()
            if np.std(xv) == 0 or np.std(yv) == 0:
                return 0.0, "pearson"
            return float(abs(np.corrcoef(xv, yv)[0, 1])), "pearson"
        if col_numeric and not tgt_numeric:
            return _correlation_ratio(target, pd.to_numeric(column, errors="coerce")), "eta"
        if not col_numeric and tgt_numeric:
            return _correlation_ratio(column, pd.to_numeric(target, errors="coerce")), "eta"
        return _cramers_v(column, target), "cramers_v"
    except (TypeError, ValueError):
        return None, None


def profile(
    data: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[Config] = None,
    *,
    compute_moments: bool = True,
    check_duplicates: bool = True,
    check_quality: bool = True,
    compute_target_association: bool = True,
    deep_memory: bool = True,
) -> DatasetProfile:
    """Measure a dataset.

    Parameters
    ----------
    data :
        The frame to profile.  It is never mutated.
    target :
        Name of the target column, if any.  Targets are excluded from identifier
        detection and gain a class-balance measurement.
    config :
        Thresholds and hints.  Per-column ``semantic_type`` overrides are honoured.
    compute_moments :
        Set ``False`` to skip skewness/kurtosis (the "quick" analysis level).
    check_duplicates :
        Duplicate-row detection hashes the whole frame; ``False`` skips it.
    check_quality :
        Sentinel/whitespace/mixed-type scans.
    compute_target_association :
        Per-feature association with the target.  O(columns), each cheap, but it is the
        main cost on very wide frames.
    deep_memory :
        Follow object pointers when measuring memory.  Accurate but O(n_objects).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"profile() expects a pandas DataFrame, got {type(data).__name__}. "
            f"If you have a Series, call .to_frame() first."
        )
    config = config or Config()
    thresholds = config.thresholds
    n_rows, n_columns = data.shape

    if n_columns == 0:
        raise EmptyDataError.no_columns("profile a dataset")
    if target is not None and target not in data.columns:
        available = ", ".join(repr(c) for c in list(data.columns)[:10])
        raise KeyError(
            f"target={target!r} is not a column of the dataset. Available columns "
            f"include: {available}."
        )

    sample, sampling = _make_sample(data, thresholds, config)
    memory = estimate_memory(data, deep=deep_memory and n_rows * n_columns < 50_000_000)

    # --- distinct counts and modal frequency (sampled) --------------------------
    n_unique: Dict[str, int] = {}
    modal: Dict[str, Tuple[Any, float]] = {}
    for name in data.columns:
        key = str(name)
        series = sample[name]
        try:
            n_unique[key] = int(series.nunique(dropna=True))
        except TypeError:  # unhashable values
            n_unique[key] = int(series.astype(str).nunique(dropna=True))
        present = int(series.notna().sum())
        if present:
            try:
                counts = series.value_counts(dropna=True)
                if len(counts):
                    modal[key] = (counts.index[0], float(counts.iloc[0]) / present)
            except TypeError:
                modal[key] = (None, 0.0)

    # --- semantic inference ------------------------------------------------------
    inferences: Dict[str, TypeInference] = {}
    for name in data.columns:
        key = str(name)
        override = config.get_column(key)
        hint = override.semantic_type if override is not None else None
        inferences[key] = infer_semantic_type(
            sample[name],
            n_unique=n_unique[key],
            thresholds=thresholds,
            is_target=(key == target),
            hint=hint,
            random_state=config.random_state,
        )

    # --- numeric statistics, one block pass -------------------------------------
    numeric_names = [
        str(c)
        for c in data.columns
        if inferences[str(c)].semantic
        in (SemanticType.NUMERIC, SemanticType.ORDINAL, SemanticType.BINARY)
        and pd.api.types.is_numeric_dtype(data[c].dtype)
    ]
    numeric_stats: Dict[str, NumericStats] = {}
    if numeric_names:
        numeric_stats = numeric_block_stats(
            sample[numeric_names], compute_moments=compute_moments
        )
        if sampling["used"]:
            # Missing counts must be exact even when the rest is sampled: they drive
            # drop/impute decisions, and an approximate 60% is not good enough.
            exact_missing = data[numeric_names].isna().sum()
            for name in numeric_names:
                stats = numeric_stats[name]
                numeric_stats[name] = NumericStats(
                    **{
                        **stats.__dict__,
                        "count": int(n_rows - exact_missing[name]),
                        "n_missing": int(exact_missing[name]),
                    }
                )

    exact_missing_all = data.isna().sum()

    # --- batched target association for numeric features -------------------------
    # Against a categorical target every numeric column shares the same grouping, so
    # they are computed together rather than one groupby per column.
    batched_eta: Dict[str, float] = {}

    # --- target ------------------------------------------------------------------
    target_kind: Optional[str] = None
    target_classes: Optional[int] = None
    imbalance: Optional[float] = None
    if target is not None:
        target_kind, target_classes, imbalance = _target_kind(data[target], thresholds)

    # --- per-column assembly -----------------------------------------------------
    if (
        compute_target_association
        and target is not None
        and numeric_names
        and _target_kind_is_categorical(target_kind)
    ):
        eligible = [
            n
            for n in numeric_names
            if n != target and not inferences[n].semantic is SemanticType.CONSTANT
        ]
        if eligible:
            batched_eta = _correlation_ratio_batch(sample[target], sample[eligible])

    columns: Dict[str, ColumnProfile] = {}
    for name in data.columns:
        key = str(name)
        inf = inferences[key]
        missing = int(exact_missing_all[name])
        present = n_rows - missing
        modal_value, modal_freq = modal.get(key, (None, 0.0))
        is_constant = inf.semantic is SemanticType.CONSTANT or n_unique[key] <= 1
        is_near_constant = (
            not is_constant and modal_freq >= thresholds.near_constant_ratio and present > 0
        )
        unique_ratio = (n_unique[key] / present) if present else 0.0

        association: Optional[float] = None
        association_kind: Optional[str] = None
        if (
            compute_target_association
            and target is not None
            and key != target
            and target_kind in ("classification", "regression")
            and not is_constant
            and inf.semantic
            not in (SemanticType.TEXT, SemanticType.IDENTIFIER, SemanticType.CONSTANT)
        ):
            if key in batched_eta:
                association, association_kind = batched_eta[key], "eta"
            else:
                association, association_kind = _target_association(
                    sample[name], inf.semantic, sample[target], target_kind
                )
            if association is not None and not np.isfinite(association):
                association, association_kind = None, None

        columns[key] = ColumnProfile(
            name=key,
            dtype=str(data[name].dtype),
            semantic=inf.semantic,
            semantic_confidence=inf.confidence,
            semantic_alternatives=tuple(inf.alternatives),
            semantic_reasons=tuple(inf.reasons),
            suggested_dtype=inf.suggested_dtype,
            n_rows=n_rows,
            n_missing=missing,
            n_unique=n_unique[key],
            memory_bytes=memory.get(key, 0),
            numeric=numeric_stats.get(key),
            top_values=tuple(top_categories(sample[name], k=10))
            if inf.semantic
            in (SemanticType.CATEGORICAL, SemanticType.BINARY, SemanticType.ORDINAL)
            else (),
            modal_value=modal_value,
            modal_frequency=modal_freq,
            is_constant=is_constant,
            is_near_constant=is_near_constant,
            # Identifier status comes from the semantic inference, which already
            # requires integrality or non-numeric values.  A bare unique-ratio test
            # would flag every continuous float column, since distinct floats are the
            # normal case rather than a sign of a row key.
            is_possible_id=inf.semantic is SemanticType.IDENTIFIER,
            is_target=(key == target),
            target_association=association,
            target_association_kind=association_kind,
        )

    # --- dataset-level checks -----------------------------------------------------
    n_duplicate_rows = 0
    if check_duplicates and n_rows:
        try:
            n_duplicate_rows = int(data.duplicated().sum())
        except TypeError:
            n_duplicate_rows = int(data.astype(str).duplicated().sum())

    sentinels: Dict[str, Dict[str, int]] = {}
    numeric_sentinels: Dict[str, Dict[str, int]] = {}
    duplicate_columns: List[List[str]] = []
    comissing: List[Tuple[str, str, float]] = []
    mixed_types: Dict[str, List[str]] = {}
    whitespace: Dict[str, int] = {}
    case_variants: Dict[str, List[List[str]]] = {}

    if check_quality:
        if config.detect_sentinels:
            sentinels = quality_mod.detect_sentinels(sample, config.sentinels)
            numeric_sentinels = quality_mod.detect_numeric_sentinels(
                sample, config.numeric_sentinels
            )
        duplicate_columns = quality_mod.detect_duplicate_columns(sample)
        comissing = quality_mod.missingness_correlation(data)
        mixed_types = quality_mod.detect_mixed_types(sample)
        whitespace = quality_mod.detect_whitespace_issues(sample)
        case_variants = quality_mod.detect_case_variants(sample)

    issues = _build_issues(
        columns=columns,
        column_order=[str(c) for c in data.columns],
        n_rows=n_rows,
        thresholds=thresholds,
        target=target,
        target_kind=target_kind,
        imbalance=imbalance,
        n_duplicate_rows=n_duplicate_rows,
        sentinels=sentinels,
        numeric_sentinels=numeric_sentinels,
        duplicate_columns=duplicate_columns,
        comissing=comissing,
        mixed_types=mixed_types,
        whitespace=whitespace,
        case_variants=case_variants,
    )

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_columns,
        memory_bytes=memory["total"],
        columns=columns,
        column_order=tuple(str(c) for c in data.columns),
        n_duplicate_rows=n_duplicate_rows,
        duplicate_row_fraction=(n_duplicate_rows / n_rows) if n_rows else 0.0,
        total_missing_cells=int(exact_missing_all.sum()),
        target=target,
        target_kind=target_kind,
        target_classes=target_classes,
        target_imbalance_ratio=imbalance,
        sentinels=sentinels,
        numeric_sentinels=numeric_sentinels,
        duplicate_columns=tuple(tuple(g) for g in duplicate_columns),
        comissing_pairs=tuple(comissing),
        mixed_type_columns=mixed_types,
        whitespace_columns=whitespace,
        case_variant_columns=case_variants,
        issues=tuple(issues),
        sampling=sampling,
    )


def _build_issues(
    *,
    columns: Dict[str, ColumnProfile],
    column_order: Sequence[str],
    n_rows: int,
    thresholds: Thresholds,
    target: Optional[str],
    target_kind: Optional[str],
    imbalance: Optional[float],
    n_duplicate_rows: int,
    sentinels: Dict[str, Dict[str, int]],
    numeric_sentinels: Dict[str, Dict[str, int]],
    duplicate_columns: Sequence[Sequence[str]],
    comissing: Sequence[Tuple[str, str, float]],
    mixed_types: Dict[str, List[str]],
    whitespace: Dict[str, int],
    case_variants: Dict[str, List[List[str]]],
) -> List[quality_mod.QualityIssue]:
    Issue = quality_mod.QualityIssue
    issues: List[Issue] = []

    if n_rows == 0:
        issues.append(
            Issue("empty_dataset", Severity.ERROR, "The dataset has 0 rows.")
        )

    constants = [c for c in column_order if columns[c].is_constant]
    if constants:
        issues.append(
            Issue(
                "constant_columns",
                Severity.WARNING,
                f"{len(constants)} constant column(s) carry no information: "
                f"{_names(constants)}.",
                tuple(constants),
                {"columns": constants},
            )
        )

    near_constants = [c for c in column_order if columns[c].is_near_constant]
    if near_constants:
        issues.append(
            Issue(
                "near_constant_columns",
                Severity.INFO,
                f"{len(near_constants)} near-constant column(s) (a single value covers "
                f">={thresholds.near_constant_ratio:.0%} of rows): "
                f"{_names(near_constants)}.",
                tuple(near_constants),
                {
                    "columns": {
                        c: round(columns[c].modal_frequency, 4) for c in near_constants
                    }
                },
            )
        )

    high_missing = [
        c
        for c in column_order
        if columns[c].missing_fraction >= thresholds.missing_drop_threshold
    ]
    if high_missing:
        issues.append(
            Issue(
                "high_missingness",
                Severity.WARNING,
                f"{len(high_missing)} column(s) exceed "
                f"{thresholds.missing_drop_threshold:.0%} missing values; imputing them "
                f"would invent most of the column: {_names(high_missing)}.",
                tuple(high_missing),
                {c: round(columns[c].missing_fraction, 4) for c in high_missing},
            )
        )

    ids = [c for c in column_order if columns[c].is_possible_id and c != target]
    if ids:
        issues.append(
            Issue(
                "identifier_columns",
                Severity.INFO,
                f"{len(ids)} likely identifier column(s) detected; identifiers carry no "
                f"generalisable signal and are excluded from features by default: "
                f"{_names(ids)}.",
                tuple(ids),
                {c: round(columns[c].unique_ratio, 4) for c in ids},
            )
        )

    uncertain = [c for c in column_order if columns[c].semantic_confidence < 0.70]
    if uncertain:
        detail = {
            c: {
                "inferred": str(columns[c].semantic),
                "confidence": round(columns[c].semantic_confidence, 3),
                "alternatives": [str(a) for a in columns[c].semantic_alternatives],
            }
            for c in uncertain
        }
        issues.append(
            Issue(
                "uncertain_semantic_type",
                Severity.WARNING,
                f"{len(uncertain)} column(s) have an uncertain semantic type; review "
                f"them and set config.column(name).semantic_type if the inference is "
                f"wrong: {_names(uncertain)}.",
                tuple(uncertain),
                detail,
            )
        )

    if n_duplicate_rows:
        issues.append(
            Issue(
                "duplicate_rows",
                Severity.WARNING if n_duplicate_rows / max(n_rows, 1) > 0.01 else Severity.INFO,
                f"{n_duplicate_rows:,} duplicate row(s) "
                f"({n_duplicate_rows / max(n_rows, 1):.2%}). Duplicates are not always "
                f"errors: repeated observations are legitimate in transactional data. "
                f"Set Config(duplicate_strategy='remove') to drop them.",
                (),
                {"count": n_duplicate_rows},
            )
        )

    if sentinels:
        issues.append(
            Issue(
                "sentinel_values",
                Severity.WARNING,
                f"{len(sentinels)} column(s) contain placeholder strings that most "
                f"likely mean 'missing' but are not recognised as NaN: "
                f"{_names(list(sentinels))}.",
                tuple(sentinels),
                sentinels,
            )
        )

    if numeric_sentinels:
        issues.append(
            Issue(
                "numeric_sentinel_values",
                Severity.WARNING,
                f"{len(numeric_sentinels)} column(s) contain suspicious numeric "
                f"placeholders (e.g. -999) far outside their own distribution. These "
                f"are reported but never replaced automatically, because the value may "
                f"be legitimate: {_names(list(numeric_sentinels))}.",
                tuple(numeric_sentinels),
                numeric_sentinels,
            )
        )

    if duplicate_columns:
        issues.append(
            Issue(
                "duplicate_columns",
                Severity.WARNING,
                f"{len(duplicate_columns)} group(s) of identical columns: "
                + "; ".join("=".join(g) for g in duplicate_columns),
                tuple(c for g in duplicate_columns for c in g),
                {"groups": [list(g) for g in duplicate_columns]},
            )
        )

    if comissing:
        top = comissing[:5]
        issues.append(
            Issue(
                "correlated_missingness",
                Severity.INFO,
                f"{len(comissing)} column pair(s) go missing together, which usually "
                f"means a shared cause: "
                + ", ".join(f"{a}~{b} ({c:.2f})" for a, b, c in top),
                tuple({c for pair in comissing for c in pair[:2]}),
                {"pairs": [[a, b, round(c, 4)] for a, b, c in comissing]},
            )
        )

    if mixed_types:
        issues.append(
            Issue(
                "mixed_types",
                Severity.WARNING,
                f"{len(mixed_types)} object column(s) hold more than one Python type, "
                f"which usually indicates a parse failure upstream: "
                f"{_names(list(mixed_types))}.",
                tuple(mixed_types),
                mixed_types,
            )
        )

    if whitespace:
        issues.append(
            Issue(
                "whitespace",
                Severity.WARNING,
                f"{len(whitespace)} column(s) contain values with leading or trailing "
                f"whitespace, which splits one category into several: "
                f"{_names(list(whitespace))}.",
                tuple(whitespace),
                whitespace,
            )
        )

    if case_variants:
        issues.append(
            Issue(
                "case_variants",
                Severity.INFO,
                f"{len(case_variants)} column(s) contain categories differing only by "
                f"case: {_names(list(case_variants))}.",
                tuple(case_variants),
                case_variants,
            )
        )

    if target is not None:
        if columns[target].n_missing:
            issues.append(
                Issue(
                    "missing_target",
                    Severity.ERROR,
                    f"The target {target!r} has {columns[target].n_missing:,} missing "
                    f"value(s). Rows without a target cannot be used for supervised "
                    f"learning; drop them before fitting.",
                    (target,),
                    {"n_missing": columns[target].n_missing},
                )
            )
        if (
            target_kind == "classification"
            and imbalance is not None
            and imbalance < thresholds.imbalance_ratio_threshold
        ):
            issues.append(
                Issue(
                    "class_imbalance",
                    Severity.WARNING,
                    f"Class imbalance: minority/majority ratio is {imbalance:.4f}. "
                    f"Consider class weights or resampling at model-fitting time. "
                    f"edaprep reports imbalance but does not resample, because "
                    f"resampling is a modelling decision and must happen after the "
                    f"train/test split.",
                    (target,),
                    {"imbalance_ratio": round(imbalance, 6)},
                )
            )
        suspicious = [
            c
            for c in column_order
            if c != target
            and columns[c].target_association is not None
            and columns[c].target_association >= thresholds.leakage_correlation_threshold
        ]
        if suspicious:
            issues.append(
                Issue(
                    "possible_target_leakage",
                    Severity.ERROR,
                    f"{len(suspicious)} column(s) are almost perfectly associated with "
                    f"the target (>= "
                    f"{thresholds.leakage_correlation_threshold:.2f}). This usually "
                    f"means the column encodes the answer, for example a value recorded "
                    f"after the outcome was known: {_names(suspicious)}.",
                    tuple(suspicious),
                    {c: round(columns[c].target_association or 0.0, 4) for c in suspicious},
                )
            )

    return issues


def _names(names: Sequence[str], limit: int = 6) -> str:
    shown = ", ".join(repr(n) for n in list(names)[:limit])
    if len(names) > limit:
        shown += f" (and {len(names) - limit} more)"
    return shown
