"""Semantic column-type inference.

The single most consequential correction this library makes to the reference workflow.
The ubiquitous ``select_dtypes(include=['int64','float64'])`` routes a zip code and a
temperature into the same preprocessing branch.

The inference is a cascade of falsifiable checks that returns a *confidence* and a list
of runner-up types rather than a bare label, so that "when uncertain, expose the
uncertainty" is satisfied structurally: low confidence becomes a reported quality issue.

Name heuristics are advisory only.  A column called ``user_id`` with 3 distinct values
in 100k rows is not an identifier; the name moves confidence, cardinality decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Thresholds
from ..types import SemanticType

__all__ = ["TypeInference", "infer_semantic_type", "is_integral", "name_suggests"]

# --- name heuristics -------------------------------------------------------------
# Anchored on word boundaries so that "residual" does not match "id".
_ID_NAME = re.compile(
    r"(^|[_\W])(id|ids|uuid|guid|key|pk|index|idx|code|no|num|number|"
    r"identifier|ref|reference|sku|isbn|ssn|account|serial)($|[_\W])",
    re.IGNORECASE,
)
_DATETIME_NAME = re.compile(
    r"(date|time|timestamp|datetime|_at$|_on$|created|updated|modified|"
    r"birth|dob|year|month|day|week|quarter|hour|minute|epoch)",
    re.IGNORECASE,
)
_ORDINAL_NAME = re.compile(
    r"(level|grade|rank|rating|score|scale|tier|severity|stage|priority|"
    r"satisfaction|quality|class|size|degree|band|bucket|star)",
    re.IGNORECASE,
)
_TEXT_NAME = re.compile(
    r"(text|comment|description|desc|review|message|body|content|note|"
    r"summary|title|address|remark|feedback)",
    re.IGNORECASE,
)
_CATEGORY_NAME = re.compile(
    r"(type|category|cat|status|state|group|kind|gender|sex|country|city|"
    r"region|segment|channel|brand|colour|color|label|flag|is_|has_)",
    re.IGNORECASE,
)

#: Values a boolean-ish column may take, lower-cased.
_BOOLEAN_VOCAB = frozenset(
    {"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0", "1.0", "0.0"}
)


def name_suggests(name: str) -> Dict[str, bool]:
    """Which name heuristics fire for ``name``.  Exposed for testing and explainability."""
    text = str(name)
    return {
        "identifier": bool(_ID_NAME.search(text)),
        "datetime": bool(_DATETIME_NAME.search(text)),
        "ordinal": bool(_ORDINAL_NAME.search(text)),
        "text": bool(_TEXT_NAME.search(text)),
        "categorical": bool(_CATEGORY_NAME.search(text)),
    }


@dataclass(frozen=True)
class TypeInference:
    """The outcome of inferring one column's semantic type."""

    semantic: SemanticType
    confidence: float
    alternatives: Tuple[SemanticType, ...] = ()
    reasons: Tuple[str, ...] = ()
    #: Populated when the column is stored as text but holds parseable dates/numbers.
    suggested_dtype: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_uncertain(self) -> bool:
        return self.confidence < 0.70

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic": str(self.semantic),
            "confidence": round(float(self.confidence), 3),
            "alternatives": [str(a) for a in self.alternatives],
            "reasons": list(self.reasons),
            "suggested_dtype": self.suggested_dtype,
        }


def is_integral(series: pd.Series, sample: Optional[np.ndarray] = None) -> bool:
    """True when every finite value is a whole number.

    ``float`` columns holding ``1.0, 2.0, 3.0`` are integral in this sense; that is
    exactly the encoded-category case that gets mishandled.
    """
    if pd.api.types.is_integer_dtype(series.dtype):
        return True
    if not pd.api.types.is_float_dtype(series.dtype):
        return False
    arr = sample if sample is not None else series.to_numpy(dtype="float64", na_value=np.nan)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(np.all(np.equal(np.mod(finite, 1.0), 0.0)))


def _sample_values(series: pd.Series, n: int, random_state: Optional[int]) -> pd.Series:
    """Deterministic sample of non-null values, for the expensive string probes."""
    non_null = series.dropna()
    if len(non_null) <= n:
        return non_null
    return non_null.sample(n=n, random_state=random_state if random_state is not None else 0)


def _parse_ratio_datetime(values: pd.Series) -> float:
    """Fraction of ``values`` parseable as datetimes.

    Pure-digit strings are excluded first: ``pd.to_datetime`` happily reads ``"2020"``
    or ``"1234"`` as a year, which would misclassify most integer-coded ID columns.
    """
    if values.empty:
        return 0.0
    as_str = values.astype(str).str.strip()
    plausible = ~as_str.str.fullmatch(r"[-+]?\d*\.?\d+")
    candidates = as_str[plausible]
    if candidates.empty:
        return 0.0
    parsed = pd.to_datetime(candidates, errors="coerce", format="mixed")
    # Scale by the full sample, not just the plausible subset: a column that is 90%
    # bare integers and 10% dates is not a datetime column.
    return float(parsed.notna().sum()) / float(len(values))


def _parse_ratio_numeric(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    parsed = pd.to_numeric(values, errors="coerce")
    return float(parsed.notna().sum()) / float(len(values))


def _string_shape(values: pd.Series) -> Tuple[float, float]:
    """(mean character length, mean whitespace-token count) over a sample."""
    if values.empty:
        return 0.0, 0.0
    as_str = values.astype(str)
    lengths = as_str.str.len()
    tokens = as_str.str.count(r"\s+") + 1
    return float(lengths.mean()), float(tokens.mean())


def _effective_categorical_max(n_rows: int, thresholds: Thresholds) -> int:
    """Cardinality ceiling for treating an integral numeric column as categorical.

    Scales with ``n_rows``: 12 distinct values is categorical in a 100k-row frame and
    is probably continuous in a 15-row frame.  Without this, tiny test fixtures and
    real datasets need different thresholds, which is how magic numbers proliferate.
    """
    ratio_cap = int(max(2, n_rows * thresholds.numeric_as_categorical_max_ratio))
    return int(min(thresholds.numeric_as_categorical_max, max(2, ratio_cap)))


def infer_semantic_type(
    series: pd.Series,
    *,
    n_unique: Optional[int] = None,
    thresholds: Optional[Thresholds] = None,
    is_target: bool = False,
    hint: Optional[SemanticType] = None,
    random_state: Optional[int] = None,
    probe_size: int = 1000,
) -> TypeInference:
    """Infer what ``series`` *means*.

    Parameters
    ----------
    series :
        The column.  Only a bounded sample is used for the expensive string probes.
    n_unique :
        Pre-computed distinct-value count, to avoid a second hashing pass when the
        profiler has already done it.
    thresholds :
        Tunable cut-points; see :class:`~edaprep.config.Thresholds`.
    is_target :
        Targets are never classified as identifiers, and low-cardinality numeric
        targets keep their numeric identity unless clearly categorical.
    hint :
        A user-supplied type.  Returned directly with confidence 1.0; the user is
        allowed to be right.
    """
    thresholds = thresholds or Thresholds()

    if hint is not None:
        return TypeInference(
            semantic=SemanticType.coerce(hint),
            confidence=1.0,
            reasons=("explicitly configured by the user",),
        )

    name = str(series.name) if series.name is not None else ""
    hints = name_suggests(name)
    dtype = series.dtype
    n_rows = len(series)
    n_missing = int(series.isna().sum())
    n_present = n_rows - n_missing

    if n_unique is None:
        n_unique = int(series.nunique(dropna=True))
    unique_ratio = (n_unique / n_present) if n_present else 0.0

    reasons: List[str] = []

    # --- 1. constants -----------------------------------------------------------
    if n_present == 0:
        return TypeInference(
            SemanticType.CONSTANT,
            confidence=1.0,
            reasons=("column is entirely missing",),
        )
    if n_unique <= 1:
        return TypeInference(
            SemanticType.CONSTANT,
            confidence=1.0,
            reasons=(f"only {n_unique} distinct non-null value",),
        )

    # --- 2. dtypes that settle the question -------------------------------------
    if pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(
        dtype, pd.DatetimeTZDtype
    ):
        return TypeInference(
            SemanticType.DATETIME, confidence=1.0, reasons=("stored as a datetime dtype",)
        )
    if pd.api.types.is_timedelta64_dtype(dtype):
        return TypeInference(
            SemanticType.NUMERIC,
            confidence=0.95,
            reasons=("stored as a timedelta; treated as a numeric duration",),
        )
    if pd.api.types.is_bool_dtype(dtype):
        return TypeInference(
            SemanticType.BINARY, confidence=1.0, reasons=("stored as a boolean dtype",)
        )

    is_categorical_dtype = isinstance(dtype, pd.CategoricalDtype)
    is_numeric = pd.api.types.is_numeric_dtype(dtype) and not is_categorical_dtype

    # --- 3. object / string / category ------------------------------------------
    if not is_numeric:
        probe = _sample_values(series, probe_size, random_state)

        if n_unique <= 4 and not is_categorical_dtype:
            try:
                lowered = {str(v).strip().lower() for v in series.dropna().unique()}
            except TypeError:
                # Unhashable cell values (lists, dicts, sets).  Rare, but a real
                # export artefact; the column cannot be a boolean, so move on.
                lowered = set()
            if lowered and lowered <= _BOOLEAN_VOCAB:
                return TypeInference(
                    SemanticType.BINARY,
                    confidence=0.95,
                    reasons=(f"values are boolean-like: {sorted(lowered)}",),
                    suggested_dtype="boolean",
                )

        if not is_categorical_dtype:
            dt_ratio = _parse_ratio_datetime(probe)
            if dt_ratio >= thresholds.datetime_parse_ratio:
                conf = 0.75 + 0.2 * dt_ratio + (0.05 if hints["datetime"] else 0.0)
                return TypeInference(
                    SemanticType.DATETIME,
                    confidence=min(conf, 0.99),
                    alternatives=(SemanticType.TEXT,),
                    reasons=(
                        f"{dt_ratio:.0%} of sampled values parse as datetimes",
                        *(("name suggests a date/time",) if hints["datetime"] else ()),
                    ),
                    suggested_dtype="datetime64[ns]",
                )

            num_ratio = _parse_ratio_numeric(probe)
            if num_ratio >= 0.95:
                reasons.append(
                    f"stored as text but {num_ratio:.0%} of sampled values parse as numbers"
                )
                coerced = pd.to_numeric(series, errors="coerce")
                inner = infer_semantic_type(
                    coerced.rename(series.name),
                    n_unique=n_unique,
                    thresholds=thresholds,
                    is_target=is_target,
                    random_state=random_state,
                    probe_size=probe_size,
                )
                return TypeInference(
                    inner.semantic,
                    confidence=inner.confidence * 0.9,
                    alternatives=tuple(
                        dict.fromkeys((*inner.alternatives, SemanticType.CATEGORICAL))
                    ),
                    reasons=tuple(reasons) + inner.reasons,
                    suggested_dtype="float64",
                )

            mean_len, mean_tokens = _string_shape(probe)
            if (
                mean_len >= thresholds.text_min_mean_length
                or mean_tokens >= thresholds.text_min_mean_tokens
            ):
                conf = 0.80 + (0.1 if hints["text"] else 0.0)
                return TypeInference(
                    SemanticType.TEXT,
                    confidence=min(conf, 0.95),
                    alternatives=(SemanticType.CATEGORICAL,),
                    reasons=(
                        f"mean length {mean_len:.0f} chars, {mean_tokens:.1f} tokens "
                        f"per value",
                    ),
                    extra={"mean_length": mean_len, "mean_tokens": mean_tokens},
                )

        if (
            not is_target
            and n_rows >= thresholds.id_min_rows
            and unique_ratio >= thresholds.id_unique_ratio
        ):
            conf = 0.70 + 0.2 * min(unique_ratio, 1.0) + (0.09 if hints["identifier"] else 0.0)
            return TypeInference(
                SemanticType.IDENTIFIER,
                confidence=min(conf, 0.99),
                alternatives=(SemanticType.CATEGORICAL, SemanticType.TEXT),
                reasons=(
                    f"{unique_ratio:.1%} of non-null values are distinct",
                    *(("name matches an identifier pattern",) if hints["identifier"] else ()),
                ),
            )

        if n_unique == 2:
            return TypeInference(
                SemanticType.BINARY,
                confidence=0.90,
                alternatives=(SemanticType.CATEGORICAL,),
                reasons=("exactly 2 distinct non-null values",),
            )

        conf = 0.85 + (0.1 if (is_categorical_dtype or hints["categorical"]) else 0.0)
        if unique_ratio > 0.5 and n_rows >= thresholds.id_min_rows:
            conf -= 0.20
            reasons.append(f"high cardinality for a category ({unique_ratio:.0%} distinct)")
        return TypeInference(
            SemanticType.CATEGORICAL,
            confidence=min(max(conf, 0.3), 0.99),
            alternatives=(SemanticType.TEXT, SemanticType.IDENTIFIER),
            reasons=tuple(reasons) or ("non-numeric values with limited cardinality",),
        )

    # --- 4. numeric --------------------------------------------------------------
    values = series.to_numpy(dtype="float64", na_value=np.nan, copy=False)
    integral = is_integral(series, values)

    if n_unique == 2:
        distinct = np.unique(values[np.isfinite(values)])
        is_zero_one = distinct.size == 2 and set(distinct.tolist()) == {0.0, 1.0}
        return TypeInference(
            SemanticType.BINARY,
            confidence=0.95 if is_zero_one else 0.85,
            alternatives=(SemanticType.NUMERIC,),
            reasons=(
                "exactly 2 distinct values"
                + (" (0/1)" if is_zero_one else f" ({distinct.tolist()})"),
            ),
        )

    if (
        not is_target
        and integral
        and n_rows >= thresholds.id_min_rows
        and unique_ratio >= thresholds.id_unique_ratio
    ):
        finite = values[np.isfinite(values)]
        strictly_increasing = bool(finite.size > 1 and np.all(np.diff(finite) > 0))
        conf = 0.55 + 0.2 * min(unique_ratio, 1.0)
        if hints["identifier"]:
            conf += 0.15
            reasons.append("name matches an identifier pattern")
        if strictly_increasing:
            conf += 0.10
            reasons.append("values are strictly increasing, consistent with a row key")
        reasons.insert(0, f"integral with {unique_ratio:.1%} distinct values")
        return TypeInference(
            SemanticType.IDENTIFIER,
            confidence=min(conf, 0.98),
            alternatives=(SemanticType.NUMERIC,),
            reasons=tuple(reasons),
        )

    cat_max = _effective_categorical_max(n_rows, thresholds)
    if integral and n_unique <= cat_max:
        finite = values[np.isfinite(values)]
        distinct = np.unique(finite)
        small_dense = (
            distinct.size > 2
            and distinct.min() >= 0
            and distinct.max() < 100
            and np.all(np.diff(distinct) == 1)
        )
        # Dense consecutive integers mean the column is *coded*, not that the codes are
        # ordered: `payment_type` in 0..4 is as dense as `satisfaction` in 1..5.  Order
        # is only claimed when the name says so, and density then raises confidence.
        # Getting this wrong is expensive in both directions -- ordinal-encoding a
        # nominal column invents a false ranking, one-hot-encoding an ordinal one
        # discards a real one -- so the conservative default is CATEGORICAL.
        if hints["ordinal"]:
            conf = 0.75 + (0.10 if small_dense else 0.0)
            return TypeInference(
                SemanticType.ORDINAL,
                confidence=min(conf, 0.90),
                alternatives=(SemanticType.CATEGORICAL, SemanticType.NUMERIC),
                reasons=(
                    "name suggests an ordered scale",
                    *(
                        (f"{n_unique} distinct consecutive integer levels",)
                        if small_dense
                        else ()
                    ),
                ),
            )
        conf = 0.55 + (0.15 if hints["categorical"] else 0.0)
        if hints["identifier"]:
            conf -= 0.05
        return TypeInference(
            SemanticType.CATEGORICAL,
            confidence=min(max(conf, 0.35), 0.90),
            alternatives=(SemanticType.NUMERIC, SemanticType.ORDINAL),
            reasons=(
                f"integer-coded with only {n_unique} distinct values "
                f"(threshold {cat_max} for {n_rows} rows)",
            ),
        )

    conf = 0.90
    if hints["identifier"] and integral:
        conf -= 0.15
        reasons.append("name matches an identifier pattern, but cardinality is too low")
    if hints["datetime"] and integral and n_unique < 200:
        conf -= 0.10
        reasons.append("name suggests a date part (year/month); may be better as datetime")
    return TypeInference(
        SemanticType.NUMERIC,
        confidence=min(max(conf, 0.4), 0.99),
        alternatives=(SemanticType.CATEGORICAL,) if integral else (),
        reasons=tuple(reasons) or ("continuous numeric values",),
    )


def infer_frame_types(
    frame: pd.DataFrame,
    *,
    thresholds: Optional[Thresholds] = None,
    target: Optional[str] = None,
    hints: Optional[Dict[str, SemanticType]] = None,
    n_unique: Optional[Dict[str, int]] = None,
    random_state: Optional[int] = None,
) -> Dict[str, TypeInference]:
    """Infer semantic types for every column of ``frame``."""
    hints = hints or {}
    n_unique = n_unique or {}
    out: Dict[str, TypeInference] = {}
    for column in frame.columns:
        key = str(column)
        out[key] = infer_semantic_type(
            frame[column],
            n_unique=n_unique.get(key),
            thresholds=thresholds,
            is_target=(key == target),
            hint=hints.get(key),
            random_state=random_state,
        )
    return out


def semantic_groups(
    inferences: Dict[str, TypeInference], types: Sequence[SemanticType]
) -> List[str]:
    """Column names whose inferred type is in ``types``, preserving input order."""
    wanted = {SemanticType.coerce(t) for t in types}
    return [name for name, inf in inferences.items() if inf.semantic in wanted]
