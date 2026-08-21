"""Configuration: global strategies, tuned thresholds, and per-column overrides.

Design notes
------------
Every number that influences a decision lives in :class:`Thresholds`, in one place, with
a comment saying where it came from.  Notebook preprocessing is full of bare literals
(``> 0.9``, ``abs(skew) > 1``, ``1.5 * IQR``, ``> 3``) whose provenance is lost; naming
them is what makes the planner's decisions auditable.

Per-column overrides are mutable and discoverable::

    config = Config()
    config.column("age").imputation = "median"
    config.column("income").outlier_strategy = "clip"
    config.column("city").encoding = "frequency"

An override recorded this way is tagged ``source="user_override"`` in the resulting
:class:`~edaprep.planning.decisions.Decision`, so ``explain()`` can distinguish a rule's
choice from the user's.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from .exceptions import ConfigurationError
from .types import ModelFamily, SemanticType

__all__ = ["Config", "ColumnConfig", "Thresholds", "AUTO"]

#: Sentinel meaning "let the planner decide".  Distinct from ``None``, which for a
#: per-column override means "no opinion, fall back to the global setting".
AUTO = "auto"

_MISSING_STRATEGIES = frozenset(
    {AUTO, "mean", "median", "mode", "constant", "ffill", "bfill", "drop_rows", "none"}
)
_OUTLIER_METHODS = frozenset({AUTO, "iqr", "zscore", "modified_zscore", "percentile", "none"})
_OUTLIER_STRATEGIES = frozenset(
    {AUTO, "report", "clip", "winsorize", "impute", "remove", "ignore"}
)
_ENCODINGS = frozenset(
    {AUTO, "onehot", "ordinal", "frequency", "count", "target", "binary", "none", "drop"}
)
_SCALINGS = frozenset({AUTO, "standard", "minmax", "robust", "maxabs", "none"})
_TRANSFORMS = frozenset(
    {AUTO, "log", "log1p", "sqrt", "boxcox", "yeojohnson", "quantile", "none"}
)
_DUPLICATE_STRATEGIES = frozenset({"report", "remove", "ignore"})
_UNKNOWN_COLUMN_POLICIES = frozenset({"error", "ignore"})
_UNKNOWN_CATEGORY_POLICIES = frozenset({"encode_as_missing", "error", "most_frequent"})

#: Values that frequently stand in for "missing" in CSV exports.  Mined from
#: ``a census-income notebook``, which scanned for ``'?'`` by hand.
DEFAULT_SENTINELS: tuple = (
    "?",
    "??",
    "-",
    "--",
    "n/a",
    "N/A",
    "na",
    "NA",
    "nan",
    "NaN",
    "null",
    "NULL",
    "none",
    "None",
    "missing",
    "MISSING",
    "unknown",
    "UNKNOWN",
    "",
    " ",
)

#: Numeric sentinels are checked separately: replacing them is far more dangerous
#: (-999 may be a legitimate value), so they are reported, never auto-replaced.
DEFAULT_NUMERIC_SENTINELS: tuple = (-999.0, -9999.0, -99999.0, 999999.0)


def _validate(name: str, value: Any, allowed: frozenset) -> Any:
    if value not in allowed:
        raise ConfigurationError.unknown_option(name, value, sorted(allowed))
    return value


def _validate_range(name: str, value: float, low: float, high: float) -> float:
    if not (low <= value <= high):
        raise ConfigurationError.out_of_range(name, value, low, high)
    return value


@dataclass
class Thresholds:
    """Every decision threshold in the library, named and sourced.

    Thresholds marked *(conventional)* are the values practice has settled on.  Keeping
    the same numbers means the planner reproduces familiar decisions; naming them means
    they can be argued with.
    """

    # --- semantic type inference -------------------------------------------------
    #: A column whose unique-value ratio exceeds this is a candidate identifier.
    id_unique_ratio: float = 0.95
    #: ...but only if the frame has at least this many rows.  In a 10-row frame every
    #: column looks unique.
    id_min_rows: int = 50
    #: An integral numeric column with at most this many distinct values is treated as
    #: categorical.  Scaled by ``n_rows`` in practice (see ``column_types``).
    numeric_as_categorical_max: int = 20
    #: ...and only when distinct values are at most this fraction of the rows.
    numeric_as_categorical_max_ratio: float = 0.05
    #: Mean character length above which an object column is TEXT rather than CATEGORICAL.
    text_min_mean_length: float = 50.0
    #: Mean whitespace-token count above which an object column is TEXT.
    text_min_mean_tokens: float = 4.0
    #: Fraction of the modal value above which a column is "near-constant".
    near_constant_ratio: float = 0.99
    #: Fraction of parseable values required to call an object column DATETIME.
    datetime_parse_ratio: float = 0.90

    # --- missing values -----------------------------------------------------------
    #: Above this missing fraction, imputing is misleading; the planner drops the
    #: column (and says so).
    missing_drop_threshold: float = 0.60
    #: Above this missing fraction, a missing-indicator column is added.
    missing_indicator_threshold: float = 0.05
    #: Below this missing fraction, imputation is applied without further comment.
    missing_impute_threshold: float = 0.60

    # --- outliers -----------------------------------------------------------------
    #: IQR fence multiplier for symmetric columns.  1.5 is Tukey's original.
    iqr_k: float = 1.5
    #: IQR fence multiplier for skewed columns; 3.0 is the usual widening.
    iqr_k_skewed: float = 3.0
    #: |z| threshold; 3.0 is conventional.
    zscore_threshold: float = 3.0
    #: Modified (MAD-based) z threshold.  Iglewicz & Hoaglin's recommendation.
    modified_zscore_threshold: float = 3.5
    #: Percentile fence for the "percentile"/"winsorize" methods; 1/99 is conventional.
    percentile_bounds: tuple = (0.01, 0.99)
    #: Only act automatically on outliers below this contamination fraction; above it,
    #: the values are probably the distribution, not errors.
    outlier_max_action_fraction: float = 0.10

    # --- skewness / distribution --------------------------------------------------
    #: |skew| below this is "symmetric": standard scaling, no transform.
    skew_moderate: float = 1.0
    #: |skew| at or above this is "heavy": power transform.
    skew_heavy: float = 5.0

    # --- categorical --------------------------------------------------------------
    #: Categories with frequency below this are grouped into a rare bucket.
    rare_category_threshold: float = 0.01
    #: Above this cardinality, one-hot encoding is refused: unbounded expansion is
    #: the usual mistake.
    high_cardinality_threshold: int = 50
    #: Hard ceiling: above this, only frequency/target encoding or dropping is offered.
    extreme_cardinality_threshold: int = 1000
    #: Maximum total columns one-hot encoding may add before the planner refuses.
    max_onehot_columns: int = 200

    # --- feature selection --------------------------------------------------------
    #: Absolute correlation above which two features are considered redundant.
    correlation_threshold: float = 0.95
    #: Variance below which a numeric feature is dropped.
    variance_threshold: float = 0.0
    #: Modal-value fraction above which a feature is dropped as near-constant.
    near_constant_drop_ratio: float = 0.995

    # --- target -------------------------------------------------------------------
    #: Distinct target values at or below this count means classification.
    classification_max_classes: int = 20
    #: Minority/majority ratio below this is reported as class imbalance.
    imbalance_ratio_threshold: float = 0.20
    #: |corr(feature, target)| above this is flagged as possible leakage.
    leakage_correlation_threshold: float = 0.98

    # --- performance --------------------------------------------------------------
    #: Rows above which expensive statistics switch to a sample.
    sampling_row_threshold: int = 200_000
    #: Sample size used once the threshold is crossed.
    sample_size: int = 50_000
    #: Columns above which the full correlation matrix is skipped in "standard" EDA.
    correlation_max_columns: int = 200

    def validate(self) -> "Thresholds":
        _validate_range("id_unique_ratio", self.id_unique_ratio, 0.0, 1.0)
        _validate_range("missing_drop_threshold", self.missing_drop_threshold, 0.0, 1.0)
        _validate_range("rare_category_threshold", self.rare_category_threshold, 0.0, 1.0)
        _validate_range("correlation_threshold", self.correlation_threshold, 0.0, 1.0)
        _validate_range("near_constant_ratio", self.near_constant_ratio, 0.0, 1.0)
        if self.skew_heavy <= self.skew_moderate:
            raise ConfigurationError(
                f"skew_heavy ({self.skew_heavy}) must be greater than skew_moderate "
                f"({self.skew_moderate}); they define adjacent tiers of a single scale."
            )
        if self.high_cardinality_threshold >= self.extreme_cardinality_threshold:
            raise ConfigurationError(
                f"high_cardinality_threshold ({self.high_cardinality_threshold}) must be "
                f"below extreme_cardinality_threshold "
                f"({self.extreme_cardinality_threshold})."
            )
        if self.iqr_k <= 0 or self.zscore_threshold <= 0:
            raise ConfigurationError(
                "iqr_k and zscore_threshold must be positive; a non-positive fence "
                "would classify every value as an outlier."
            )
        lo, hi = self.percentile_bounds
        if not 0.0 <= lo < hi <= 1.0:
            raise ConfigurationError(
                f"percentile_bounds={self.percentile_bounds} must satisfy "
                f"0 <= lower < upper <= 1."
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ColumnConfig:
    """Per-column overrides.

    ``None`` means "no opinion": the global setting, and then the planner's rules,
    decide.  Anything else overrides the planner and is reported as a user override.
    """

    name: str
    semantic_type: Optional[Union[SemanticType, str]] = None
    role: Optional[str] = None
    imputation: Optional[str] = None
    imputation_fill_value: Any = None
    outlier_method: Optional[str] = None
    outlier_strategy: Optional[str] = None
    transform: Optional[str] = None
    encoding: Optional[str] = None
    scaling: Optional[str] = None
    rare_category_threshold: Optional[float] = None
    datetime_features: Optional[Sequence[str]] = None
    drop: Optional[bool] = None

    def validate(self) -> "ColumnConfig":
        if self.imputation is not None:
            _validate(f"column({self.name!r}).imputation", self.imputation, _MISSING_STRATEGIES)
        if self.outlier_method is not None:
            _validate(
                f"column({self.name!r}).outlier_method", self.outlier_method, _OUTLIER_METHODS
            )
        if self.outlier_strategy is not None:
            _validate(
                f"column({self.name!r}).outlier_strategy",
                self.outlier_strategy,
                _OUTLIER_STRATEGIES,
            )
        if self.transform is not None:
            _validate(f"column({self.name!r}).transform", self.transform, _TRANSFORMS)
        if self.encoding is not None:
            _validate(f"column({self.name!r}).encoding", self.encoding, _ENCODINGS)
        if self.scaling is not None:
            _validate(f"column({self.name!r}).scaling", self.scaling, _SCALINGS)
        if self.semantic_type is not None:
            self.semantic_type = SemanticType.coerce(self.semantic_type)
        if self.imputation == "constant" and self.imputation_fill_value is None:
            raise ConfigurationError(
                f"column({self.name!r}).imputation='constant' also requires "
                f"imputation_fill_value to be set, otherwise there is nothing to fill with."
            )
        return self

    def has_overrides(self) -> bool:
        return any(
            getattr(self, f.name) is not None
            for f in dataclasses.fields(self)
            if f.name != "name"
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }
        if isinstance(out.get("semantic_type"), SemanticType):
            out["semantic_type"] = str(out["semantic_type"])
        return out


@dataclass
class Config:
    """Global configuration.

    Parameters
    ----------
    missing_strategy, outlier_method, outlier_strategy, transform_strategy,
    categorical_encoding, scaling :
        Global defaults.  ``"auto"`` (the default) hands the decision to the planner.
        Any other value pins every applicable column to that choice.
    model_family :
        What the output is destined for.  Drives scaling and encoding choices; see
        ``docs/architecture.md`` section 5.3.  ``None`` selects a conservative branch
        that makes no modelling assumptions.
    thresholds :
        See :class:`Thresholds`.
    random_state :
        Seed for every stochastic operation (sampling, cross-fitting folds, optional
        IsolationForest).  Recorded in the report.
    """

    # --- strategy selection -----------------------------------------------------
    missing_strategy: str = AUTO
    outlier_method: str = AUTO
    outlier_strategy: str = "report"  # conservative: detect, do not delete
    transform_strategy: str = AUTO
    categorical_encoding: str = AUTO
    scaling: str = AUTO
    duplicate_strategy: str = "report"

    # --- behaviour switches -----------------------------------------------------
    model_family: Optional[Union[ModelFamily, str]] = None
    add_missing_indicators: bool = True
    detect_sentinels: bool = True
    replace_sentinels: bool = True
    downcast_numeric: bool = False
    expand_datetime: bool = True
    drop_identifiers: bool = True
    drop_constants: bool = True
    drop_duplicate_columns: bool = True
    drop_high_missing: bool = True
    correlation_filter: bool = False  # off by default: it removes information
    handle_unknown_categories: str = "encode_as_missing"
    on_unknown_columns: str = "error"

    # --- knobs ------------------------------------------------------------------
    rare_category_threshold: Optional[float] = None
    high_cardinality_threshold: Optional[int] = None
    target_encoding_folds: int = 5
    target_encoding_smoothing: float = 20.0
    sentinels: Sequence[str] = DEFAULT_SENTINELS
    numeric_sentinels: Sequence[float] = DEFAULT_NUMERIC_SENTINELS

    # --- reproducibility / performance ------------------------------------------
    random_state: Optional[int] = None
    sample_size: Optional[int] = None
    n_jobs: int = 1
    verbose: bool = False

    thresholds: Thresholds = field(default_factory=Thresholds)
    columns: Dict[str, ColumnConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def __setattr__(self, name: str, value: Any) -> None:
        """Coerce enum-valued settings on every assignment, not only at construction.

        ``model_family`` is compared with ``is`` throughout the planner and the
        transformers, because an enum identity check is the clearest way to say "this
        exact family".  A plain string assigned after construction --
        ``config.model_family = "tree"``, the obvious thing to write -- compares equal
        but is not identical, which silently disabled every one of those checks and
        made a tree pipeline scale its features.  Coercing here makes the obvious
        thing correct.  (A property cannot be used: it would shadow the dataclass
        field and become its own default value.)
        """
        if name == "model_family" and value is not None and not isinstance(value, ModelFamily):
            value = ModelFamily.coerce(value)
        object.__setattr__(self, name, value)

    # -- overrides ---------------------------------------------------------------

    def column(self, name: str) -> ColumnConfig:
        """Return the (mutable) override record for ``name``, creating it on demand."""
        if name not in self.columns:
            self.columns[name] = ColumnConfig(name=name)
        return self.columns[name]

    def get_column(self, name: str) -> Optional[ColumnConfig]:
        """Return the override record for ``name`` without creating one."""
        return self.columns.get(name)

    def set_columns(self, overrides: Mapping[str, Mapping[str, Any]]) -> "Config":
        """Bulk-apply overrides: ``{"age": {"imputation": "median"}, ...}``."""
        for name, kwargs in overrides.items():
            col = self.column(name)
            for key, value in kwargs.items():
                if not hasattr(col, key):
                    valid = [
                        f.name for f in dataclasses.fields(ColumnConfig) if f.name != "name"
                    ]
                    raise ConfigurationError.unknown_option(
                        f"column({name!r}) setting", key, valid
                    )
                setattr(col, key, value)
            col.validate()
        return self

    # -- validation / serialisation ----------------------------------------------

    def validate(self) -> "Config":
        _validate("missing_strategy", self.missing_strategy, _MISSING_STRATEGIES)
        _validate("outlier_method", self.outlier_method, _OUTLIER_METHODS)
        _validate("outlier_strategy", self.outlier_strategy, _OUTLIER_STRATEGIES)
        _validate("transform_strategy", self.transform_strategy, _TRANSFORMS)
        _validate("categorical_encoding", self.categorical_encoding, _ENCODINGS)
        _validate("scaling", self.scaling, _SCALINGS)
        _validate("duplicate_strategy", self.duplicate_strategy, _DUPLICATE_STRATEGIES)
        _validate("on_unknown_columns", self.on_unknown_columns, _UNKNOWN_COLUMN_POLICIES)
        _validate(
            "handle_unknown_categories",
            self.handle_unknown_categories,
            _UNKNOWN_CATEGORY_POLICIES,
        )
        if self.target_encoding_folds < 2:
            raise ConfigurationError(
                f"target_encoding_folds={self.target_encoding_folds} must be at least 2; "
                f"cross-fitting with fewer folds cannot hold out any rows, which is the "
                f"whole point of the technique."
            )
        if self.rare_category_threshold is not None:
            _validate_range("rare_category_threshold", self.rare_category_threshold, 0.0, 1.0)
        if self.sample_size is not None and self.sample_size < 1:
            raise ConfigurationError.out_of_range("sample_size", self.sample_size, 1, 1 << 62)
        self.thresholds.validate()
        for col in self.columns.values():
            col.validate()
        return self

    # -- resolved accessors -------------------------------------------------------

    @property
    def effective_rare_threshold(self) -> float:
        if self.rare_category_threshold is not None:
            return self.rare_category_threshold
        return self.thresholds.rare_category_threshold

    @property
    def effective_high_cardinality(self) -> int:
        if self.high_cardinality_threshold is not None:
            return self.high_cardinality_threshold
        return self.thresholds.high_cardinality_threshold

    @property
    def effective_sample_size(self) -> int:
        if self.sample_size is not None:
            return self.sample_size
        return self.thresholds.sample_size

    def copy(self) -> "Config":
        """Deep copy, so that mutating overrides on the copy is safe."""
        import copy as _copy

        return _copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "thresholds":
                out[f.name] = value.to_dict()
            elif f.name == "columns":
                out[f.name] = {k: v.to_dict() for k, v in value.items() if v.has_overrides()}
            elif isinstance(value, ModelFamily):
                out[f.name] = str(value)
            elif isinstance(value, tuple):
                out[f.name] = list(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        data = dict(data)
        thresholds = Thresholds(**data.pop("thresholds", {}))
        columns_raw = data.pop("columns", {})
        cfg = cls(thresholds=thresholds, **data)
        for name, kwargs in columns_raw.items():
            kwargs = dict(kwargs)
            kwargs.pop("name", None)
            cfg.set_columns({name: kwargs})
        return cfg

    def __repr__(self) -> str:
        non_default = []
        blank = Config.__new__(Config)
        for f in dataclasses.fields(self):
            if f.name in ("thresholds", "columns"):
                continue
            default = f.default if f.default is not dataclasses.MISSING else None
            value = getattr(self, f.name)
            if value != default:
                non_default.append(f"{f.name}={value!r}")
        del blank
        overrides = (
            f", {len(self.columns)} column override(s)"
            if any(c.has_overrides() for c in self.columns.values())
            else ""
        )
        inner = ", ".join(non_default)
        return f"Config({inner}{overrides})"
