"""Core enumerations and light-weight value types.

This module deliberately imports nothing beyond the standard library.  Almost every
other module imports it, so keeping it dependency-free is what prevents import cycles
without scattering ``TYPE_CHECKING`` guards through the package.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

__all__ = [
    "SemanticType",
    "ColumnRole",
    "Stage",
    "ModelFamily",
    "AnalysisLevel",
    "Severity",
    "NUMERIC_LIKE",
    "CATEGORICAL_LIKE",
]


class _StrEnum(str, Enum):
    """A ``str`` subclass enum.

    ``enum.StrEnum`` only exists on Python 3.11+, and the mixin form behaves the same
    for our purposes: members compare equal to their value, so a user may write
    ``"numeric"`` wherever :class:`SemanticType` is accepted.
    """

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}.{self.name}"

    @classmethod
    def coerce(cls, value):
        """Accept a member, its value, or its (case-insensitive) name."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for member in cls:
                if member.value == lowered or member.name.lower() == lowered:
                    return member
        raise ValueError(
            f"{value!r} is not a valid {cls.__name__}. "
            f"Valid options are: {', '.join(m.value for m in cls)}."
        )


class SemanticType(_StrEnum):
    """What a column *means*, as opposed to how it is stored.

    docs/design-rationale.md identifies conflating these two as the single most common defect
    in notebook preprocessing: ``select_dtypes(include=['int64'])`` puts a zip code
    and a temperature in the same bucket.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


#: Semantic types that a numeric transformer may operate on.
NUMERIC_LIKE = frozenset({SemanticType.NUMERIC, SemanticType.ORDINAL})

#: Semantic types that a categorical encoder may operate on.
CATEGORICAL_LIKE = frozenset(
    {SemanticType.CATEGORICAL, SemanticType.BINARY, SemanticType.ORDINAL}
)


class ColumnRole(_StrEnum):
    """The part a column plays in the modelling problem."""

    FEATURE = "feature"
    TARGET = "target"
    IDENTIFIER = "identifier"
    METADATA = "metadata"
    IGNORED = "ignored"


class Stage(_StrEnum):
    """Preprocessing stages, in canonical execution order.

    The ordering is a design claim; see ``docs/architecture.md`` section 5.2.  Notably
    ``MISSING_FLAG`` precedes imputation (so that informative missingness survives) and
    ``OUTLIERS`` precedes ``MISSING`` (so that the ``"impute"`` outlier strategy fills
    from statistics computed without the outliers).
    """

    CAST = "cast"
    DROP_COLUMNS = "drop_columns"
    DEDUPLICATE = "deduplicate"
    MISSING_FLAG = "missing_flag"
    DATETIME = "datetime"
    OUTLIERS = "outliers"
    MISSING = "missing"
    TRANSFORM = "transform"
    RARE_CATEGORY = "rare_category"
    ENCODE = "encode"
    SCALE = "scale"
    SELECT = "select"
    FEATURE_ENGINEERING = "feature_engineering"  # reserved extension point, unused in v1

    @property
    def order(self) -> int:
        return _STAGE_ORDER[self]


_STAGE_ORDER = {
    Stage.CAST: 0,
    Stage.DROP_COLUMNS: 1,
    Stage.DEDUPLICATE: 2,
    Stage.MISSING_FLAG: 3,
    Stage.DATETIME: 4,
    Stage.OUTLIERS: 5,
    Stage.MISSING: 6,
    Stage.TRANSFORM: 7,
    Stage.RARE_CATEGORY: 8,
    Stage.ENCODE: 9,
    Stage.SCALE: 10,
    Stage.SELECT: 11,
    Stage.FEATURE_ENGINEERING: 12,
}


class ModelFamily(_StrEnum):
    """The family of model the output is destined for.

    Reflects the common practice of maintaining parallel preprocessing branches for
    tree-based and linear models.  It is an *input to
    planning* only; ``edaprep`` never trains anything.
    """

    LINEAR = "linear"
    TREE = "tree"
    DISTANCE = "distance"
    NEURAL = "neural"

    @property
    def needs_scaling(self) -> bool:
        return self is not ModelFamily.TREE

    @property
    def needs_dense_numeric(self) -> bool:
        """True when categorical levels must not be given a false ordering."""
        return self is not ModelFamily.TREE


class AnalysisLevel(_StrEnum):
    """How much work :class:`edaprep.EDA` is permitted to do."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

    @property
    def rank(self) -> int:
        return {"quick": 0, "standard": 1, "deep": 2}[self.value]


class Severity(_StrEnum):
    """Severity of a reported data-quality issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2}[self.value]


def describe_semantic(semantic: SemanticType, confidence: Optional[float] = None) -> str:
    """Render a semantic type for human-facing output."""
    if confidence is None:
        return str(semantic)
    return f"{semantic} (confidence {confidence:.2f})"
