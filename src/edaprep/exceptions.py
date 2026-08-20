"""Typed exceptions with actionable messages.

Notebook practice suppressed warnings globally (``warnings.filterwarnings("ignore")``
in 10 notebooks), which hid a real ``SettingWithCopyWarning``.  ``edaprep`` never
suppresses warnings and never raises a bare ``ValueError``: every raise site states what
was attempted, on which column, why it failed, and what to do instead.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = [
    "EdaPrepError",
    "ConfigurationError",
    "NotFittedError",
    "SchemaError",
    "DataError",
    "EmptyDataError",
    "TransformationError",
    "LeakageError",
]


class EdaPrepError(Exception):
    """Base class for every error raised by edaprep."""


class ConfigurationError(EdaPrepError):
    """The requested configuration is unknown, contradictory, or out of range."""

    @classmethod
    def unknown_option(
        cls, name: str, value: object, valid: Iterable[str]
    ) -> "ConfigurationError":
        valid_list = ", ".join(repr(v) for v in valid)
        return cls(
            f"{value!r} is not a valid value for {name!r}. Valid values are: {valid_list}."
        )

    @classmethod
    def out_of_range(
        cls, name: str, value: object, low: float, high: float
    ) -> "ConfigurationError":
        return cls(
            f"{name}={value!r} is out of range; it must satisfy {low} <= {name} <= {high}."
        )


class NotFittedError(EdaPrepError):
    """``transform`` was called before ``fit``."""

    @classmethod
    def for_object(cls, obj: object) -> "NotFittedError":
        name = type(obj).__name__
        return cls(
            f"This {name} instance is not fitted yet. Call 'fit' with training data "
            f"before calling 'transform'. If you meant to fit and transform in one "
            f"step, use 'fit_transform'."
        )


class SchemaError(EdaPrepError):
    """The frame passed to ``transform`` disagrees with the frame seen at ``fit``.

    Tolerating this silently is how train/serve skew becomes invisible, so it is an
    error by default.  Set ``Config(on_unknown_columns="ignore")`` to downgrade extra
    columns to a reported warning.
    """

    @classmethod
    def missing_columns(
        cls, missing: Sequence[str], transformer: Optional[str] = None
    ) -> "SchemaError":
        where = f" required by {transformer}" if transformer else ""
        shown = ", ".join(repr(c) for c in list(missing)[:10])
        more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        return cls(
            f"{len(missing)} column(s){where} are missing from the input: {shown}{more}. "
            f"The frame passed to 'transform' must contain every column present at 'fit' "
            f"time. If these columns are genuinely absent, refit the pipeline on data "
            f"with the same schema."
        )

    @classmethod
    def unexpected_columns(cls, extra: Sequence[str]) -> "SchemaError":
        shown = ", ".join(repr(c) for c in list(extra)[:10])
        more = f" (and {len(extra) - 10} more)" if len(extra) > 10 else ""
        return cls(
            f"{len(extra)} column(s) are present now but were not seen at 'fit' time: "
            f"{shown}{more}. These columns cannot be processed because no statistics "
            f"were learned for them. Either drop them before calling 'transform', or "
            f"set Config(on_unknown_columns='ignore') to pass them through untouched."
        )


class DataError(EdaPrepError):
    """The data cannot support the requested operation."""


class EmptyDataError(DataError):
    """A dataset with no rows or no columns was supplied where data is required."""

    @classmethod
    def no_rows(cls, operation: str) -> "EmptyDataError":
        return cls(
            f"Cannot {operation}: the dataset has 0 rows. Statistics such as means, "
            f"quantiles and category frequencies are undefined on an empty frame."
        )

    @classmethod
    def no_columns(cls, operation: str) -> "EmptyDataError":
        return cls(f"Cannot {operation}: the dataset has 0 columns.")


class TransformationError(DataError):
    """A mathematical transformation is invalid for the data it was given."""

    @classmethod
    def non_positive(
        cls, column: str, transform: str, n_bad: int, suggestion: str
    ) -> "TransformationError":
        return cls(
            f"Column {column!r} contains {n_bad} non-positive value(s), so the "
            f"{transform} transformation cannot be applied. {suggestion}"
        )

    @classmethod
    def degenerate(cls, column: str, transform: str, reason: str) -> "TransformationError":
        return cls(
            f"The {transform} transformation cannot be fitted on column {column!r}: "
            f"{reason}. Configure a different strategy for this column, for example "
            f"config.column({column!r}).transform = 'none'."
        )


class LeakageError(EdaPrepError):
    """An operation that may only run at fit time was reached at transform time.

    This is an internal invariant check.  Seeing it means a transformer tried to learn
    a statistic from data it was not fitted on, which is exactly the defect class the
    library exists to prevent.
    """

    @classmethod
    def fit_time_only(cls, operation: str, transformer: str) -> "LeakageError":
        return cls(
            f"{transformer} attempted {operation} during 'transform'. Statistics may "
            f"only be learned during 'fit'; learning them at transform time would let "
            f"validation or test data influence the transformation. This is a bug in "
            f"the transformer, please report it."
        )

    @classmethod
    def target_required(cls, transformer: str) -> "LeakageError":
        return cls(
            f"{transformer} requires the target ('y') at fit time but none was given. "
            f"Pass y to 'fit', or set the target on the pipeline, or configure a "
            f"strategy that does not use the target (for example encoding='onehot' "
            f"instead of 'target')."
        )
