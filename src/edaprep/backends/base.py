"""The backend protocol.

A narrow seam between "what the library wants done to a frame" and "how a particular
dataframe implementation does it".  ``pandas_backend.PandasBackend`` is the only
implementation today.

Why it is this small
--------------------
This is not a general dataframe abstraction, and it is deliberately not trying to be.
Wrapping every pandas call would be abstraction for its own sake: it would slow the
common path, obscure the code, and buy nothing until a second backend exists.

What it does cover is the handful of operations that are (a) hot enough to be worth
routing through a seam and (b) the ones an Arrow or Polars implementation would
genuinely do differently.  Everything else in the library calls pandas directly.

The motivation is concrete rather than speculative: tabular ML frames routinely reach
500,000 x 400, and an Arrow-backed implementation is a foreseeable need for frames of
that shape and larger.  Nothing here is used to support a second backend yet.

Before writing one
------------------
Read ``docs/performance.md`` first.  The one place in this library where a hand-written
replacement for pandas looked obviously worthwhile turned out to be 2.1x *slower* than
the pandas code it replaced, and was deleted.  Measure before committing to an
implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Backend", "get_backend", "register_backend"]


class Backend(ABC):
    """Operations the library routes through a backend.

    Implementations receive and return their own native frame and column types; the
    library treats them as opaque except through these methods.
    """

    #: Short name, used by :func:`get_backend`.
    name: str = "base"

    # -- introspection ---------------------------------------------------------------

    @abstractmethod
    def is_frame(self, obj: Any) -> bool:
        """True when ``obj`` is a frame this backend understands."""

    @abstractmethod
    def shape(self, frame: Any) -> Tuple[int, int]:
        """``(n_rows, n_columns)``."""

    @abstractmethod
    def column_names(self, frame: Any) -> List[str]:
        """Column names, in order."""

    @abstractmethod
    def dtype_of(self, frame: Any, column: str) -> str:
        """A string naming the column's storage type."""

    @abstractmethod
    def memory_usage(self, frame: Any, deep: bool = True) -> Dict[str, int]:
        """Per-column bytes, plus a ``"total"`` key."""

    # -- column access ----------------------------------------------------------------

    @abstractmethod
    def get_column(self, frame: Any, column: str) -> Any:
        """One column, in the backend's native representation."""

    @abstractmethod
    def to_float_array(self, frame: Any, column: str) -> np.ndarray:
        """One column as a float64 NumPy array, with missing values as NaN.

        Every numeric kernel in the library ultimately works on NumPy, so this is the
        single conversion point a backend must provide.
        """

    @abstractmethod
    def select(self, frame: Any, columns: Sequence[str]) -> Any:
        """A frame restricted to ``columns``, without copying where possible."""

    @abstractmethod
    def assign(self, frame: Any, columns: Mapping[str, Any]) -> Any:
        """A new frame with ``columns`` replaced or appended.

        Must not mutate ``frame``.  Implementations should pass untouched columns
        through by reference rather than copying the whole frame -- the copy discipline
        described in ``docs/architecture.md`` section 6.
        """

    # -- aggregation -------------------------------------------------------------------

    @abstractmethod
    def null_mask(self, frame: Any, column: str) -> np.ndarray:
        """Boolean array: True where the value is missing."""

    @abstractmethod
    def n_unique(self, frame: Any, column: str, dropna: bool = True) -> int:
        """Distinct value count."""

    @abstractmethod
    def value_counts(
        self, frame: Any, column: str, dropna: bool = True
    ) -> List[Tuple[Any, int]]:
        """``(value, count)`` pairs, most frequent first."""

    @abstractmethod
    def quantiles(
        self, frame: Any, columns: Sequence[str], levels: Sequence[float]
    ) -> np.ndarray:
        """Quantile matrix of shape ``(len(levels), len(columns))``."""

    @abstractmethod
    def group_mean(self, frame: Any, value_column: str, group_column: str) -> Dict[Any, float]:
        """Mean of ``value_column`` per level of ``group_column``."""

    @abstractmethod
    def duplicated_rows(self, frame: Any, subset: Optional[Sequence[str]] = None) -> np.ndarray:
        """Boolean array: True where the row repeats an earlier one."""

    # -- construction --------------------------------------------------------------------

    @abstractmethod
    def concat_columns(self, frames: Iterable[Any]) -> Any:
        """Join frames side by side.  Indexes are assumed aligned."""

    @abstractmethod
    def take_rows(self, frame: Any, mask: np.ndarray) -> Any:
        """The rows where ``mask`` is True."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"


_REGISTRY: Dict[str, Backend] = {}


def register_backend(backend: Backend) -> Backend:
    """Make a backend available to :func:`get_backend`."""
    _REGISTRY[backend.name] = backend
    return backend


def get_backend(name: str = "pandas") -> Backend:
    """Look up a registered backend by name."""
    if name not in _REGISTRY:
        if name == "pandas":
            from .pandas_backend import PandasBackend

            return register_backend(PandasBackend())
        available = ", ".join(sorted(_REGISTRY)) or "pandas"
        raise ValueError(
            f"No backend named {name!r} is registered. Available: {available}. "
            f"Register one with edaprep.backends.register_backend(MyBackend())."
        )
    return _REGISTRY[name]
