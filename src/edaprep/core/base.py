"""The transformer contract.

Four rules, enforced here rather than documented elsewhere, are what make leakage
structurally impossible instead of merely discouraged:

1. All learned state lives in attributes ending in ``_``, written only inside ``_fit``.
   :meth:`Transformer.transform` reads them and nothing else.
2. ``transform`` never computes a statistic over its input.  A test greps the
   ``_transform`` bodies for aggregation calls and fails on a hit.
3. ``fit_transform`` equals ``fit().transform()`` unless the transformer declares
   ``cross_fitted = True`` (target encoding), where the difference is the point.
4. ``uses_target = True`` is a declaration, and reaching such a transformer without
   ``y`` raises :class:`~edaprep.exceptions.LeakageError` rather than silently
   degrading.

scikit-learn interoperability
-----------------------------
``get_params``/``set_params`` are implemented by introspecting ``__init__``, exactly as
scikit-learn does, so edaprep transformers drop into an ``sklearn.pipeline.Pipeline``.
``BaseEstimator`` is deliberately *not* subclassed: scikit-learn is an optional
dependency, and reimplementing the ~40-line protocol costs nothing at import time while
inheriting from it would make the core package depend on it.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import LeakageError, NotFittedError, SchemaError
from ..types import Stage
from .context import FitContext

__all__ = ["Transformer", "ColumnTransformerMixin", "check_is_fitted"]


def check_is_fitted(transformer: "Transformer") -> None:
    """Raise :class:`NotFittedError` unless ``fit`` has completed."""
    if not getattr(transformer, "_is_fitted", False):
        raise NotFittedError.for_object(transformer)


class Transformer(ABC):
    """Base class for every preprocessing step.

    Subclasses implement :meth:`_fit` and :meth:`_transform`.  The public
    :meth:`fit`/:meth:`transform` wrappers handle validation, journalling and the
    fitted-state flag, so subclasses cannot forget them.
    """

    #: The pipeline stage this transformer belongs to.  Used for ordering and reporting.
    stage: Stage = Stage.FEATURE_ENGINEERING
    #: Declares that ``fit`` consumes ``y``.  Audited; ``None`` y raises.
    uses_target: bool = False
    #: Declares that ``fit_transform`` legitimately differs from ``fit().transform()``.
    cross_fitted: bool = False

    def __init__(self, columns: Optional[Sequence[str]] = None) -> None:
        self.columns = list(columns) if columns is not None else None
        self._is_fitted = False
        self.columns_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    # -- subclass hooks -----------------------------------------------------------

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        """Learn state.  Must write only ``*_`` attributes."""

    @abstractmethod
    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        """Apply learned state.  Must not compute statistics over ``X``."""

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        """Which columns this transformer applies to when ``columns`` was not given.

        The default is "all of them"; type-aware transformers override this to consult
        the profile, which is how a numeric scaler avoids touching a text column.
        """
        return [str(c) for c in X.columns]

    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext
    ) -> pd.DataFrame:
        """Override only when ``cross_fitted`` is True."""
        self._fit(X, y, context)
        return self._transform(X, context)

    # -- public API ----------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        context: Optional[FitContext] = None,
    ) -> "Transformer":
        context = self._prepare(X, y, context)
        self.feature_names_in_ = [str(c) for c in X.columns]
        self.columns_ = self._resolve_columns(X, context)
        self._fit(X, y, context)
        self._is_fitted = True
        self.feature_names_out_ = self._compute_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame, context: Optional[FitContext] = None) -> pd.DataFrame:
        check_is_fitted(self)
        context = context or FitContext()
        self._check_schema(X, context)
        return self._transform(X, context)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        context: Optional[FitContext] = None,
    ) -> pd.DataFrame:
        context = self._prepare(X, y, context)
        self.feature_names_in_ = [str(c) for c in X.columns]
        self.columns_ = self._resolve_columns(X, context)
        out = self._fit_transform(X, y, context)
        self._is_fitted = True
        self.feature_names_out_ = self._compute_feature_names_out()
        return out

    # -- introspection --------------------------------------------------------------

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Output column names, in order.  scikit-learn compatible."""
        check_is_fitted(self)
        if input_features is not None:
            expected = list(self.feature_names_in_)
            if list(input_features) != expected:
                raise SchemaError(
                    f"input_features does not match the columns seen at fit time. "
                    f"Expected {len(expected)} names starting with "
                    f"{expected[:3]}, got {len(list(input_features))}."
                )
        return np.asarray(self.feature_names_out_, dtype=object)

    def _compute_feature_names_out(self) -> List[str]:
        """Default: the input columns, unchanged.  Overridden by column-adding steps."""
        return list(self.feature_names_in_)

    @classmethod
    def _param_names(cls) -> List[str]:
        signature = inspect.signature(cls.__init__)
        return sorted(
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Constructor parameters.  scikit-learn compatible."""
        out: Dict[str, Any] = {}
        for name in self._param_names():
            value = getattr(self, name, None)
            out[name] = value
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for sub_name, sub_value in value.get_params(deep=True).items():
                    out[f"{name}__{sub_name}"] = sub_value
        return out

    def set_params(self, **params: Any) -> "Transformer":
        """Set constructor parameters.  scikit-learn compatible."""
        if not params:
            return self
        valid = self._param_names()
        nested: Dict[str, Dict[str, Any]] = {}
        for key, value in params.items():
            head, _, tail = key.partition("__")
            if head not in valid:
                raise ValueError(
                    f"Invalid parameter {head!r} for {type(self).__name__}. "
                    f"Valid parameters are: {', '.join(valid)}."
                )
            if tail:
                nested.setdefault(head, {})[tail] = value
            else:
                setattr(self, head, value)
        for head, sub_params in nested.items():
            getattr(self, head).set_params(**sub_params)
        return self

    # -- internals -------------------------------------------------------------------

    def _prepare(
        self, X: pd.DataFrame, y: Optional[pd.Series], context: Optional[FitContext]
    ) -> FitContext:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__}.fit expects a pandas DataFrame, got "
                f"{type(X).__name__}."
            )
        if self.uses_target and y is None:
            raise LeakageError.target_required(type(self).__name__)
        if y is not None and len(y) != len(X):
            raise SchemaError(
                f"X has {len(X)} rows but y has {len(y)}. They must align row-for-row; "
                f"a mismatch usually means one of them was filtered without the other."
            )
        return context or FitContext()

    def _resolve_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        if self.columns is None:
            return self._select_columns(X, context)
        present = set(map(str, X.columns))
        missing = [c for c in self.columns if c not in present]
        if missing:
            raise SchemaError.missing_columns(missing, type(self).__name__)
        return [str(c) for c in self.columns]

    def _check_schema(self, X: pd.DataFrame, context: FitContext) -> None:
        """Transform-time columns must agree with fit-time columns.

        Tolerating a mismatch silently is how train/serve skew becomes invisible, so
        missing columns are always an error and extra columns are an error unless the
        user opted into ``on_unknown_columns="ignore"``.
        """
        present = set(map(str, X.columns))
        needed = [c for c in self.columns_ if c not in present]
        if needed:
            raise SchemaError.missing_columns(needed, type(self).__name__)
        if context.config.on_unknown_columns == "error" and self.feature_names_in_:
            known = set(self.feature_names_in_)
            extra = [c for c in map(str, X.columns) if c not in known]
            if extra:
                raise SchemaError.unexpected_columns(extra)

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        shown = ", ".join(
            f"{k}={v!r}" for k, v in sorted(params.items()) if v is not None and k != "columns"
        )
        scope = ""
        if self.columns is not None:
            scope = f"columns={len(self.columns)}"
        elif self._is_fitted:
            scope = f"columns_={len(self.columns_)}"
        inner = ", ".join(p for p in (shown, scope) if p)
        return f"{type(self).__name__}({inner})"


class ColumnTransformerMixin:
    """Helpers for transformers that rewrite a subset of columns in place.

    The output frame is assembled from the input's column blocks with only the touched
    columns replaced.  Untouched columns are passed through by reference, so a pipeline
    that touches 5 of 400 columns allocates 5 columns rather than copying the frame --
    the ``df.copy()`` habit that triples peak memory on wide frames in notebook practice.
    """

    @staticmethod
    def _rebuild(
        X: pd.DataFrame,
        replacements: Dict[str, pd.Series],
        added: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Return a new frame with ``replacements`` applied and ``added`` appended."""
        if not replacements and not added:
            return X
        data: Dict[str, Any] = {}
        for name in X.columns:
            key = str(name)
            data[key] = replacements.get(key, X[name])
        for key, series in (added or {}).items():
            data[key] = series
        return pd.DataFrame(data, index=X.index, copy=False)

    @staticmethod
    def _numeric_values(series: pd.Series) -> np.ndarray:
        """A float64 view of a column, with missing values as NaN."""
        try:
            return series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        except (TypeError, ValueError):
            return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
