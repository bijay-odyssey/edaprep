"""Lightweight text handling.

Per the design goal, text processing stays minimal in v1: detect likely text columns, report
their shape, and get them out of the way of the numeric pipeline.  Running tokenisation
or TF-IDF automatically would be expensive, opinionated, and almost never what the user
wants by default.

:class:`TextColumnHandler` is also the extension point.  A future ``TextVectorizer``
slots into ``Stage.ENCODE`` and consumes the same column selection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..core.base import ColumnTransformerMixin, Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError
from ..types import SemanticType, Severity, Stage

__all__ = ["TextColumnHandler", "text_statistics"]


def text_statistics(series: pd.Series, sample_size: int = 1000) -> Dict[str, float]:
    """Cheap shape statistics for a text column."""
    values = series.dropna()
    if len(values) > sample_size:
        values = values.iloc[:sample_size]
    if values.empty:
        return {"mean_length": 0.0, "max_length": 0.0, "mean_tokens": 0.0, "n_empty": 0}
    as_str = values.astype(str)
    lengths = as_str.str.len()
    tokens = as_str.str.split().str.len()
    return {
        "mean_length": float(lengths.mean()),
        "max_length": float(lengths.max()),
        "mean_tokens": float(tokens.mean()),
        "n_empty": int((as_str.str.strip() == "").sum()),
    }


class TextColumnHandler(Transformer, ColumnTransformerMixin):
    """Handle text columns without pretending to do NLP.

    Parameters
    ----------
    strategy :
        ``"drop"`` (default) removes text columns, since a raw string cannot be
        consumed by a tabular model and silently one-hot encoding thousands of unique
        sentences is worse than useless.  ``"length_features"`` replaces each column
        with cheap surface statistics (character count, word count), which sometimes
        carry real signal.  ``"keep"`` leaves them untouched for the caller to handle.
    """

    stage = Stage.ENCODE

    def __init__(
        self, columns: Optional[Sequence[str]] = None, strategy: str = "drop"
    ) -> None:
        super().__init__(columns)
        self.strategy = strategy

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        out: List[str] = []
        for name in map(str, X.columns):
            if name == context.target:
                continue
            cp = context.column_profile(name)
            if cp is not None and cp.semantic is SemanticType.TEXT:
                out.append(name)
        return out

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        if self.strategy not in ("drop", "length_features", "keep"):
            raise ConfigurationError.unknown_option(
                "text strategy", self.strategy, ["drop", "length_features", "keep"]
            )
        self.statistics_ = {c: text_statistics(X[c]) for c in self.columns_}
        if self.columns_ and self.strategy == "drop":
            context.journal.warn(
                "text_columns_dropped",
                f"{len(self.columns_)} text column(s) were dropped: "
                f"{', '.join(repr(c) for c in self.columns_[:5])}. edaprep does not "
                f"vectorise text in this version. Set text_strategy='length_features' "
                f"for cheap surface features, or 'keep' to handle them yourself.",
                Severity.INFO,
                tuple(self.columns_),
                {"statistics": self.statistics_},
            )
        context.journal.record(
            self.stage,
            type(self).__name__,
            f"text_{self.strategy}",
            "fit",
            columns=self.columns_,
            effect={"statistics": self.statistics_},
        )

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        if self.strategy == "keep" or not self.columns_:
            return X
        present = [c for c in self.columns_ if c in map(str, X.columns)]
        if self.strategy == "drop":
            keep = [c for c in X.columns if str(c) not in present]
            return X[keep]

        added: Dict[str, pd.Series] = {}
        for column in present:
            as_str = X[column].astype(str)
            added[f"{column}__length"] = as_str.str.len().astype("float64")
            added[f"{column}__n_words"] = as_str.str.split().str.len().astype("float64")
        keep = {str(c): X[c] for c in X.columns if str(c) not in present}
        return pd.DataFrame({**keep, **added}, index=X.index, copy=False)

    def _compute_feature_names_out(self) -> List[str]:
        if self.strategy == "keep":
            return list(self.feature_names_in_)
        names = [c for c in self.feature_names_in_ if c not in self.columns_]
        if self.strategy == "length_features":
            for column in self.columns_:
                names.extend([f"{column}__length", f"{column}__n_words"])
        return names
