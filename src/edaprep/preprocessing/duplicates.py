"""Duplicate row handling.

Deliberately conservative.  Duplicate rows are not always errors.  In transactional data two identical rows are two identical events, and
deduplicating them destroys information and distorts class balance.

So the default is ``"report"``, and removal is a fit-time-only operation: dropping rows
inside ``transform`` would change how many predictions the caller gets back for a test
set, which nothing downstream expects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..core.base import Transformer
from ..core.context import FitContext
from ..exceptions import ConfigurationError
from ..types import Severity, Stage

__all__ = ["DuplicateRowHandler", "duplicate_report"]


def duplicate_report(
    frame: pd.DataFrame, subset: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    """Count duplicate rows without changing anything.

    ``n_duplicate_rows`` counts rows that repeat an earlier row (so a value appearing
    three times contributes two), while ``n_duplicated_groups`` counts the distinct
    values that repeat.  Both are reported because they answer different questions.
    """
    if len(frame) == 0:
        return {"n_duplicate_rows": 0, "n_duplicated_groups": 0, "fraction": 0.0}
    try:
        marked = frame.duplicated(subset=list(subset) if subset else None, keep="first")
        all_marked = frame.duplicated(subset=list(subset) if subset else None, keep=False)
    except TypeError:
        # Unhashable cell values; fall back to a string view, which is slower but works.
        as_str = frame.astype(str)
        marked = as_str.duplicated(subset=list(subset) if subset else None, keep="first")
        all_marked = as_str.duplicated(subset=list(subset) if subset else None, keep=False)
    n_dup = int(marked.sum())
    return {
        "n_duplicate_rows": n_dup,
        "n_duplicated_groups": int(all_marked.sum() - n_dup),
        "fraction": n_dup / len(frame),
    }


class DuplicateRowHandler(Transformer):
    """Detect, and optionally remove, exact duplicate rows.

    Parameters
    ----------
    strategy :
        ``"report"`` (default) measures and records only.  ``"remove"`` drops
        duplicates from the *training* frame during ``fit_transform``.  ``"ignore"``
        skips the check entirely, which is worth doing on very wide frames where
        hashing every row is not free.
    subset :
        Restrict the comparison to these columns.  Useful when a row is uniquely
        identified by a key and the rest is payload.
    keep :
        Which occurrence to keep, as in ``pandas.DataFrame.drop_duplicates``.

    Row removal is fit-time only
    ----------------------------
    ``transform`` never drops rows, whatever the strategy.  ``fit_transform`` does,
    because there the caller is holding the training frame and expects it to change.
    :meth:`duplicate_mask` exposes the mask for callers who want to align ``y``.
    """

    stage = Stage.DEDUPLICATE

    def __init__(
        self,
        strategy: str = "report",
        subset: Optional[Sequence[str]] = None,
        keep: str = "first",
    ) -> None:
        super().__init__(None)
        self.strategy = strategy
        self.subset = list(subset) if subset else None
        self.keep = keep

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        return list(self.subset) if self.subset else [str(c) for c in X.columns]

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        if self.strategy not in ("report", "remove", "ignore"):
            raise ConfigurationError.unknown_option(
                "duplicate strategy", self.strategy, ["report", "remove", "ignore"]
            )
        self.stats_: Dict[str, object] = {
            "n_duplicate_rows": 0,
            "n_duplicated_groups": 0,
            "fraction": 0.0,
        }
        if self.strategy == "ignore":
            return

        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as timer:
            self.stats_ = duplicate_report(X, self.subset)
            n_dup = int(self.stats_["n_duplicate_rows"])
            if n_dup:
                context.journal.warn(
                    "duplicate_rows",
                    f"{n_dup:,} duplicate row(s) "
                    f"({self.stats_['fraction']:.2%}) in the training data"
                    + (
                        ", removed."
                        if self.strategy == "remove"
                        else ". They were kept: repeated observations are legitimate in "
                        "transactional data. Set duplicate_strategy='remove' to drop "
                        "them."
                    ),
                    Severity.WARNING if self.strategy == "report" else Severity.INFO,
                    (),
                    dict(self.stats_),
                )
            timer.params = {"strategy": self.strategy, "subset": self.subset}
            timer.effect = dict(self.stats_)

    def duplicate_mask(self, X: pd.DataFrame) -> pd.Series:
        """Boolean mask of rows that duplicate an earlier row."""
        if self.strategy == "ignore":
            return pd.Series(False, index=X.index)
        try:
            return X.duplicated(subset=self.subset, keep=self.keep)
        except TypeError:
            return X.astype(str).duplicated(subset=self.subset, keep=self.keep)

    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext
    ) -> pd.DataFrame:
        self._fit(X, y, context)
        if self.strategy != "remove":
            return X
        mask = self.duplicate_mask(X)
        n = int(mask.sum())
        context.journal.record(
            self.stage,
            type(self).__name__,
            "remove_duplicate_rows",
            "fit",
            effect={"n_rows_removed": n, "n_rows_before": len(X), "n_rows_after": len(X) - n},
        )
        return X.loc[~mask]

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        if self.strategy == "ignore":
            return X
        stats = duplicate_report(X, self.subset)
        context.journal.record(
            self.stage,
            type(self).__name__,
            "report_duplicate_rows",
            "transform",
            effect={
                **stats,
                "note": (
                    "rows are never dropped during transform; removing them would "
                    "change how many predictions the caller receives"
                ),
            },
        )
        return X
