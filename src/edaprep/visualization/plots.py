"""Optional matplotlib renderers.

Visualisation is not the core of the library, and matplotlib is an optional dependency:
importing ``edaprep`` must work without it.  Everything here is reached through
``edaprep.visualization``, which is resolved lazily by ``edaprep.__getattr__`` and
raises an install hint if matplotlib is absent.

Each function takes an ``ax`` and returns it, so plots compose into a caller's figure
rather than owning the figure.  Nothing calls ``plt.show()``: that decision belongs to
the caller, and calling it inside a library is what makes notebooks and scripts behave
differently.

Colours are left to the active matplotlib style.  Hard-coding a palette would fight the
user's own theme and break in dark mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..profiling.profiler import DatasetProfile
from ..profiling.profiler import profile as profile_dataset
from ..types import SemanticType

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "missing_matrix",
    "missing_bar",
    "histogram",
    "boxplot",
    "correlation_heatmap",
    "category_bar",
    "target_distribution",
    "feature_target",
    "plot_profile",
]


def _plt():
    """Import pyplot on demand, with an install hint if it is missing.

    matplotlib is imported here rather than at module scope so that
    ``import edaprep.visualization`` itself works without it -- only actually drawing
    something requires the extra.  This is where the error belongs, and where the
    message has to be useful.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Plotting requires matplotlib, which is an optional dependency of "
            "edaprep. Install it with: pip install 'edaprep[visualization]'"
        ) from exc
    return plt


def _axes(ax: "Optional[Axes]", figsize=(8, 5)) -> "Axes":
    if ax is not None:
        return ax
    return _plt().subplots(figsize=figsize)[1]


def missing_matrix(
    data: pd.DataFrame,
    ax: "Optional[Axes]" = None,
    max_rows: int = 2000,
    random_state: Optional[int] = 0,
) -> "Axes":
    """Nullity matrix: one stripe per row, dark where a value is present.

    The familiar ``missingno.matrix`` plot.  Large frames are subsampled: the plot has
    a fixed pixel budget, so drawing 500,000 rows renders the same picture far more
    slowly.
    """
    frame = data
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=random_state).sort_index()
    ax = _axes(ax, figsize=(min(20, 0.4 * frame.shape[1] + 4), 6))
    ax.imshow(
        frame.notna().to_numpy(),
        aspect="auto",
        interpolation="none",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(frame.shape[1]))
    ax.set_xticklabels(frame.columns, rotation=90, fontsize=8)
    ax.set_ylabel(f"rows (n={len(frame):,})")
    ax.set_title("Missing-value matrix (light = missing)")
    return ax


def missing_bar(data: pd.DataFrame, ax: "Optional[Axes]" = None, top: int = 30) -> "Axes":
    """Missing fraction per column, worst first."""
    fractions = data.isna().mean().sort_values(ascending=False)
    fractions = fractions[fractions > 0].head(top)
    ax = _axes(ax, figsize=(8, max(3, 0.3 * len(fractions) + 1)))
    if fractions.empty:
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center")
        ax.set_axis_off()
        return ax
    ax.barh(range(len(fractions)), fractions.to_numpy())
    ax.set_yticks(range(len(fractions)))
    ax.set_yticklabels(fractions.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("fraction missing")
    ax.set_title("Missingness by column")
    return ax


def histogram(
    data: pd.DataFrame, column: str, bins: int = 40, ax: "Optional[Axes]" = None
) -> "Axes":
    """Histogram with the skewness annotated, as the usual loop does."""
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    ax = _axes(ax)
    ax.hist(values.to_numpy(), bins=bins)
    skew = float(values.skew()) if len(values) > 2 else float("nan")
    ax.set_title(f"{column}   skew {skew:.2f}" if np.isfinite(skew) else str(column))
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    return ax


def boxplot(
    data: pd.DataFrame, columns: Sequence[str], ax: "Optional[Axes]" = None
) -> "Axes":
    """Box plots for several columns side by side."""
    series = [pd.to_numeric(data[c], errors="coerce").dropna().to_numpy() for c in columns]
    ax = _axes(ax, figsize=(max(6, 1.2 * len(columns)), 5))
    ax.boxplot(series)
    ax.set_xticks(range(1, len(columns) + 1))
    ax.set_xticklabels(list(columns))
    ax.set_title("Distribution and outliers")
    ax.tick_params(axis="x", rotation=45)
    return ax


def correlation_heatmap(
    corr: pd.DataFrame, ax: "Optional[Axes]" = None, annotate: Optional[bool] = None
) -> "Axes":
    """Correlation heatmap on a symmetric, zero-centred scale.

    The scale is fixed to [-1, 1] rather than the data range: a heatmap whose colour
    scale floats makes a maximum correlation of 0.2 look identical to one of 0.99.
    """
    ax = _axes(ax, figsize=(min(14, 0.5 * len(corr) + 3),) * 2)
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    if annotate is None:
        annotate = len(corr) <= 12
    if annotate:
        for i in range(len(corr)):
            for j in range(len(corr)):
                value = corr.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.figure.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation")
    return ax


def category_bar(
    data: pd.DataFrame, column: str, top: int = 20, ax: "Optional[Axes]" = None
) -> "Axes":
    """Category frequencies, with the tail collapsed into one bar.

    Plotting 300 categories produces an unreadable axis; collapsing the tail and
    labelling it keeps the total honest.
    """
    counts = data[column].value_counts(dropna=False)
    shown = counts.head(top)
    remainder = int(counts.iloc[top:].sum())
    labels = [str(i) for i in shown.index]
    values = list(shown.to_numpy())
    if remainder:
        labels.append(f"({len(counts) - top} others)")
        values.append(remainder)
    ax = _axes(ax, figsize=(8, max(3, 0.3 * len(labels) + 1)))
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title(f"{column}  ({counts.shape[0]} levels)")
    return ax


def target_distribution(
    data: pd.DataFrame, target: str, ax: "Optional[Axes]" = None
) -> "Axes":
    """Target distribution: a bar chart for classification, a histogram otherwise."""
    series = data[target].dropna()
    ax = _axes(ax)
    if series.nunique() <= 20:
        counts = series.value_counts()
        ax.bar(range(len(counts)), counts.to_numpy())
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(i) for i in counts.index])
        ratio = counts.min() / counts.max()
        ax.set_title(f"{target}  (minority/majority {ratio:.3f})")
    else:
        ax.hist(pd.to_numeric(series, errors="coerce").dropna().to_numpy(), bins=40)
        ax.set_title(str(target))
    ax.set_ylabel("count")
    return ax


def feature_target(
    data: pd.DataFrame, column: str, target: str, ax: "Optional[Axes]" = None
) -> "Axes":
    """Feature against target: box-by-class, scatter, or mean-by-category."""
    ax = _axes(ax)
    feature, response = data[column], data[target]
    feature_numeric = pd.api.types.is_numeric_dtype(feature.dtype)
    target_categorical = response.nunique() <= 20

    if feature_numeric and target_categorical:
        groups, labels = [], []
        for level in sorted(response.dropna().unique(), key=str):
            values = pd.to_numeric(feature[response == level], errors="coerce").dropna()
            if len(values):
                groups.append(values.to_numpy())
                labels.append(str(level))
        ax.boxplot(groups)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlabel(target)
        ax.set_ylabel(column)
    elif feature_numeric:
        ax.scatter(
            pd.to_numeric(feature, errors="coerce"),
            pd.to_numeric(response, errors="coerce"),
            s=6,
            alpha=0.4,
        )
        ax.set_xlabel(column)
        ax.set_ylabel(target)
    else:
        means = (
            pd.to_numeric(response, errors="coerce")
            .groupby(feature, observed=True)
            .mean()
            .sort_values(ascending=False)
            .head(25)
        )
        ax.barh(range(len(means)), means.to_numpy())
        ax.set_yticks(range(len(means)))
        ax.set_yticklabels([str(i) for i in means.index], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"mean {target}")
    ax.set_title(f"{column} vs {target}")
    return ax


def plot_profile(
    data: pd.DataFrame,
    target: Optional[str] = None,
    profile: Optional[DatasetProfile] = None,
    max_numeric: int = 6,
    max_categorical: int = 4,
    config: Optional[Config] = None,
) -> "Figure":
    """One overview figure: missingness, top distributions, categories, target.

    Replaces the usual per-column plotting loops, which emit one figure per column
    and make a 400-column dataset unreadable.  Columns are chosen by how much they
    matter (missingness, skewness, target association), not by position.
    """
    plt = _plt()
    config = config or Config()
    profile = profile or profile_dataset(data, target=target, config=config)

    numeric = [
        c
        for c in profile.column_order
        if profile.columns[c].semantic is SemanticType.NUMERIC and c != target
    ]
    numeric.sort(
        key=lambda c: abs(profile.columns[c].skew)
        if np.isfinite(profile.columns[c].skew)
        else 0.0,
        reverse=True,
    )
    numeric = numeric[:max_numeric]

    categorical = [
        c
        for c in profile.column_order
        if profile.columns[c].semantic
        in (SemanticType.CATEGORICAL, SemanticType.BINARY, SemanticType.ORDINAL)
        and c != target
    ][:max_categorical]

    panels: List[tuple] = [("missing", None)]
    panels += [("hist", c) for c in numeric]
    panels += [("cat", c) for c in categorical]
    if target is not None:
        panels.append(("target", target))

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4 * rows))
    flat = np.atleast_1d(axes).ravel()

    for ax, (kind, column) in zip(flat, panels):
        if kind == "missing":
            missing_bar(data, ax=ax)
        elif kind == "hist":
            histogram(data, column, ax=ax)
        elif kind == "cat":
            category_bar(data, column, ax=ax)
        else:
            target_distribution(data, column, ax=ax)
    for ax in flat[len(panels) :]:
        ax.set_axis_off()

    figure.tight_layout()
    return figure
