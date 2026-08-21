"""Optional matplotlib visualisations.

Requires the ``visualization`` extra::

    pip install "edaprep[visualization]"

Reached through ``edaprep.visualization``, which resolves lazily so that importing
``edaprep`` never requires matplotlib.
"""

from .plots import (
    boxplot,
    category_bar,
    correlation_heatmap,
    feature_target,
    histogram,
    missing_bar,
    missing_matrix,
    plot_profile,
    target_distribution,
)

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
