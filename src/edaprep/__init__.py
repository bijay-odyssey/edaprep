"""edaprep: transparent, leakage-safe EDA and ML preprocessing.

Quick start
-----------

Understand a dataset::

    import edaprep

    profile = edaprep.profile(df, target="churn")
    print(profile.summary())

    report = edaprep.EDA(df, target="churn").analyze()
    print(report.summary())

Prepare it, automatically but not opaquely::

    pipe = edaprep.AutoPipeline(target="churn", model_family="tree", random_state=42)
    pipe.fit(train_df)
    pipe.explain()                    # why every column was treated as it was

    X_train = pipe.transform(train_df)
    X_test  = pipe.transform(test_df)  # same fitted statistics, no leakage

    print(pipe.report_.summary())

Or say exactly what should happen::

    pipe = (
        edaprep.Pipeline(target="churn")
        .handle_missing()
        .handle_outliers(strategy="clip")
        .encode_categorical()
        .scale_numeric()
    )
    X = pipe.fit_transform(train_df)

Design guarantees
-----------------
* Every learned statistic is fitted on the training frame only; ``transform`` is a
  pure function of that fitted state.
* Every automatic decision is inspectable (``pipe.plan_``), explainable
  (``pipe.explain()``), overridable (``config.column("age").imputation = "mean"``)
  and reproducible (``random_state``, serialisable plan and report).
* Nothing is silently discarded.  Dropped columns, imputed values, grouped
  categories and clipped rows are all counted and reported.
"""

from ._version import __version__
from .config import AUTO, ColumnConfig, Config, Thresholds
from .core.base import Transformer
from .core.context import FitContext
from .core.pipeline import AutoPipeline, Pipeline
from .eda.analyzer import EDA, EDAReport
from .exceptions import (
    ConfigurationError,
    DataError,
    EdaPrepError,
    EmptyDataError,
    LeakageError,
    NotFittedError,
    SchemaError,
    TransformationError,
)
from .planning.decisions import Decision, Plan, PlannedStep
from .planning.planner import Planner
from .planning.rules import Rule, RuleSet, default_rules
from .preprocessing import (
    CategoricalEncoder,
    ColumnDropper,
    ConstantFilter,
    CorrelationFilter,
    DataTypeInference,
    DateTimeExpander,
    DistributionTransformer,
    DuplicateColumnFilter,
    DuplicateRowHandler,
    FrequencyEncoder,
    MissingIndicator,
    MissingnessFilter,
    MissingValueHandler,
    OneHotEncoder,
    OrdinalEncoder,
    OutlierHandler,
    RareCategoryGrouper,
    Scaler,
    TargetEncoder,
    TextColumnHandler,
    VarianceFilter,
    detect_outliers,
)
from .profiling.profiler import ColumnProfile, DatasetProfile, profile
from .reporting.report import Report
from .types import AnalysisLevel, ModelFamily, SemanticType, Severity, Stage

__all__ = [
    "__version__",
    # entry points
    "profile",
    "EDA",
    "AutoPipeline",
    "Pipeline",
    # configuration
    "Config",
    "ColumnConfig",
    "Thresholds",
    "AUTO",
    # planning
    "Planner",
    "Plan",
    "PlannedStep",
    "Decision",
    "Rule",
    "RuleSet",
    "default_rules",
    # results
    "DatasetProfile",
    "ColumnProfile",
    "Report",
    "EDAReport",
    # types
    "SemanticType",
    "ModelFamily",
    "Stage",
    "Severity",
    "AnalysisLevel",
    # extension
    "Transformer",
    "FitContext",
    # transformers
    "DataTypeInference",
    "DateTimeExpander",
    "DuplicateRowHandler",
    "MissingValueHandler",
    "MissingIndicator",
    "OutlierHandler",
    "detect_outliers",
    "CategoricalEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "FrequencyEncoder",
    "TargetEncoder",
    "RareCategoryGrouper",
    "Scaler",
    "DistributionTransformer",
    "TextColumnHandler",
    "ColumnDropper",
    "ConstantFilter",
    "MissingnessFilter",
    "DuplicateColumnFilter",
    "CorrelationFilter",
    "VarianceFilter",
    # exceptions
    "EdaPrepError",
    "ConfigurationError",
    "NotFittedError",
    "SchemaError",
    "DataError",
    "EmptyDataError",
    "TransformationError",
    "LeakageError",
]


def __getattr__(name: str):
    """Lazily expose optional subpackages.

    ``edaprep.visualization`` needs matplotlib, which is an optional dependency.
    Importing it eagerly would make ``import edaprep`` fail for users who installed
    only the core, so it is resolved on first access with a message that says what to
    install.
    """
    if name == "visualization":
        try:
            # importlib rather than `from . import visualization`: the latter goes
            # through _handle_fromlist, which calls getattr on this module again and
            # re-enters __getattr__, recursing until the stack blows instead of
            # surfacing the ImportError.
            import importlib

            module = importlib.import_module("edaprep.visualization")
        except ImportError as exc:
            raise ImportError(
                "edaprep.visualization requires matplotlib, which is an optional "
                "dependency. Install it with: pip install 'edaprep[visualization]'"
            ) from exc
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
