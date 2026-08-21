"""Preprocessing transformers.

Each is independently usable and independently testable.  The planner assembles them;
nothing here requires the planner.
"""

from .casting import DataTypeInference
from .datetime_features import DateTimeExpander
from .duplicates import DuplicateRowHandler, duplicate_report
from .encoding import (
    CategoricalEncoder,
    FrequencyEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareCategoryGrouper,
    TargetEncoder,
)
from .missing import MissingIndicator, MissingValueHandler
from .outliers import (
    IQRDetector,
    ModifiedZScoreDetector,
    OutlierHandler,
    PercentileDetector,
    ZScoreDetector,
    detect_outliers,
)
from .scaling import Scaler
from .selection import (
    ColumnDropper,
    ConstantFilter,
    CorrelationFilter,
    DuplicateColumnFilter,
    MissingnessFilter,
    VarianceFilter,
)
from .text import TextColumnHandler
from .transformations import DistributionTransformer

__all__ = [
    "DataTypeInference",
    "DateTimeExpander",
    "DuplicateRowHandler",
    "duplicate_report",
    "MissingValueHandler",
    "MissingIndicator",
    "OutlierHandler",
    "IQRDetector",
    "ZScoreDetector",
    "ModifiedZScoreDetector",
    "PercentileDetector",
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
]
