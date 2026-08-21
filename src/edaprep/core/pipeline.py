"""Pipelines: explicit and automatic.

:class:`Pipeline` is a plain sequence of transformers with a chainable builder, for
users who want to say exactly what happens.

:class:`AutoPipeline` profiles, plans, and then executes the plan.  Its distinguishing
feature is that the plan is inspectable *before* it runs and explainable *after*:
``pipeline.plan_``, ``pipeline.explain()``, ``pipeline.report()``.

Leakage
-------
``fit`` sees the training frame and nothing else.  Every learned statistic lives inside
a transformer's fitted state, and ``transform`` is a pure function of that state.  The
target is separated from the features at fit time and is never present in the output of
``transform``, so it cannot be accidentally fed to a model as a feature.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import Config
from ..exceptions import ConfigurationError, NotFittedError, SchemaError
from ..planning.decisions import Plan
from ..planning.planner import Planner
from ..profiling.profiler import DatasetProfile, profile as profile_dataset
from ..reporting.report import Report
from ..types import ModelFamily, Stage
from .base import Transformer, check_is_fitted
from .context import FitContext
from .journal import Journal

__all__ = ["Pipeline", "AutoPipeline"]

Step = Tuple[str, Transformer]


class Pipeline(Transformer):
    """An explicit sequence of transformers.

    Two ways to build one::

        Pipeline([("impute", MissingValueHandler()), ("scale", Scaler())])

        Pipeline().handle_missing().scale_numeric()

    The builder methods append a transformer and return ``self``, so they chain.  They
    are a convenience over the list form, not a separate mechanism.
    """

    stage = Stage.FEATURE_ENGINEERING

    def __init__(
        self,
        steps: Optional[Sequence[Union[Step, Transformer]]] = None,
        config: Optional[Config] = None,
        target: Optional[str] = None,
    ) -> None:
        super().__init__(None)
        self.steps: List[Step] = []
        self.config = config or Config()
        self.target = target
        for step in steps or ():
            if isinstance(step, tuple):
                self.add(step[1], name=step[0])
            else:
                self.add(step)

    # -- construction ---------------------------------------------------------------

    def add(self, transformer: Transformer, name: Optional[str] = None) -> "Pipeline":
        """Append a transformer.  Names are auto-generated when not given."""
        if not isinstance(transformer, Transformer):
            raise ConfigurationError(
                f"Pipeline steps must be edaprep Transformer instances, got "
                f"{type(transformer).__name__}. To use a scikit-learn transformer, "
                f"place the edaprep pipeline inside an sklearn Pipeline instead."
            )
        if name is None:
            base = _snake(type(transformer).__name__)
            existing = {n for n, _ in self.steps}
            name = base
            suffix = 2
            while name in existing:
                name = f"{base}_{suffix}"
                suffix += 1
        elif any(n == name for n, _ in self.steps):
            raise ConfigurationError(
                f"A step named {name!r} already exists in this pipeline. Step names "
                f"must be unique so that get_params()/set_params() can address them."
            )
        self.steps.append((name, transformer))
        return self

    # -- chainable builder ------------------------------------------------------------

    def infer_types(self, **kwargs) -> "Pipeline":
        from ..preprocessing.casting import DataTypeInference

        return self.add(DataTypeInference(**kwargs))

    def drop_columns(self, columns: Sequence[str], **kwargs) -> "Pipeline":
        from ..preprocessing.selection import ColumnDropper

        return self.add(ColumnDropper(columns, **kwargs))

    def handle_duplicates(self, **kwargs) -> "Pipeline":
        from ..preprocessing.duplicates import DuplicateRowHandler

        return self.add(DuplicateRowHandler(**kwargs))

    def expand_datetime(self, **kwargs) -> "Pipeline":
        from ..preprocessing.datetime_features import DateTimeExpander

        return self.add(DateTimeExpander(**kwargs))

    def flag_missing(self, **kwargs) -> "Pipeline":
        from ..preprocessing.missing import MissingIndicator

        return self.add(MissingIndicator(**kwargs))

    def handle_outliers(self, **kwargs) -> "Pipeline":
        from ..preprocessing.outliers import OutlierHandler

        return self.add(OutlierHandler(**kwargs))

    def handle_missing(self, **kwargs) -> "Pipeline":
        from ..preprocessing.missing import MissingValueHandler

        return self.add(MissingValueHandler(**kwargs))

    def transform_distributions(self, **kwargs) -> "Pipeline":
        from ..preprocessing.transformations import DistributionTransformer

        return self.add(DistributionTransformer(**kwargs))

    def group_rare_categories(self, **kwargs) -> "Pipeline":
        from ..preprocessing.encoding import RareCategoryGrouper

        return self.add(RareCategoryGrouper(**kwargs))

    def encode_categorical(self, **kwargs) -> "Pipeline":
        from ..preprocessing.encoding import CategoricalEncoder

        return self.add(CategoricalEncoder(**kwargs))

    def scale_numeric(self, **kwargs) -> "Pipeline":
        from ..preprocessing.scaling import Scaler

        return self.add(Scaler(**kwargs))

    def handle_text(self, **kwargs) -> "Pipeline":
        from ..preprocessing.text import TextColumnHandler

        return self.add(TextColumnHandler(**kwargs))

    def select_features(self, **kwargs) -> "Pipeline":
        from ..preprocessing.selection import CorrelationFilter

        return self.add(CorrelationFilter(**kwargs))

    def drop_constants(self, **kwargs) -> "Pipeline":
        from ..preprocessing.selection import ConstantFilter

        return self.add(ConstantFilter(**kwargs))

    # -- execution ---------------------------------------------------------------------

    def _make_context(self, X: pd.DataFrame, context: Optional[FitContext]) -> FitContext:
        if context is not None:
            return context
        return FitContext(config=self.config, target=self.target)

    def _split_target(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Separate the target from the features.

        A target left among the features is the simplest possible leak: the model reads
        the answer straight off an input column.  It is removed here, once, rather than
        relied upon to be absent.
        """
        if self.target is None or self.target not in X.columns:
            return X, y
        extracted = X[self.target]
        return X.drop(columns=[self.target]), y if y is not None else extracted

    def _select_columns(self, X: pd.DataFrame, context: FitContext) -> List[str]:
        return [str(c) for c in X.columns]

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext) -> None:
        self._fit_transform(X, y, context)

    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext
    ) -> pd.DataFrame:
        features, target = self._split_target(X, y)
        current = features
        for _, transformer in self.steps:
            current = transformer.fit_transform(current, target, context)
            transformer._is_fitted = True
        self.n_features_out_ = current.shape[1]
        return current

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        context: Optional[FitContext] = None,
    ) -> "Pipeline":
        context = self._make_context(X, context)
        self._context_ = context
        super().fit(X, y, context)
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        context: Optional[FitContext] = None,
    ) -> pd.DataFrame:
        context = self._make_context(X, context)
        self._context_ = context
        return super().fit_transform(X, y, context)

    def transform(
        self, X: pd.DataFrame, context: Optional[FitContext] = None
    ) -> pd.DataFrame:
        check_is_fitted(self)
        context = context or getattr(self, "_context_", None) or FitContext(config=self.config)
        # Only the most recent transform is described in the report; otherwise calling
        # transform on train, valid and test would produce a journal describing an
        # arbitrary concatenation of the three.
        context.journal.clear_transform_entries()
        features, _ = self._split_target(X, None)
        current = features
        for _, transformer in self.steps:
            current = transformer.transform(current, context)
        return current

    def _transform(self, X: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        current = X
        for _, transformer in self.steps:
            current = transformer.transform(current, context)
        return current

    # -- introspection --------------------------------------------------------------------

    def _compute_feature_names_out(self) -> List[str]:
        if not self.steps:
            return list(self.feature_names_in_)
        last = self.steps[-1][1]
        try:
            return list(last.get_feature_names_out())
        except (NotFittedError, SchemaError):  # pragma: no cover - defensive
            return list(self.feature_names_in_)

    @property
    def named_steps(self) -> Dict[str, Transformer]:
        return dict(self.steps)

    def __getitem__(self, key: Union[int, str]) -> Transformer:
        if isinstance(key, int):
            return self.steps[key][1]
        return self.named_steps[key]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "steps": self.steps,
            "config": self.config,
            "target": self.target,
        }
        if deep:
            for name, transformer in self.steps:
                params[name] = transformer
                for key, value in transformer.get_params(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "Pipeline":
        for key in ("steps", "config", "target"):
            if key in params:
                setattr(self, key, params.pop(key))
        nested: Dict[str, Dict[str, Any]] = {}
        for key, value in params.items():
            head, _, tail = key.partition("__")
            if head not in self.named_steps:
                raise ValueError(
                    f"Invalid parameter {head!r} for Pipeline. Valid step names are: "
                    f"{', '.join(self.named_steps) or '(none)'}."
                )
            if tail:
                nested.setdefault(head, {})[tail] = value
            else:
                index = [i for i, (n, _) in enumerate(self.steps) if n == head][0]
                self.steps[index] = (head, value)
        for head, sub in nested.items():
            self.named_steps[head].set_params(**sub)
        return self

    @property
    def journal(self) -> Journal:
        context = getattr(self, "_context_", None)
        return context.journal if context is not None else Journal()

    def report(self) -> Report:
        """The record of what actually happened."""
        check_is_fitted(self)
        return Report.from_run(
            journal=self.journal,
            profile=getattr(self, "profile_", None),
            plan=getattr(self, "plan_", None),
            config=self.config,
            feature_names_in=self.feature_names_in_,
            feature_names_out=self.feature_names_out_,
        )

    def __repr__(self) -> str:
        if not self.steps:
            return "Pipeline([])"
        rendered = ",\n  ".join(f"({name!r}, {t!r})" for name, t in self.steps)
        return f"Pipeline([\n  {rendered}\n])"


class AutoPipeline(Pipeline):
    """Profile, plan, explain, execute.

    ::

        pipe = edaprep.AutoPipeline(target="churn", model_family="tree", random_state=42)
        pipe.fit(train_df)
        pipe.explain()                       # why each column was treated as it was
        X_train = pipe.transform(train_df)
        X_test  = pipe.transform(test_df)
        print(pipe.report().summary())

    Parameters
    ----------
    target :
        Name of the target column.  Excluded from the features, used for target
        association and target encoding, and never returned by ``transform``.
    model_family :
        ``"tree"``, ``"linear"``, ``"distance"``, ``"neural"``, or ``None``.  Drives
        scaling, encoding and transform decisions.  ``None`` selects a conservative
        branch that makes no modelling assumptions.
    config :
        Full configuration.  ``model_family`` and ``random_state`` given directly here
        are written into it.
    planner :
        A customised :class:`~edaprep.planning.planner.Planner`, to change the rules.

    Attributes set after ``fit``
    ----------------------------
    ``profile_``  the measurements
    ``plan_``     the decisions, inspectable and serialisable
    ``report_``   what actually happened, with counts
    """

    def __init__(
        self,
        target: Optional[str] = None,
        model_family: Optional[Union[ModelFamily, str]] = None,
        config: Optional[Config] = None,
        random_state: Optional[int] = None,
        planner: Optional[Planner] = None,
        sample_size: Optional[int] = None,
    ) -> None:
        config = (config or Config()).copy()
        if model_family is not None:
            config.model_family = ModelFamily.coerce(model_family)
        if random_state is not None:
            config.random_state = random_state
        if sample_size is not None:
            config.sample_size = sample_size
        config.validate()

        super().__init__(steps=None, config=config, target=target)
        self.model_family = config.model_family
        self.random_state = config.random_state
        self.sample_size = config.sample_size
        self.planner = planner or Planner(config=config)
        self.profile_: Optional[DatasetProfile] = None
        self.plan_: Optional[Plan] = None
        self.report_: Optional[Report] = None

    # -- execution ---------------------------------------------------------------------

    def _prepare_plan(self, X: pd.DataFrame, context: FitContext) -> None:
        """Profile the training frame, plan from the profile, materialise the steps."""
        if self.target is not None and self.target not in X.columns:
            available = ", ".join(repr(c) for c in list(X.columns)[:10])
            raise KeyError(
                f"target={self.target!r} is not a column of the frame passed to fit(). "
                f"Available columns include: {available}. If the target is held "
                f"separately, pass it as y instead of naming it."
            )

        self.profile_ = profile_dataset(X, target=self.target, config=self.config)
        context.profile = self.profile_
        self.plan_ = self.planner.plan(self.profile_, target=self.target)
        self.steps = []
        for step in self.plan_:
            self.add(_materialise(step, self.config))

    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], context: FitContext
    ) -> pd.DataFrame:
        self._prepare_plan(X, context)
        return super()._fit_transform(X, y, context)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        context: Optional[FitContext] = None,
    ) -> "AutoPipeline":
        super().fit(X, y, context)
        self.report_ = self.report()
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        context: Optional[FitContext] = None,
    ) -> pd.DataFrame:
        out = super().fit_transform(X, y, context)
        self.report_ = self.report()
        return out

    def transform(
        self, X: pd.DataFrame, context: Optional[FitContext] = None
    ) -> pd.DataFrame:
        out = super().transform(X, context)
        self.report_ = self.report()
        return out

    # -- inspection -----------------------------------------------------------------------

    def explain(self, column: Optional[str] = None) -> str:
        """Per-column reasoning for every decision.  Printed, not just returned."""
        if self.plan_ is None:
            raise NotFittedError(
                "explain() needs a plan. Call fit() first, or call plan(df) to build "
                "a plan without fitting anything."
            )
        text = self.plan_.explain(column)
        print(text)
        return text

    def plan(self, X: pd.DataFrame) -> Plan:
        """Profile and plan without fitting.

        Lets a user inspect and edit the decisions before any data is touched, which is
        the point of keeping planning and execution apart.
        """
        self.profile_ = profile_dataset(X, target=self.target, config=self.config)
        self.plan_ = self.planner.plan(self.profile_, target=self.target)
        return self.plan_

    @property
    def statistics_(self) -> Dict[str, Any]:
        """The learned parameters, per transformer."""
        check_is_fitted(self)
        out: Dict[str, Any] = {}
        for name, transformer in self.steps:
            learned = {
                key: value
                for key, value in vars(transformer).items()
                if key.endswith("_") and not key.startswith("_")
            }
            if learned:
                out[name] = learned
        return out

    @property
    def transformations_(self) -> pd.DataFrame:
        """One row per decision: column, stage, action, rationale, source."""
        if self.plan_ is None:
            raise NotFittedError.for_object(self)
        return pd.DataFrame(
            [
                {
                    "column": d.column,
                    "stage": str(d.stage),
                    "action": d.action,
                    "rationale": d.rationale,
                    "rule": d.rule,
                    "source": d.source,
                }
                for d in sorted(
                    self.plan_.decisions, key=lambda d: (d.column, d.stage.order)
                )
            ]
        )

    def __repr__(self) -> str:
        if self.plan_ is None:
            return (
                f"AutoPipeline(target={self.target!r}, "
                f"model_family={self.model_family}, not fitted)"
            )
        return (
            f"AutoPipeline(target={self.target!r}, model_family={self.model_family}, "
            f"{len(self.steps)} steps, {len(self.plan_.dropped_columns)} dropped)"
        )


def _materialise(step, config: Config) -> Transformer:
    """Build the transformer a :class:`PlannedStep` describes.

    The plan names a transformer and supplies parameters; nothing else in the library
    needs to know how a stage is implemented, which is what keeps the plan portable.
    """
    from ..preprocessing.casting import DataTypeInference
    from ..preprocessing.datetime_features import DateTimeExpander
    from ..preprocessing.duplicates import DuplicateRowHandler
    from ..preprocessing.encoding import CategoricalEncoder, RareCategoryGrouper
    from ..preprocessing.missing import MissingIndicator, MissingValueHandler
    from ..preprocessing.outliers import OutlierHandler
    from ..preprocessing.scaling import Scaler
    from ..preprocessing.selection import ColumnDropper, CorrelationFilter
    from ..preprocessing.transformations import DistributionTransformer

    name = step.transformer
    columns = list(step.columns)
    params = dict(step.params)

    if name == "DataTypeInference":
        return DataTypeInference(
            columns or None,
            replace_sentinels=params.get("replace_sentinels", True),
            downcast_integers=params.get("downcast_integers", False),
            downcast_floats=params.get("downcast_floats", False),
        )
    if name == "ColumnDropper":
        return ColumnDropper(columns, reason=params.get("reason", "planner"))
    if name == "DuplicateRowHandler":
        return DuplicateRowHandler(strategy=params.get("strategy", "report"))
    if name == "DateTimeExpander":
        return DateTimeExpander(columns or None)
    if name == "MissingIndicator":
        return MissingIndicator(columns or None)
    if name == "OutlierHandler":
        return OutlierHandler(
            columns or None,
            per_column=params.get("per_column_strategy"),
            method="auto",
        )
    if name == "MissingValueHandler":
        return MissingValueHandler(columns or None, per_column=params.get("per_column"))
    if name == "DistributionTransformer":
        return DistributionTransformer(columns or None, per_column=params.get("per_column"))
    if name == "RareCategoryGrouper":
        return RareCategoryGrouper(columns or None, threshold=params.get("threshold"))
    if name == "CategoricalEncoder":
        return CategoricalEncoder(columns or None, per_column=params.get("per_column"))
    if name == "Scaler":
        return Scaler(columns or None, per_column=params.get("per_column"))
    if name == "CorrelationFilter":
        return CorrelationFilter(
            None,
            threshold=params.get("threshold", 0.95),
            method=params.get("method", "spearman"),
        )
    raise ConfigurationError(
        f"The plan names a transformer this version cannot build: {name!r}. The plan "
        f"may have been produced by a newer edaprep."
    )


def _snake(name: str) -> str:
    out = []
    for i, char in enumerate(name):
        if char.isupper() and i:
            out.append("_")
        out.append(char.lower())
    return "".join(out)
