"""The planner: ``(DatasetProfile, Config) -> Plan``.

It never sees a DataFrame.  Three consequences follow, and they are the reason the
architecture is shaped this way:

* it is trivially unit-testable -- build a synthetic profile, assert on the plan;
* it cannot leak, because it has nothing to leak *from*: the profile it reads was
  computed on the training frame alone;
* planning a 50 GB dataset costs the same as planning a 50-row one.

The planner's job is narrow: run the rules per column per stage, group the resulting
decisions into steps, and order the steps.  All the domain knowledge lives in
``rules.py``; all the execution lives in ``core/pipeline.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..profiling.profiler import DatasetProfile
from ..types import SemanticType, Severity, Stage
from .decisions import Decision, Plan, PlannedStep
from .rules import RuleContext, RuleSet, default_rules

__all__ = ["Planner"]

#: Which transformer implements which action, per stage.  Keeping this table separate
#: from the rules means a rule says *what* should happen and this says *how*, so a new
#: backend can reimplement the how without touching the rules.
_TRANSFORMER_FOR_ACTION = {
    Stage.DATETIME: {"expand_datetime": "DateTimeExpander"},
    Stage.MISSING_FLAG: {"add_missing_indicator": "MissingIndicator"},
    Stage.OUTLIERS: {"*": "OutlierHandler"},
    Stage.MISSING: {"*": "MissingValueHandler"},
    Stage.TRANSFORM: {"*": "DistributionTransformer"},
    Stage.RARE_CATEGORY: {"group_rare_categories": "RareCategoryGrouper"},
    Stage.ENCODE: {"*": "CategoricalEncoder"},
    Stage.SCALE: {"*": "Scaler"},
}

#: Stages executed in this order.  See docs/architecture.md section 5.2 for why
#: MISSING_FLAG precedes OUTLIERS and MISSING, and why OUTLIERS precedes MISSING.
_STAGE_ORDER: Tuple[Stage, ...] = (
    Stage.CAST,
    Stage.DROP_COLUMNS,
    Stage.DEDUPLICATE,
    Stage.MISSING_FLAG,
    Stage.DATETIME,
    Stage.OUTLIERS,
    Stage.MISSING,
    Stage.TRANSFORM,
    Stage.RARE_CATEGORY,
    Stage.ENCODE,
    Stage.SCALE,
    Stage.SELECT,
)


class Planner:
    """Turns measurements into an explainable plan.

    Parameters
    ----------
    config :
        Thresholds, global strategies and per-column overrides.
    rules :
        The rule set.  Defaults to :func:`~edaprep.planning.rules.default_rules`;
        pass a customised :class:`~edaprep.planning.rules.RuleSet` to change or extend
        the decision logic without subclassing anything.
    """

    def __init__(
        self, config: Optional[Config] = None, rules: Optional[RuleSet] = None
    ) -> None:
        self.config = config or Config()
        self.rules = rules or default_rules()

    # -- public API ----------------------------------------------------------------

    def plan(self, profile: DatasetProfile, target: Optional[str] = None) -> Plan:
        """Build a :class:`~edaprep.planning.decisions.Plan` from a profile."""
        target = target if target is not None else profile.target
        context = RuleContext(profile=profile, config=self.config, target=target)

        notes: List[str] = list(self._planning_notes(profile, context))
        decisions_by_stage: Dict[Stage, List[Decision]] = {}
        dropped: Dict[str, str] = {}

        feature_columns = [c for c in profile.column_order if c != target]

        # DROP_COLUMNS first: a dropped column takes no further part in planning, which
        # keeps the plan free of steps that operate on columns that will not exist.
        for name in feature_columns:
            decision = self.rules.decide(
                Stage.DROP_COLUMNS, profile.columns[name], context
            )
            if decision is not None and decision.action == "drop":
                dropped[name] = decision.rationale
                decisions_by_stage.setdefault(Stage.DROP_COLUMNS, []).append(decision)

        remaining = [c for c in feature_columns if c not in dropped]

        for stage in _STAGE_ORDER:
            if stage in (Stage.CAST, Stage.DROP_COLUMNS, Stage.DEDUPLICATE, Stage.SELECT):
                continue
            for name in remaining:
                decision = self.rules.decide(stage, profile.columns[name], context)
                if decision is not None:
                    decisions_by_stage.setdefault(stage, []).append(decision)

        steps = self._build_steps(decisions_by_stage, profile, context, dropped)

        emitted = {id(d) for step in steps for d in step.decisions}
        noops = [
            d
            for stage_decisions in decisions_by_stage.values()
            for d in stage_decisions
            if id(d) not in emitted
        ]

        return Plan(
            steps=tuple(steps),
            noop_decisions=tuple(noops),
            target=target,
            model_family=str(self.config.model_family) if self.config.model_family else None,
            dropped_columns=dropped,
            notes=tuple(notes),
        )

    # -- step construction ------------------------------------------------------------

    def _build_steps(
        self,
        decisions_by_stage: Dict[Stage, List[Decision]],
        profile: DatasetProfile,
        context: RuleContext,
        dropped: Dict[str, str],
    ) -> List[PlannedStep]:
        steps: List[PlannedStep] = []
        config = self.config

        # --- CAST -------------------------------------------------------------------
        cast_columns = [
            name
            for name in profile.column_order
            if name not in dropped
            and (
                profile.columns[name].suggested_dtype is not None
                or name in profile.sentinels
                or name in profile.whitespace_columns
            )
        ]
        if cast_columns or config.downcast_numeric:
            steps.append(
                PlannedStep(
                    stage=Stage.CAST,
                    transformer="DataTypeInference",
                    columns=tuple(cast_columns),
                    params={
                        "replace_sentinels": config.replace_sentinels,
                        "downcast_integers": config.downcast_numeric,
                        "downcast_floats": config.downcast_numeric,
                    },
                    decisions=tuple(
                        Decision(
                            column=name,
                            stage=Stage.CAST,
                            action="cast",
                            params={
                                "to": profile.columns[name].suggested_dtype,
                                "sentinels": name in profile.sentinels,
                            },
                            rationale=self._cast_rationale(profile, name),
                            rule="cast_from_profile",
                        )
                        for name in cast_columns
                    ),
                )
            )

        # --- DROP_COLUMNS ------------------------------------------------------------
        if dropped:
            steps.append(
                PlannedStep(
                    stage=Stage.DROP_COLUMNS,
                    transformer="ColumnDropper",
                    columns=tuple(dropped),
                    params={"reason": "planner"},
                    decisions=tuple(decisions_by_stage.get(Stage.DROP_COLUMNS, ())),
                )
            )

        # --- DEDUPLICATE --------------------------------------------------------------
        if config.duplicate_strategy != "ignore" and profile.n_duplicate_rows:
            steps.append(
                PlannedStep(
                    stage=Stage.DEDUPLICATE,
                    transformer="DuplicateRowHandler",
                    columns=(),
                    params={"strategy": config.duplicate_strategy},
                    decisions=(),
                )
            )

        # --- per-column stages ---------------------------------------------------------
        for stage in (
            Stage.MISSING_FLAG,
            Stage.DATETIME,
            Stage.OUTLIERS,
            Stage.MISSING,
            Stage.TRANSFORM,
            Stage.RARE_CATEGORY,
            Stage.ENCODE,
            Stage.SCALE,
        ):
            decisions = [d for d in decisions_by_stage.get(stage, ()) if not d.is_noop]
            if not decisions:
                continue
            step = self._step_for(stage, decisions)
            if step is not None:
                steps.append(step)

        # --- SELECT ----------------------------------------------------------------------
        if config.correlation_filter:
            steps.append(
                PlannedStep(
                    stage=Stage.SELECT,
                    transformer="CorrelationFilter",
                    columns=(),
                    params={
                        "threshold": config.thresholds.correlation_threshold,
                        "method": "spearman",
                    },
                    decisions=(),
                )
            )

        return steps

    def _step_for(self, stage: Stage, decisions: Sequence[Decision]) -> Optional[PlannedStep]:
        """Group one stage's decisions into a single transformer step."""
        table = _TRANSFORMER_FOR_ACTION.get(stage, {})
        transformer = table.get("*")
        if transformer is None:
            transformer = table.get(decisions[0].action)
        if transformer is None:
            return None

        columns = tuple(d.column for d in decisions)
        params: Dict[str, Any] = {}

        # MISSING and SCALE run after stages that add columns -- calendar features from
        # the datetime expander, indicators from one-hot encoding.  Pinning them to the
        # columns known at planning time would leave those new columns unhandled (a
        # NaN calendar feature surviving into the output), so these two stages resolve
        # their scope at fit time and use the planned decisions as per-column overrides.
        if stage in (Stage.MISSING, Stage.SCALE):
            columns = ()

        if stage is Stage.OUTLIERS:
            params["per_column_method"] = {
                d.column: d.params.get("method") for d in decisions
            }
            params["per_column_strategy"] = {
                d.column: d.params.get("strategy") for d in decisions
            }
        elif stage is Stage.MISSING:
            params["per_column"] = {d.column: d.params["strategy"] for d in decisions}
        elif stage is Stage.TRANSFORM:
            params["per_column"] = {d.column: d.params["method"] for d in decisions}
        elif stage is Stage.ENCODE:
            params["per_column"] = {d.column: d.params["encoding"] for d in decisions}
        elif stage is Stage.SCALE:
            params["per_column"] = {d.column: d.params["scaling"] for d in decisions}
        elif stage is Stage.RARE_CATEGORY:
            params["threshold"] = decisions[0].params.get("threshold")
        elif stage is Stage.DATETIME:
            explicit = {
                d.column: d.params["features"]
                for d in decisions
                if d.params.get("features")
            }
            if explicit:
                params["per_column_features"] = explicit

        return PlannedStep(
            stage=stage,
            transformer=transformer,
            columns=columns,
            params=params,
            decisions=tuple(decisions),
        )

    # -- notes ---------------------------------------------------------------------------

    def _cast_rationale(self, profile: DatasetProfile, name: str) -> str:
        parts = []
        cp = profile.columns[name]
        if name in profile.sentinels:
            found = ", ".join(repr(s) for s in profile.sentinels[name])
            parts.append(f"placeholder values ({found}) converted to missing")
        if name in profile.whitespace_columns:
            parts.append("surrounding whitespace stripped")
        if cp.suggested_dtype:
            parts.append(f"stored as text but parses as {cp.suggested_dtype}")
        return "; ".join(parts) or "dtype correction"

    def _planning_notes(self, profile: DatasetProfile, context: RuleContext) -> List[str]:
        """Advisories that belong to the plan as a whole."""
        notes: List[str] = []

        if context.config.model_family is None:
            notes.append(
                "No model_family was declared, so conservative defaults were used "
                "(standard scaling, one-hot encoding). Declaring one -- 'tree', "
                "'linear', 'distance' or 'neural' -- lets the planner skip work that "
                "your model does not need."
            )

        uncertain = profile.uncertain_columns
        if uncertain:
            notes.append(
                f"{len(uncertain)} column(s) have an uncertain semantic type and were "
                f"planned on a best guess: {', '.join(repr(c) for c in uncertain[:6])}"
                + (f" (+{len(uncertain) - 6} more)" if len(uncertain) > 6 else "")
                + ". Set config.column(name).semantic_type to correct any of them."
            )

        leakage = [i for i in profile.issues if i.code == "possible_target_leakage"]
        for issue in leakage:
            notes.append(
                f"LEAKAGE SUSPECTED: {', '.join(repr(c) for c in issue.columns)} "
                f"are almost perfectly associated with the target. edaprep does not "
                f"drop them automatically, because a legitimately strong feature looks "
                f"identical from here. Investigate before modelling."
            )

        imbalance = [i for i in profile.issues if i.code == "class_imbalance"]
        for issue in imbalance:
            notes.append(
                f"{issue.message} No resampling step was planned: resampling is a "
                f"modelling decision and belongs after the train/test split."
            )

        if profile.n_duplicate_rows and context.config.duplicate_strategy == "report":
            notes.append(
                f"{profile.n_duplicate_rows:,} duplicate rows were found but kept. "
                f"Repeated observations are legitimate in transactional data; set "
                f"Config(duplicate_strategy='remove') if they are errors here."
            )

        numeric_sentinels = [i for i in profile.issues if i.code == "numeric_sentinel_values"]
        for issue in numeric_sentinels:
            notes.append(
                f"{issue.message} No automatic replacement was planned; set "
                f"config.column(name).imputation after converting them yourself if "
                f"they are indeed placeholders."
            )

        return notes

    def __repr__(self) -> str:
        return f"Planner(rules={len(self.rules)}, config={self.config!r})"
