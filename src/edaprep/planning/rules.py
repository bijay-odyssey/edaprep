"""The rule set.

Rules are objects, not ``if`` statements buried in a planner method.  Each declares the
stage it belongs to, a predicate saying when it applies, and a function producing a
:class:`~edaprep.planning.decisions.Decision` with a *rationale in English*.

Within a stage, rules are evaluated in descending priority and the first one that
produces a decision for a column wins.  That is a documented, testable
conflict-resolution policy rather than an emergent property of statement order.

Provenance
----------
The three routing axes below are mined from notebook practice (see
``docs/design-rationale.md``, section 5), where they exist as hand-written branches
inside a ``make_preprocessor`` closure.  Formalising them means the thresholds are
named, the reasoning is printed, and a user can override any of it per column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import AUTO, Config
from ..preprocessing.encoding import resolve_encoding
from ..preprocessing.scaling import resolve_scaling
from ..preprocessing.transformations import choose_transform
from ..profiling.profiler import ColumnProfile, DatasetProfile
from ..types import ModelFamily, SemanticType, Stage
from .decisions import DEFAULT, RULE, USER_OVERRIDE, Decision

__all__ = ["Rule", "RuleSet", "default_rules", "RuleContext"]


@dataclass
class RuleContext:
    """What a rule may look at.  Never a DataFrame."""

    profile: DatasetProfile
    config: Config
    target: Optional[str] = None

    @property
    def model_family(self) -> Optional[ModelFamily]:
        return self.config.model_family

    @property
    def thresholds(self):
        return self.config.thresholds


@dataclass(frozen=True)
class Rule:
    """A named, testable decision rule."""

    name: str
    stage: Stage
    decide: Callable[[ColumnProfile, RuleContext], Optional[Decision]]
    priority: int = 0
    description: str = ""

    def __call__(
        self, column: ColumnProfile, context: RuleContext
    ) -> Optional[Decision]:
        return self.decide(column, context)

    def __repr__(self) -> str:
        return f"Rule({self.name!r}, stage={self.stage}, priority={self.priority})"


class RuleSet:
    """Rules grouped by stage, evaluated in priority order.

    Extension point: ``ruleset.register(Rule(...))`` adds a rule.  Giving it a higher
    priority than the built-ins lets a user pre-empt them without editing the library.
    """

    def __init__(self, rules: Optional[Sequence[Rule]] = None) -> None:
        self._by_stage: Dict[Stage, List[Rule]] = {}
        for rule in rules or ():
            self.register(rule)

    def register(self, rule: Rule) -> "RuleSet":
        bucket = self._by_stage.setdefault(rule.stage, [])
        bucket.append(rule)
        bucket.sort(key=lambda r: (-r.priority, r.name))
        return self

    def for_stage(self, stage: Stage) -> List[Rule]:
        return list(self._by_stage.get(stage, ()))

    @property
    def stages(self) -> List[Stage]:
        return sorted(self._by_stage, key=lambda s: s.order)

    def decide(
        self, stage: Stage, column: ColumnProfile, context: RuleContext
    ) -> Optional[Decision]:
        """First rule in this stage that produces a decision wins."""
        for rule in self.for_stage(stage):
            decision = rule(column, context)
            if decision is not None:
                return decision
        return None

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_stage.values())

    def __repr__(self) -> str:
        return f"RuleSet({len(self)} rules across {len(self._by_stage)} stages)"


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------


def _override(column: str, context: RuleContext, attribute: str):
    """The user's value for one per-column setting, or None."""
    config = context.config.get_column(column)
    if config is None:
        return None
    return getattr(config, attribute, None)


def _pct(value: float) -> str:
    return f"{value:.1%}"


# ---------------------------------------------------------------------------------
# DROP_COLUMNS
# ---------------------------------------------------------------------------------


def _rule_user_drop(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if _override(cp.name, ctx, "drop") is not True:
        return None
    return Decision(
        cp.name,
        Stage.DROP_COLUMNS,
        "drop",
        rationale="marked for removal in the configuration",
        rule="user_drop",
        source=USER_OVERRIDE,
    )


def _rule_drop_constant(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if not ctx.config.drop_constants or cp.is_target or not cp.is_constant:
        return None
    return Decision(
        cp.name,
        Stage.DROP_COLUMNS,
        "drop",
        params={"n_unique": cp.n_unique},
        rationale=(
            "entirely missing" if cp.missing_fraction == 1.0
            else f"constant ({cp.n_unique} distinct value)"
        ),
        rule="drop_constant",
    )


def _rule_drop_identifier(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if not ctx.config.drop_identifiers or cp.is_target:
        return None
    if cp.semantic is not SemanticType.IDENTIFIER:
        return None
    notes = ()
    if cp.semantic_confidence < 0.75:
        notes = (
            f"identifier inference is only {cp.semantic_confidence:.0%} confident; set "
            f"config.column({cp.name!r}).semantic_type if this column is a real "
            f"feature",
        )
    return Decision(
        cp.name,
        Stage.DROP_COLUMNS,
        "drop",
        params={"unique_ratio": round(cp.unique_ratio, 4)},
        rationale=(
            f"identifier: {_pct(cp.unique_ratio)} of values are distinct, so it cannot "
            f"generalise beyond the rows it was fitted on"
        ),
        rule="drop_identifier",
        confidence=cp.semantic_confidence,
        notes=notes,
    )


def _rule_drop_high_missing(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if not ctx.config.drop_high_missing or cp.is_target:
        return None
    threshold = ctx.thresholds.missing_drop_threshold
    if cp.missing_fraction < threshold:
        return None
    return Decision(
        cp.name,
        Stage.DROP_COLUMNS,
        "drop",
        params={"missing_fraction": round(cp.missing_fraction, 4)},
        rationale=(
            f"{_pct(cp.missing_fraction)} missing, above the {_pct(threshold)} ceiling; "
            f"imputing would invent most of the column"
        ),
        rule="drop_high_missing",
    )


def _rule_drop_text(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target or cp.semantic is not SemanticType.TEXT:
        return None
    return Decision(
        cp.name,
        Stage.DROP_COLUMNS,
        "drop",
        rationale=(
            "free text; edaprep does not vectorise text in this version. Set "
            "config.column(name).semantic_type='categorical' to encode it instead"
        ),
        rule="drop_text",
        confidence=cp.semantic_confidence,
    )


# ---------------------------------------------------------------------------------
# MISSING_FLAG
# ---------------------------------------------------------------------------------


def _rule_missing_indicator(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target or not ctx.config.add_missing_indicators:
        return None
    threshold = ctx.thresholds.missing_indicator_threshold
    if cp.missing_fraction < threshold or cp.missing_fraction >= 1.0:
        return None
    return Decision(
        cp.name,
        Stage.MISSING_FLAG,
        "add_missing_indicator",
        params={"missing_fraction": round(cp.missing_fraction, 4)},
        rationale=(
            f"{_pct(cp.missing_fraction)} missing, above the {_pct(threshold)} flag "
            f"threshold; the flag is added before imputation, which would otherwise "
            f"destroy the signal"
        ),
        rule="missing_indicator",
    )


# ---------------------------------------------------------------------------------
# OUTLIERS  (notebook practice axis 3: choose the method by skewness)
# ---------------------------------------------------------------------------------


def _rule_outliers(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target or cp.semantic is not SemanticType.NUMERIC or cp.numeric is None:
        return None
    # Two-valued numeric columns have a degenerate fence; see OutlierHandler.
    if cp.n_unique <= 2:
        return None

    user_method = _override(cp.name, ctx, "outlier_method")
    user_strategy = _override(cp.name, ctx, "outlier_strategy")

    skew = cp.skew
    thresholds = ctx.thresholds
    if user_method not in (None, AUTO):
        method, reason = user_method, "method set in the configuration"
    elif not np.isfinite(skew):
        method, reason = "iqr", "skewness undefined; falling back to the IQR fence"
    elif abs(skew) >= thresholds.skew_heavy:
        method = "modified_zscore"
        reason = (
            f"skew {skew:.2f} is heavy (>= {thresholds.skew_heavy}); the MAD-based "
            f"score is used because the mean and standard deviation are themselves "
            f"distorted by the values being detected"
        )
    elif abs(skew) >= thresholds.skew_moderate:
        method = "iqr"
        reason = (
            f"skew {skew:.2f} is moderate (>= {thresholds.skew_moderate}); IQR fence "
            f"widened to k={thresholds.iqr_k_skewed} for the asymmetry"
        )
    else:
        method = "zscore"
        reason = (
            f"skew {skew:.2f} is small (< {thresholds.skew_moderate}), so the "
            f"distribution is near-symmetric and the z-score fence applies"
        )

    strategy = user_strategy or ctx.config.outlier_strategy
    source = USER_OVERRIDE if user_strategy or user_method else RULE
    if strategy == AUTO:
        # Conservative by design: an unusual value is not evidence of an error, and the
        # brief is explicit that detection and removal are different decisions.
        strategy = "clip" if ctx.model_family in (
            ModelFamily.LINEAR,
            ModelFamily.DISTANCE,
            ModelFamily.NEURAL,
        ) else "report"

    return Decision(
        cp.name,
        Stage.OUTLIERS,
        f"outliers_{strategy}",
        params={"method": method, "strategy": strategy},
        rationale=reason,
        rule="outlier_method_by_skew",
        source=source,
    )


# ---------------------------------------------------------------------------------
# MISSING
# ---------------------------------------------------------------------------------


def _rule_impute(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target:
        return None
    user = _override(cp.name, ctx, "imputation")
    # A column with no missing values still gets a decision when the user asked for
    # one, and when outlier handling upstream may introduce NaN.
    outlier_may_impute = (
        cp.semantic is SemanticType.NUMERIC and ctx.config.outlier_strategy == "impute"
    )
    if cp.n_missing == 0 and user is None and not outlier_may_impute:
        return None

    if user is not None:
        return Decision(
            cp.name,
            Stage.MISSING,
            f"impute_{user}",
            params={"strategy": user},
            rationale="imputation strategy set in the configuration",
            rule="user_imputation",
            source=USER_OVERRIDE,
        )

    global_strategy = ctx.config.missing_strategy
    if global_strategy != AUTO:
        return Decision(
            cp.name,
            Stage.MISSING,
            f"impute_{global_strategy}",
            params={"strategy": global_strategy},
            rationale=f"global missing_strategy={global_strategy!r}",
            rule="global_imputation",
            source=DEFAULT,
        )

    if cp.semantic in (SemanticType.NUMERIC, SemanticType.ORDINAL):
        strategy = "median"
        reason = (
            f"{_pct(cp.missing_fraction)} missing; median rather than mean because it "
            f"is unaffected by the skew ({cp.skew:.2f}) and by outliers"
            if np.isfinite(cp.skew)
            else f"{_pct(cp.missing_fraction)} missing; median is robust to outliers"
        )
    elif cp.semantic is SemanticType.DATETIME:
        strategy, reason = "median", f"{_pct(cp.missing_fraction)} missing; median date"
    elif cp.n_unique > ctx.config.effective_high_cardinality:
        strategy = "missing_category"
        reason = (
            f"{cp.n_unique} distinct levels: the mode covers only "
            f"{_pct(cp.modal_frequency)} of the column, so filling with it would invent "
            f"a concentration that is not there"
        )
    else:
        strategy = "mode"
        reason = (
            f"{_pct(cp.missing_fraction)} missing; most frequent category "
            f"({cp.modal_value!r}, {_pct(cp.modal_frequency)} of rows)"
        )

    notes = ()
    comissing = [
        other
        for a, b, _ in ctx.profile.comissing_pairs
        for other in ((b,) if a == cp.name else (a,) if b == cp.name else ())
    ]
    if comissing:
        notes = (
            f"missingness is correlated with {', '.join(sorted(set(comissing))[:3])}, "
            f"which suggests a shared cause rather than independent gaps",
        )

    return Decision(
        cp.name,
        Stage.MISSING,
        f"impute_{strategy}",
        params={"strategy": strategy, "missing_fraction": round(cp.missing_fraction, 4)},
        rationale=reason,
        rule="impute_by_type",
        notes=notes,
    )


# ---------------------------------------------------------------------------------
# TRANSFORM  (notebook practice axis 1: route numeric columns by distribution shape)
# ---------------------------------------------------------------------------------


def _rule_transform(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target or cp.semantic is not SemanticType.NUMERIC or cp.numeric is None:
        return None

    user = _override(cp.name, ctx, "transform")
    if user is not None:
        return Decision(
            cp.name,
            Stage.TRANSFORM,
            f"transform_{user}" if user != "none" else "no_transform",
            params={"method": user},
            rationale="transform set in the configuration",
            rule="user_transform",
            source=USER_OVERRIDE,
        )

    if ctx.config.transform_strategy != AUTO:
        method = ctx.config.transform_strategy
        return Decision(
            cp.name,
            Stage.TRANSFORM,
            f"transform_{method}" if method != "none" else "no_transform",
            params={"method": method},
            rationale=f"global transform_strategy={method!r}",
            rule="global_transform",
            source=DEFAULT,
        )

    # Tree ensembles are invariant to monotone transforms: a split on log(x) is a split
    # on x. Transforming for a tree costs interpretability and buys nothing.
    if ctx.model_family is ModelFamily.TREE:
        return Decision(
            cp.name,
            Stage.TRANSFORM,
            "no_transform",
            params={"method": "none"},
            rationale=(
                "tree models are invariant to monotone transforms, so correcting skew "
                "would change nothing but readability"
            ),
            rule="no_transform_for_trees",
        )

    thresholds = ctx.thresholds
    skew = cp.skew
    minimum = cp.numeric.minimum
    method = choose_transform(skew, minimum, thresholds.skew_moderate, thresholds.skew_heavy)

    if method == "none":
        return Decision(
            cp.name,
            Stage.TRANSFORM,
            "no_transform",
            params={"method": "none"},
            rationale=(
                f"skew {skew:.2f} is within +/-{thresholds.skew_moderate}, so the "
                f"distribution is close enough to symmetric"
                if np.isfinite(skew)
                else "skewness undefined"
            ),
            rule="transform_by_skew",
        )

    if method == "log1p":
        reason = (
            f"skew {skew:.2f} is moderate and the column is non-negative "
            f"(min {minimum:g}), so log1p applies and is invertible"
        )
    else:
        reason = (
            f"skew {skew:.2f} is heavy (>= {thresholds.skew_heavy}); Yeo-Johnson is "
            f"used rather than Box-Cox because it is defined for the column's range "
            f"(min {minimum:g})"
            if abs(skew) >= thresholds.skew_heavy
            else (
                f"skew {skew:.2f} is moderate but the column has negative values "
                f"(min {minimum:g}), so log is undefined and Yeo-Johnson is used"
            )
        )

    notes = ()
    if cp.has_zero and method == "log1p":
        notes = (
            f"{cp.numeric.n_zeros} zero value(s) map to 0 under log1p, which is "
            f"well defined",
        )

    return Decision(
        cp.name,
        Stage.TRANSFORM,
        f"transform_{method}",
        params={"method": method, "skew": round(float(skew), 4) if np.isfinite(skew) else None},
        rationale=reason,
        rule="transform_by_skew",
        notes=notes,
    )


# ---------------------------------------------------------------------------------
# RARE_CATEGORY / ENCODE  (notebook practice axis 2: route by consuming model family)
# ---------------------------------------------------------------------------------


def _rule_rare_category(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target or cp.semantic not in (SemanticType.CATEGORICAL, SemanticType.BINARY):
        return None
    threshold = ctx.config.effective_rare_threshold
    n_rows = cp.n_rows
    floor = max(1, int(np.ceil(threshold * n_rows)))
    # Only worth a step when there is something to group.  Approximated from the top-10
    # counts held on the profile: if the tenth most common level is already above the
    # floor there may still be rare levels below it, so cardinality is the real signal.
    if cp.n_unique <= 2:
        return None
    if cp.n_unique < 5:
        return None
    return Decision(
        cp.name,
        Stage.RARE_CATEGORY,
        "group_rare_categories",
        params={"threshold": threshold, "min_count": floor},
        rationale=(
            f"{cp.n_unique} levels; those appearing in fewer than {floor} rows "
            f"({_pct(threshold)}) are grouped, since they cannot support a reliable "
            f"estimate and would each add a near-empty column"
        ),
        rule="group_rare",
    )


def _rule_encode(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target:
        return None
    if cp.semantic not in (
        SemanticType.CATEGORICAL,
        SemanticType.BINARY,
        SemanticType.ORDINAL,
    ):
        return None

    user = _override(cp.name, ctx, "encoding")
    if user is not None:
        return Decision(
            cp.name,
            Stage.ENCODE,
            f"encode_{user}",
            params={"encoding": user},
            rationale="encoding set in the configuration",
            rule="user_encoding",
            source=USER_OVERRIDE,
        )

    if ctx.config.categorical_encoding != AUTO:
        encoding = ctx.config.categorical_encoding
        return Decision(
            cp.name,
            Stage.ENCODE,
            f"encode_{encoding}",
            params={"encoding": encoding},
            rationale=f"global categorical_encoding={encoding!r}",
            rule="global_encoding",
            source=DEFAULT,
        )

    if cp.semantic is SemanticType.ORDINAL:
        return Decision(
            cp.name,
            Stage.ENCODE,
            "encode_ordinal",
            params={"encoding": "ordinal", "cardinality": cp.n_unique},
            rationale=(
                f"ordered scale with {cp.n_unique} levels; integer codes preserve the "
                f"ordering, which one-hot encoding would discard"
            ),
            rule="encode_ordinal_scale",
            confidence=cp.semantic_confidence,
        )

    high = ctx.config.effective_high_cardinality
    extreme = ctx.thresholds.extreme_cardinality_threshold
    encoding = resolve_encoding(
        cp.n_unique, ctx.model_family, high, extreme, has_target=ctx.target is not None
    )

    family = ctx.model_family
    if cp.n_unique <= 2:
        reason = "two levels; a single 0/1 column, with no ordering to impose"
    elif family is ModelFamily.TREE:
        reason = (
            f"{cp.n_unique} levels; tree models split on thresholds and can isolate any "
            f"subset of codes, so integer codes cost nothing and avoid "
            f"{cp.n_unique} extra columns"
        )
    elif encoding == "onehot":
        reason = (
            f"{cp.n_unique} levels, below the {high}-level ceiling; one-hot avoids "
            f"imposing an ordering that is not there"
        )
    elif encoding == "target":
        folds = ctx.config.target_encoding_folds
        reason = (
            f"{cp.n_unique} levels exceeds the {high}-level one-hot ceiling; target "
            f"encoding is used with {folds}-fold cross-fitting so no row is encoded "
            f"using its own target"
        )
    else:
        reason = (
            f"{cp.n_unique} levels exceeds the {high}-level one-hot ceiling and no "
            f"target is available for target encoding; frequency encoding gives one "
            f"column and cannot leak"
        )

    notes = ()
    if encoding == "onehot" and cp.n_unique > 20:
        notes = (f"adds {cp.n_unique} columns",)

    return Decision(
        cp.name,
        Stage.ENCODE,
        f"encode_{encoding}",
        params={"encoding": encoding, "cardinality": cp.n_unique},
        rationale=reason,
        rule="encode_by_cardinality_and_family",
        confidence=cp.semantic_confidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------------
# DATETIME
# ---------------------------------------------------------------------------------


def _rule_datetime(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target or cp.semantic is not SemanticType.DATETIME:
        return None
    if not ctx.config.expand_datetime:
        return Decision(
            cp.name,
            Stage.DATETIME,
            "drop",
            rationale=(
                "datetime expansion disabled; a raw datetime cannot be consumed by a "
                "tabular model, so the column is dropped"
            ),
            rule="datetime_disabled",
            source=DEFAULT,
        )
    override = _override(cp.name, ctx, "datetime_features")
    return Decision(
        cp.name,
        Stage.DATETIME,
        "expand_datetime",
        params={"features": list(override) if override else None},
        rationale=(
            "expanded into calendar features; only those that vary in the training "
            "data are kept, and the set is frozen so test data gets identical columns"
        ),
        rule="expand_datetime",
        source=USER_OVERRIDE if override else RULE,
    )


# ---------------------------------------------------------------------------------
# SCALE
# ---------------------------------------------------------------------------------


def _rule_scale(cp: ColumnProfile, ctx: RuleContext) -> Optional[Decision]:
    if cp.is_target:
        return None
    # DATETIME is absent here on purpose: by the time SCALE runs, the raw datetime
    # column has been expanded into calendar features and dropped, so naming it would
    # produce a step referring to a column that no longer exists.  The derived features
    # are picked up by the Scaler's own fit-time selection.
    if cp.semantic not in (
        SemanticType.NUMERIC,
        SemanticType.ORDINAL,
        SemanticType.BINARY,
    ):
        return None

    user = _override(cp.name, ctx, "scaling")
    if user is not None:
        return Decision(
            cp.name,
            Stage.SCALE,
            f"scale_{user}" if user != "none" else "no_scaling",
            params={"scaling": user},
            rationale="scaling set in the configuration",
            rule="user_scaling",
            source=USER_OVERRIDE,
        )

    if ctx.config.scaling != AUTO:
        strategy = ctx.config.scaling
        return Decision(
            cp.name,
            Stage.SCALE,
            f"scale_{strategy}" if strategy != "none" else "no_scaling",
            params={"scaling": strategy},
            rationale=f"global scaling={strategy!r}",
            rule="global_scaling",
            source=DEFAULT,
        )

    family = ctx.model_family
    strategy = resolve_scaling(family)
    if strategy == "none":
        return Decision(
            cp.name,
            Stage.SCALE,
            "no_scaling",
            params={"scaling": "none"},
            rationale=(
                "tree models are invariant to monotone rescaling, so scaling would "
                "cost a pass over the data and change no split"
            ),
            rule="scale_by_family",
        )

    # Robust scaling for skewed columns: standard scaling divides by a standard
    # deviation that the tail dominates, leaving the bulk of the data compressed into a
    # narrow band.  Notebook practice reached the same conclusion by hand.
    if (
        family is not ModelFamily.NEURAL
        and cp.numeric is not None
        and np.isfinite(cp.skew)
        and abs(cp.skew) >= ctx.thresholds.skew_moderate
    ):
        return Decision(
            cp.name,
            Stage.SCALE,
            "scale_robust",
            params={"scaling": "robust", "skew": round(float(cp.skew), 4)},
            rationale=(
                f"skew {cp.skew:.2f}; robust scaling (median and IQR) rather than "
                f"standard, whose standard deviation is dominated by the tail"
            ),
            rule="scale_by_family",
        )

    if family is None:
        reason = (
            "no model family declared; standard scaling is applied as a conservative "
            "default. Pass model_family='tree' to skip scaling entirely"
        )
    else:
        reason = f"{family} models are sensitive to feature scale"

    return Decision(
        cp.name,
        Stage.SCALE,
        f"scale_{strategy}",
        params={"scaling": strategy},
        rationale=reason,
        rule="scale_by_family",
        source=RULE if family is not None else DEFAULT,
    )


# ---------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------


def default_rules() -> RuleSet:
    """The built-in rule set.

    Priorities encode precedence within a stage: user overrides first, then the
    cheapest and least reversible decisions (dropping a constant column) before the
    more nuanced ones.
    """
    return RuleSet(
        [
            # DROP_COLUMNS
            Rule("user_drop", Stage.DROP_COLUMNS, _rule_user_drop, priority=100,
                 description="Honour config.column(name).drop = True"),
            Rule("drop_constant", Stage.DROP_COLUMNS, _rule_drop_constant, priority=90,
                 description="Remove columns with a single distinct value"),
            Rule("drop_identifier", Stage.DROP_COLUMNS, _rule_drop_identifier, priority=80,
                 description="Remove row keys, which cannot generalise"),
            Rule("drop_high_missing", Stage.DROP_COLUMNS, _rule_drop_high_missing, priority=70,
                 description="Remove columns too sparse to impute honestly"),
            Rule("drop_text", Stage.DROP_COLUMNS, _rule_drop_text, priority=60,
                 description="Remove free-text columns, which v1 does not vectorise"),
            # DATETIME
            Rule("expand_datetime", Stage.DATETIME, _rule_datetime, priority=50,
                 description="Expand datetimes into varying calendar features"),
            # MISSING_FLAG
            Rule("missing_indicator", Stage.MISSING_FLAG, _rule_missing_indicator, priority=50,
                 description="Flag missingness before imputation destroys it"),
            # OUTLIERS
            Rule("outlier_method_by_skew", Stage.OUTLIERS, _rule_outliers, priority=50,
                 description="Choose the fence from measured skewness (notebook practice axis 3)"),
            # MISSING
            Rule("impute_by_type", Stage.MISSING, _rule_impute, priority=50,
                 description="Median for numeric, mode or explicit category otherwise"),
            # TRANSFORM
            Rule("transform_by_skew", Stage.TRANSFORM, _rule_transform, priority=50,
                 description="Skew tiering with support validation (notebook practice axis 1)"),
            # RARE_CATEGORY
            Rule("group_rare", Stage.RARE_CATEGORY, _rule_rare_category, priority=50,
                 description="Collapse levels too rare to estimate"),
            # ENCODE
            Rule("encode_by_cardinality_and_family", Stage.ENCODE, _rule_encode, priority=50,
                 description="Route by cardinality and model family (notebook practice axis 2)"),
            # SCALE
            Rule("scale_by_family", Stage.SCALE, _rule_scale, priority=50,
                 description="Scale only when the model family needs it"),
        ]
    )
