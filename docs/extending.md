# Extending edaprep

The library is built from three replaceable pieces: transformers that do the work, rules
that decide what work applies, and a backend protocol that says how frames are touched.
Each can be extended without editing the package.

---

## 1. A custom transformer

Subclass `Transformer` and implement two methods. The base class handles validation,
journalling, the fitted-state flag, `get_params`/`set_params` and schema checking, so a
new transformer cannot forget them.

```python
import numpy as np
import pandas as pd
from edaprep import Transformer
from edaprep.core.base import ColumnTransformerMixin
from edaprep.types import SemanticType, Stage


class WinsorizeByGroup(Transformer, ColumnTransformerMixin):
    """Clip each column to percentile bounds learned per group.

    Learned state lives in trailing-underscore attributes written only inside _fit.
    transform reads them and computes nothing, which is what makes it leakage-safe.
    """

    stage = Stage.OUTLIERS

    def __init__(self, columns=None, group_column=None, lower=0.01, upper=0.99):
        super().__init__(columns)
        self.group_column = group_column
        self.lower = lower
        self.upper = upper

    def _select_columns(self, X, context):
        """Which columns to act on when the caller did not say.

        Consulting the profile is what keeps a numeric transformer away from a text
        column, and it is why type-aware defaults are possible at all.
        """
        return [
            name
            for name in map(str, X.columns)
            if name != context.target
            and (cp := context.column_profile(name)) is not None
            and cp.semantic is SemanticType.NUMERIC
        ]

    def _fit(self, X, y, context):
        self.bounds_ = {}
        groups = X[self.group_column] if self.group_column else pd.Series("_", index=X.index)
        with context.journal.timer(self.stage, type(self).__name__, "fit", "fit") as t:
            for column in self.columns_:
                for key, values in X[column].groupby(groups, observed=True):
                    lo, hi = values.quantile([self.lower, self.upper])
                    self.bounds_[(column, key)] = (float(lo), float(hi))
            t.columns = list(self.columns_)
            t.effect = {"n_bounds": len(self.bounds_)}

    def _transform(self, X, context):
        replacements, affected = {}, {}
        groups = X[self.group_column] if self.group_column else pd.Series("_", index=X.index)
        for column in self.columns_:
            if column not in X.columns:
                continue
            out = X[column].copy()
            for key, index in groups.groupby(groups, observed=True).groups.items():
                bounds = self.bounds_.get((column, key))
                if bounds is None:
                    continue        # a group unseen at fit time: leave it alone
                out.loc[index] = out.loc[index].clip(*bounds)
            affected[column] = int((out != X[column]).sum())
            replacements[column] = out
        context.journal.record(
            self.stage, type(self).__name__, "winsorize_by_group", "transform",
            columns=list(replacements), effect={"n_clipped": affected},
        )
        return self._rebuild(X, replacements)
```

Then use it anywhere a built-in goes:

```python
pipe = edaprep.Pipeline(target="churn").add(WinsorizeByGroup(group_column="region"))
```

### 1.1 The contract

| requirement | why |
|---|---|
| learned state in `name_` attributes, written only in `_fit` | `transform` has nothing else to read, so it cannot leak |
| `_transform` computes no statistic over its input | recomputing on the incoming batch is the defect this library exists to prevent |
| `uses_target = True` if `_fit` reads `y` | audited; reaching such a transformer without `y` raises `LeakageError` |
| `cross_fitted = True` if `fit_transform` must differ from `fit().transform()` | the only legitimate reason is out-of-fold encoding |
| `stage = Stage.X` | ordering and reporting |
| never mutate the input frame | asserted for every built-in transformer by a test |

`ColumnTransformerMixin._rebuild(X, replacements, added)` assembles the output from the
input's column blocks with only the touched columns replaced, so a transformer that
changes 5 of 400 columns allocates 5 columns rather than copying the frame.

### 1.2 Testing it

The same properties the built-ins are held to:

```python
def test_no_leakage(frame):
    """Transforming whole must equal transforming row by row."""
    t = WinsorizeByGroup(group_column="region").fit(train, None, context)
    whole = t.transform(test, context)
    rows = pd.concat([t.transform(test.iloc[[i]], context) for i in range(len(test))])
    pd.testing.assert_frame_equal(whole, rows)


def test_input_not_mutated(frame):
    before = frame.copy(deep=True)
    WinsorizeByGroup().fit_transform(frame, None, context)
    pd.testing.assert_frame_equal(frame, before)
```

---

## 2. A custom planning rule

Rules are objects with a stage, a priority and a function. Within a stage they run in
descending priority, and the first that returns a `Decision` wins — so a rule with a
higher priority than the built-ins pre-empts them without editing the library.

```python
from edaprep import Planner, Config, default_rules
from edaprep.planning.rules import Rule
from edaprep.planning.decisions import Decision
from edaprep.types import SemanticType, Stage


def monetary_columns_use_log(column_profile, context):
    """House rule: anything that looks like money gets log1p, whatever its skew."""
    if column_profile.semantic is not SemanticType.NUMERIC:
        return None
    if not any(k in column_profile.name.lower() for k in ("price", "cost", "revenue")):
        return None
    if column_profile.numeric is None or column_profile.numeric.minimum < 0:
        return None                       # log1p needs x > -1; let the built-in decide
    return Decision(
        column=column_profile.name,
        stage=Stage.TRANSFORM,
        action="transform_log1p",
        params={"method": "log1p"},
        rationale="house rule: monetary columns are modelled on a log scale",
        rule="monetary_log",
    )


rules = default_rules()
rules.register(Rule("monetary_log", Stage.TRANSFORM, monetary_columns_use_log, priority=100))

config = edaprep.Config(model_family="linear", random_state=42)
pipe = edaprep.AutoPipeline(
    target="churn", config=config, planner=Planner(config=config, rules=rules)
)
```

`pipe.explain("unit_price")` now prints the house rule's rationale, and
`transformations_` shows `rule="monetary_log"`, so the decision is attributable.

### 2.1 What a rule may look at

A rule receives a `ColumnProfile` and a `RuleContext` (profile, config, target). **It
never receives a DataFrame.** That is what makes the planner unit-testable without
fixtures and incapable of leaking: the profile it reads was computed on the training
frame alone, so no test-set quantity can reach a decision.

Testing a rule needs no data at all:

```python
def test_monetary_rule_fires_on_price():
    profile = edaprep.profile(pd.DataFrame({"unit_price": [1.0, 2.0, 3.0] * 50}))
    context = RuleContext(profile=profile, config=Config())
    decision = monetary_columns_use_log(profile["unit_price"], context)
    assert decision.action == "transform_log1p"
```

### 2.2 Returning `None`

A rule that does not apply returns `None` and the next rule is tried. Returning a
"do nothing" `Decision` is different and sometimes better: an action named `no_*` emits
no transformer step but still appears in `explain()`, which is how "no scaling, because
tree models are invariant to monotone rescaling" gets said out loud.

---

## 3. Editing a plan directly

A `Plan` is inert data. Read it, change it, store it, re-run it.

```python
plan = edaprep.AutoPipeline(target="churn").plan(train_df)

plan = plan.without_stage(Stage.SCALE)          # frozen; returns a new plan
plan = plan.without_columns(["legacy_field"])

import json
json.dump(plan.to_dict(), open("plan.json", "w"))
restored = edaprep.Plan.from_dict(json.load(open("plan.json")))
```

Storing the plan next to a trained model records exactly how its inputs were built —
the thing `processed_train.csv` in notebook practice could not do.

---

## 4. A custom backend

`backends/base.py` defines a narrow protocol — about fifteen operations: column access,
null mask, distinct count, quantiles, grouped mean, concat. `pandas_backend.py` is the
only implementation today.

The protocol is deliberately small, and only hot paths are written against it; cold
paths use pandas directly, because pretending otherwise would be abstraction for its own
sake. It exists because the frames of interest in notebook practice reach
590,000 × 434, and an Arrow-backed implementation is a foreseeable need — not because
backend-swapping is a goal in itself.

```python
from edaprep.backends.base import Backend

class ArrowBackend(Backend):
    ...
```

If you are considering this, read `docs/performance.md` first. The one place a
hand-written kernel looked obviously worthwhile in this library turned out to be 2.1×
slower than the pandas code it replaced.

---

## 5. Reserved extension points

Named seams that exist and are unused. Each costs nothing now (an unused enum member, a
registration function already used internally) and their absence would force a redesign
later.

| capability | seam |
|---|---|
| feature engineering | `Stage.FEATURE_ENGINEERING`, present in the stage enum and ordering |
| NLP | `preprocessing/text.py` detects and reports; a `TextVectorizer` slots into `Stage.ENCODE` |
| time series | `Stage.DATETIME` already produces calendar features; lag and rolling features are further transformers |
| data validation | `profiling/quality.py` issue records are already schema-shaped |
| GPU / distributed execution | the `backends/` protocol |
| model selection, HPO | consume `Plan` and `Report`; outside the package by design |

None is implemented in v1, and note that they should not be.

---

## 6. Contributing

```bash
pip install -e ".[dev]"
pytest                       # 353 tests
python benchmarks/bench.py
```

Two expectations beyond passing tests:

- **A performance claim needs a benchmark row.** `docs/performance.md` §5 sets the
  method: minimum of N runs, interleaved candidates, and baselines that compute the same
  statistics as the thing they are compared against.
- **A new decision threshold goes in `Thresholds`**, named, with a comment saying where
  the number came from. A magic number inside a rule is the defect this library was
  built to remove.
