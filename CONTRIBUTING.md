# Contributing to edaprep

Thanks for looking. This document tells you how the project is put together, what
"done" means here, and where the useful work is.

**New here?** Issues labelled [`good first issue`](https://github.com/bijay-odyssey/edaprep/labels/good%20first%20issue)
are scoped to be self-contained: each one names the file to change and the test to
write. Comment on one to claim it — no need to ask permission first.

---

## Getting set up

```bash
git clone https://github.com/bijay-odyssey/edaprep
cd edaprep
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                                              # ~15s
ruff check src/ tests/ benchmarks/ examples/
python examples/end_to_end.py                       # the worked example
```

Python 3.9 or newer. There is no build step and no compiled code.

---

## What this project is, in one page

The core idea is that **deciding** and **doing** are separate:

```
DataFrame ─→ DatasetProfile ─→ Planner ─→ Plan ─→ Pipeline ─→ DataFrame + Report
             (measurement)     (decision)  (data)  (execution)
```

Three consequences drive almost every design rule below:

1. **The planner never sees a DataFrame.** It takes a `DatasetProfile` and a `Config`
   and returns a `Plan`. That is what makes it unit-testable without fixtures, and
   what makes it structurally incapable of leaking.
2. **A `Plan` is inert data.** No transformer references, no closures. It round-trips
   through `to_dict()`/`from_dict()`, which is what gives us `explain()`, per-column
   overrides and reproducibility for free.
3. **`transform` is a pure function of fitted state.** Everything learned lives in
   `name_` attributes written only inside `_fit`.

`docs/architecture.md` covers this properly. `docs/design-rationale.md` explains *why*
any of it is necessary — read that one first if you want to understand the project
rather than just patch it.

---

## The rules that are actually enforced

These are not style preferences. Each is checked by a test, and a PR that breaks one
will fail CI.

### 1. `transform` may not compute a statistic over its input

This is the whole point of the library. If `transform` calls `.mean()`, `.median()`,
`.quantile()`, `.std()`, `.mode()`, `.value_counts()`, `.nunique()`, `.corr()` or
`.factorize()` on the frame it was given, the output for one row depends on the other
rows in the batch — which is leakage.

Two tests enforce it:

- `test_transform_methods_compute_no_statistics` parses every `_transform` body with
  `ast` and fails on a forbidden call, naming the offender.
- `test_transform_output_is_independent_of_batching` transforms a frame whole, then
  row by row, and requires identical output.

If you have a legitimate exception, add it to the allow-list **with the reason**, and
expect that reason to be scrutinised in review.

### 2. Never mutate the caller's frame

`transform()` returns a new frame. Use
`ColumnTransformerMixin._rebuild(X, replacements, added)`, which reuses untouched
column blocks by reference — a step touching 5 of 400 columns should allocate 5
columns, not copy the frame. Asserted for every transformer by a parametrised test.

### 3. Decision thresholds go in `Thresholds`, named, with their provenance

A bare `0.9` or `1.5` inside a rule is the exact defect this library exists to remove.
Add the constant to `edaprep/config.py` with a comment saying where the number comes
from.

### 4. Every automatic decision needs an English rationale

A `Decision` without a `rationale` is a black box. Write the sentence a user would
need to understand *why*, and name the measurement:

```python
rationale=f"skew {skew:.2f} is heavy (>= {thresholds.skew_heavy}); Yeo-Johnson is "
          f"used rather than Box-Cox because it is defined for the column's range"
```

### 5. Nothing is discarded silently

Dropped columns, imputed values, grouped categories, clipped rows and unseen
categories all get counted and recorded through `context.journal`.

### 6. No `warnings.filterwarnings` at module scope

Suppressing warnings globally is how real bugs hide. Scope any suppression to the
single expression that needs it, and say why.

---

## Adding a transformer

```python
class MyTransformer(Transformer, ColumnTransformerMixin):
    stage = Stage.OUTLIERS          # where it belongs in the ordering
    uses_target = False             # True if _fit reads y — audited, raises without it
    cross_fitted = False            # True only if fit_transform must differ

    def _select_columns(self, X, context):
        """Which columns to act on when the caller did not say.
        Consult context.column_profile(name) so a numeric step avoids a text column."""

    def _fit(self, X, y, context):
        """Learn state. Write only trailing-underscore attributes."""

    def _transform(self, X, context):
        """Apply it. Compute nothing over X."""
```

The base class handles validation, journalling, the fitted flag, `get_params`,
`set_params` and schema checks, so you cannot forget them.

`docs/extending.md` has a complete worked example, plus how to register a planning
rule and what a rule is allowed to look at.

---

## Testing expectations

Every behavioural change needs a test. Two things we care about more than coverage:

**Tests should be non-vacuous.** If you fix a bug, confirm the test fails without the
fix. Several tests in this repo document exactly that, e.g. the read-only-array
regression tests reproduce the original CI failure on any pandas version.

**Numerical code is checked against a trusted implementation** — NumPy, pandas, SciPy
or scikit-learn — not against itself. Where we deliberately disagree with a reference
(pandas' absolute `1e-14` moment cut-off, for instance), the test says so and explains
why we are the correct one.

```bash
pytest                                  # everything
pytest tests/test_leakage.py -v         # the ones that matter most
pytest -k "outlier"                     # by keyword
```

---

## Performance claims need a benchmark

`docs/performance.md` §5 sets the method: minimum of N runs, candidates interleaved,
and baselines that compute **the same statistics** as the thing they are compared
against.

```bash
python benchmarks/bench.py --rows 100000 --repeat 5
```

The cautionary tale is in §1: a hand-written NumPy kernel in this repo turned out to
be 2.1× *slower* than the pandas code it replaced, and was deleted. Measure before you
optimise, and be willing to delete.

---

## Pull requests

- Branch from `main`.
- One logical change per PR. A refactor and a fix in the same diff is two PRs.
- Commit messages: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `ci:`, `refactor:`.
  Explain **why** in the body, not just what — the git log is documentation.
- Run `pytest` and `ruff check` before pushing.
- CI runs Python 3.9–3.13 on Linux, plus macOS and Windows. It also installs the built
  wheel and uses it from outside the source tree.

Draft PRs are welcome if you want direction before finishing.

---

## Things that are deliberately out of scope

Proposals for these will likely be declined, so it is worth knowing up front:

- **Model training, hyperparameter search, model selection.** This library prepares
  data; it does not model.
- **Resampling (SMOTE and friends).** Class imbalance is measured and reported.
  Resampling belongs after the train/test split and with the model.
- **Transforming the target in place.** Reported with a recommendation instead — it is
  too easy to forget to invert.
- **Automatically dropping suspected leaks.** Flagged as an error, never removed: a
  legitimately strong feature is indistinguishable from a leak without domain
  knowledge.
- **Native code (Rust/Cython) without a benchmark justifying it.** See above.

Extension points for the genuinely-future items (NLP, time series, GPU, distributed)
are listed in `docs/extending.md` §5.

---

## Questions

Open an issue, or start a
[Discussion](https://github.com/bijay-odyssey/edaprep/discussions) if it is more of a
conversation than a bug. Design disagreements are welcome — several decisions here are
trade-offs rather than facts, and the reasoning is written down precisely so it can be
argued with.
