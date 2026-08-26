"""Pipelines and the planner: composition, explainability, overrides, edge cases."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import edaprep
from edaprep import AutoPipeline, Config, Pipeline
from edaprep.exceptions import ConfigurationError, NotFittedError
from edaprep.planning import Plan, Planner
from edaprep.planning.rules import Rule, default_rules
from edaprep.preprocessing import MissingValueHandler, Scaler
from edaprep.profiling import profile
from edaprep.types import SemanticType, Stage


@pytest.fixture
def frame() -> pd.DataFrame:
    gen = np.random.default_rng(21)
    n = 400
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "age": np.where(gen.random(n) < 0.1, np.nan, gen.normal(40, 12, n)),
            "income": gen.lognormal(10, 1.2, n),
            "city": gen.choice([f"c{i}" for i in range(80)], n),
            "grade": gen.integers(1, 5, n),
            "flag": gen.choice(["yes", "no"], n),
            "signup": pd.to_datetime("2021-01-01")
            + pd.to_timedelta(gen.integers(0, 900, n), unit="D"),
            "const": 3.0,
            "y": (gen.random(n) < 0.3).astype(int),
        }
    )


# ============================== explicit Pipeline ====================================


def test_explicit_pipeline_runs_in_order(frame) -> None:
    pipe = Pipeline(
        [("impute", MissingValueHandler()), ("scale", Scaler(["age", "income"]))],
        target="y",
    )
    out = pipe.fit_transform(frame)
    assert out["age"].isna().sum() == 0
    assert out["age"].mean() == pytest.approx(0.0, abs=1e-9)
    assert list(pipe.named_steps) == ["impute", "scale"]


def test_builder_api_chains(frame) -> None:
    pipe = (
        Pipeline(target="y")
        .flag_missing()
        .handle_outliers(strategy="clip")
        .handle_missing()
        .group_rare_categories()
        .encode_categorical()
        .scale_numeric()
    )
    out = pipe.fit_transform(frame)
    assert len(pipe) == 6
    assert out.isna().sum().sum() == 0


def test_pipeline_indexing_by_name_and_position(frame) -> None:
    pipe = Pipeline(target="y").handle_missing().scale_numeric()
    pipe.fit(frame)
    assert isinstance(pipe["scaler"], Scaler)
    assert isinstance(pipe[0], MissingValueHandler)


def test_duplicate_step_names_rejected() -> None:
    pipe = Pipeline()
    pipe.add(Scaler(), name="s")
    with pytest.raises(ConfigurationError, match="already exists"):
        pipe.add(Scaler(), name="s")


def test_auto_generated_step_names_are_unique() -> None:
    pipe = Pipeline().scale_numeric().scale_numeric()
    assert list(pipe.named_steps) == ["scaler", "scaler_2"]


def test_non_transformer_step_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Transformer"):
        Pipeline([("bad", object())])  # type: ignore[list-item]


def test_pipeline_get_set_params(frame) -> None:
    pipe = Pipeline(target="y").handle_missing().scale_numeric()
    params = pipe.get_params()
    assert "scaler__strategy" in params
    pipe.set_params(scaler__strategy="minmax")
    assert pipe["scaler"].strategy == "minmax"


def test_pipeline_set_params_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="Invalid parameter"):
        Pipeline().scale_numeric().set_params(nope__x=1)


# ============================== AutoPipeline ==========================================


def test_autopipeline_produces_a_model_ready_frame(frame) -> None:
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0)
    out = pipe.fit_transform(frame)
    assert out.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(d) for d in out.dtypes)
    assert "y" not in out.columns
    assert "row_id" not in out.columns  # identifier dropped
    assert "const" not in out.columns  # constant dropped


def test_autopipeline_is_deterministic(frame) -> None:
    a = AutoPipeline(target="y", random_state=7).fit_transform(frame)
    b = AutoPipeline(target="y", random_state=7).fit_transform(frame)
    pd.testing.assert_frame_equal(a, b)


def test_random_state_changes_only_stochastic_parts(frame) -> None:
    """Cross-fitting folds differ; everything else must not."""
    a = AutoPipeline(target="y", random_state=1).fit_transform(frame)
    b = AutoPipeline(target="y", random_state=2).fit_transform(frame)
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_series_equal(a["age"], b["age"])


def test_plan_without_fitting_touches_no_state(frame) -> None:
    pipe = AutoPipeline(target="y", model_family="tree")
    plan = pipe.plan(frame)
    assert isinstance(plan, Plan)
    assert len(plan) > 0
    with pytest.raises(NotFittedError):
        pipe.transform(frame)


def test_model_family_changes_the_plan(frame) -> None:
    tree = AutoPipeline(target="y", model_family="tree", random_state=0).fit(frame)
    linear = AutoPipeline(target="y", model_family="linear", random_state=0).fit(frame)

    # Tree pipelines still record a SCALE *decision* per column -- explain() should say
    # why nothing happens -- but it is a no-op, so no SCALE step is emitted.
    tree_actions = {d.column: d.action for d in tree.plan_.decisions if d.stage is Stage.SCALE}
    assert set(tree_actions.values()) == {"no_scaling"}
    assert not any(s.stage is Stage.SCALE for s in tree.plan_)
    assert any(s.stage is Stage.SCALE for s in linear.plan_)

    tree_enc = {d.column: d.action for d in tree.plan_.decisions if d.stage is Stage.ENCODE}
    linear_enc = {
        d.column: d.action for d in linear.plan_.decisions if d.stage is Stage.ENCODE
    }
    assert tree_enc["city"] == "encode_ordinal"
    assert linear_enc["city"] == "encode_target"


def test_tree_family_skips_transforms(frame) -> None:
    tree = AutoPipeline(target="y", model_family="tree", random_state=0).fit(frame)
    decisions = [d for d in tree.plan_.decisions if d.stage is Stage.TRANSFORM]
    assert {d.action for d in decisions} == {"no_transform"}
    assert "invariant to monotone transforms" in decisions[0].rationale
    assert not any(s.stage is Stage.TRANSFORM for s in tree.plan_)


def test_explain_names_the_rule_and_the_measurement(frame, capsys) -> None:
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0).fit(frame)
    text = pipe.explain("income")
    captured = capsys.readouterr().out
    assert "income" in text and text in captured
    assert "skew" in text  # the measurement, not just the verdict


def test_explain_before_fit_raises(frame) -> None:
    with pytest.raises(NotFittedError):
        AutoPipeline(target="y").explain()


def test_transformations_frame(frame) -> None:
    pipe = AutoPipeline(target="y", random_state=0).fit(frame)
    table = pipe.transformations_
    assert set(table.columns) == {
        "column",
        "stage",
        "action",
        "rationale",
        "rule",
        "source",
    }
    assert (table["rationale"].str.len() > 0).all()


def test_statistics_exposes_learned_parameters(frame) -> None:
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0).fit(frame)
    stats = pipe.statistics_
    assert "scaler" in stats
    assert "centers_" in stats["scaler"]


def test_unknown_target_raises_with_a_helpful_message(frame) -> None:
    with pytest.raises(KeyError, match="not a column"):
        AutoPipeline(target="nope").fit(frame)


# ============================== overrides =============================================


def test_column_override_wins_and_is_labelled(frame) -> None:
    config = Config(random_state=0)
    config.column("age").imputation = "mean"
    config.column("income").outlier_strategy = "clip"
    config.column("city").encoding = "frequency"

    pipe = AutoPipeline(target="y", config=config).fit(frame)
    by_column = {(d.column, d.stage): d for d in pipe.plan_.decisions}

    age = by_column[("age", Stage.MISSING)]
    assert age.action == "impute_mean"
    assert age.is_override
    assert "configuration" in age.rationale

    assert by_column[("city", Stage.ENCODE)].action == "encode_frequency"
    assert pipe["categorical_encoder"].assignments_["city"] == "frequency"
    assert len(pipe.plan_.overrides) >= 3


def test_semantic_type_override_changes_downstream_treatment(frame) -> None:
    """Declaring grade numeric must stop it being encoded as a category."""
    config = Config(random_state=0)
    config.column("grade").semantic_type = "numeric"
    pipe = AutoPipeline(target="y", model_family="linear", config=config).fit(frame)
    encode = {d.column for d in pipe.plan_.decisions if d.stage is Stage.ENCODE}
    assert "grade" not in encode


def test_drop_override(frame) -> None:
    config = Config(random_state=0)
    config.column("income").drop = True
    pipe = AutoPipeline(target="y", config=config).fit(frame)
    assert "income" in pipe.plan_.dropped_columns
    assert "income" not in pipe.transform(frame).columns


def test_bulk_overrides(frame) -> None:
    config = Config(random_state=0).set_columns(
        {"age": {"imputation": "mean"}, "city": {"encoding": "frequency"}}
    )
    pipe = AutoPipeline(target="y", config=config).fit(frame)
    assert pipe["missing_value_handler"].strategies_["age"] == "mean"


def test_unknown_override_key_rejected() -> None:
    with pytest.raises(ConfigurationError, match="not a valid value"):
        Config().set_columns({"a": {"nonsense": 1}})


def test_invalid_override_value_rejected() -> None:
    config = Config()
    config.column("a").imputation = "telepathy"
    with pytest.raises(ConfigurationError, match="not a valid value"):
        config.validate()


def test_global_strategy_pins_every_column(frame) -> None:
    config = Config(random_state=0, scaling="minmax", categorical_encoding="frequency")
    pipe = AutoPipeline(target="y", config=config).fit(frame)
    scale = [d for d in pipe.plan_.decisions if d.stage is Stage.SCALE]
    assert scale and all(d.action == "scale_minmax" for d in scale)


# ============================== planner ===============================================


def test_planner_never_receives_a_dataframe(frame) -> None:
    """The structural guarantee: planning is a pure function of the profile."""
    prof = profile(frame, target="y")
    plan = Planner(Config(random_state=0)).plan(prof)
    assert len(plan) > 0
    assert plan.target == "y"


def test_plan_round_trips_through_json(frame) -> None:
    prof = profile(frame, target="y")
    plan = Planner(Config(random_state=0, model_family="linear")).plan(prof)
    restored = Plan.from_dict(json.loads(plan.to_json()))
    assert len(restored) == len(plan)
    assert [s.transformer for s in restored] == [s.transformer for s in plan]
    assert restored.dropped_columns == plan.dropped_columns
    assert len(restored.decisions) == len(plan.decisions)


def test_plan_summary_and_explain_render(frame) -> None:
    prof = profile(frame, target="y")
    plan = Planner(Config(random_state=0)).plan(prof)
    assert "Preprocessing plan" in plan.summary()
    assert "income" in plan.explain()
    assert str(plan) == plan.summary()


def test_plan_editing_is_non_destructive(frame) -> None:
    prof = profile(frame, target="y")
    plan = Planner(Config(random_state=0, model_family="linear")).plan(prof)
    trimmed = plan.without_stage(Stage.SCALE)
    assert any(s.stage is Stage.SCALE for s in plan)  # original untouched
    assert not any(s.stage is Stage.SCALE for s in trimmed)


def test_plan_without_columns(frame) -> None:
    prof = profile(frame, target="y")
    plan = Planner(Config(random_state=0)).plan(prof)
    trimmed = plan.without_columns(["income"])
    assert not any(d.column == "income" for d in trimmed.decisions)


def test_stage_order_is_the_documented_one(frame) -> None:
    prof = profile(frame, target="y")
    plan = Planner(Config(random_state=0, model_family="linear")).plan(prof)
    orders = [s.stage.order for s in plan]
    assert orders == sorted(orders)
    stages = [s.stage for s in plan]
    # Flags before datetime expansion, outliers before imputation.
    if Stage.MISSING_FLAG in stages and Stage.DATETIME in stages:
        assert stages.index(Stage.MISSING_FLAG) < stages.index(Stage.DATETIME)
    if Stage.OUTLIERS in stages and Stage.MISSING in stages:
        assert stages.index(Stage.OUTLIERS) < stages.index(Stage.MISSING)


def test_custom_rule_can_pre_empt_a_builtin(frame) -> None:
    """The documented extension point: register a higher-priority rule."""
    from edaprep.planning.decisions import Decision

    def always_maxabs(cp, ctx):
        if cp.semantic is not SemanticType.NUMERIC or cp.is_target:
            return None
        return Decision(
            cp.name,
            Stage.SCALE,
            "scale_maxabs",
            params={"scaling": "maxabs"},
            rationale="house rule: everything is max-abs scaled",
            rule="house_maxabs",
        )

    rules = default_rules()
    rules.register(Rule("house_maxabs", Stage.SCALE, always_maxabs, priority=999))
    pipe = AutoPipeline(
        target="y",
        model_family="linear",
        random_state=0,
        planner=Planner(Config(random_state=0, model_family="linear"), rules=rules),
    ).fit(frame)
    # The custom rule only claims NUMERIC columns; ordinal and binary ones fall
    # through to the built-in, which is exactly how priority-ordered rules should work.
    scale = {
        d.column: d.action for d in pipe.plan_.decisions if d.stage is Stage.SCALE
    }
    assert scale["income"] == "scale_maxabs"
    assert scale["age"] == "scale_maxabs"
    assert scale["grade"] == "scale_standard"


def test_planning_notes_flag_a_missing_model_family(frame) -> None:
    pipe = AutoPipeline(target="y", random_state=0).fit(frame)
    assert any("model_family" in note for note in pipe.plan_.notes)


# ============================== report ================================================


def test_report_renders_and_serialises(frame) -> None:
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0).fit(frame)
    report = pipe.report_
    text = report.summary()
    assert "edaprep report" in text
    assert "Leakage audit" in text

    data = json.loads(report.to_json())
    assert data["n_features_in"] > 0
    assert "plan" in data and "profile" in data

    html = report.to_html()
    assert "<title>edaprep report</title>" in html
    assert "http://" not in html and "https://" not in html  # self-contained


def test_report_counts_what_happened(frame) -> None:
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0)
    pipe.fit(frame)
    pipe.transform(frame)
    imputed = [
        e for e in pipe.report_.entries if e.action == "impute" and e.phase == "transform"
    ]
    assert imputed
    assert imputed[0].effect["n_values_imputed"] > 0


def test_repeated_transform_does_not_grow_the_journal(frame) -> None:
    pipe = AutoPipeline(target="y", random_state=0).fit(frame)
    pipe.transform(frame)
    first = len(pipe.report_.entries)
    for _ in range(5):
        pipe.transform(frame)
    assert len(pipe.report_.entries) == first


# ============================== edge cases ============================================


def test_single_column_numeric_frame() -> None:
    frame = pd.DataFrame({"x": np.arange(100.0)})
    out = AutoPipeline(random_state=0).fit_transform(frame)
    assert out.shape[0] == 100


def test_numeric_only_frame() -> None:
    gen = np.random.default_rng(1)
    frame = pd.DataFrame({f"c{i}": gen.normal(size=200) for i in range(5)})
    out = AutoPipeline(random_state=0).fit_transform(frame)
    assert out.shape == (200, 5)


def test_categorical_only_frame() -> None:
    gen = np.random.default_rng(2)
    frame = pd.DataFrame({f"c{i}": gen.choice(list("abc"), 200) for i in range(3)})
    out = AutoPipeline(random_state=0).fit_transform(frame)
    assert all(pd.api.types.is_numeric_dtype(d) for d in out.dtypes)


def test_single_row_frame() -> None:
    frame = pd.DataFrame({"a": [1.0], "b": ["x"], "y": [0]})
    out = AutoPipeline(target="y", random_state=0).fit_transform(frame)
    assert len(out) == 1


def test_frame_where_everything_is_dropped() -> None:
    """All-constant input: the pipeline must produce an empty frame, not crash."""
    frame = pd.DataFrame({"a": [1.0] * 50, "b": ["x"] * 50, "y": np.arange(50)})
    pipe = AutoPipeline(target="y", random_state=0)
    out = pipe.fit_transform(frame)
    assert out.shape[1] == 0
    assert len(pipe.plan_.dropped_columns) == 2


def test_extreme_missingness() -> None:
    gen = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "mostly_gone": np.where(gen.random(300) < 0.95, np.nan, 1.0),
            "fine": gen.normal(size=300),
            "y": gen.integers(0, 2, 300),
        }
    )
    pipe = AutoPipeline(target="y", random_state=0).fit(frame)
    assert "mostly_gone" in pipe.plan_.dropped_columns


def test_high_cardinality_is_not_one_hot_encoded() -> None:
    gen = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "c": gen.choice([f"v{i}" for i in range(5000)], 8000),
            "y": gen.integers(0, 2, 8000),
        }
    )
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0).fit(frame)
    out = pipe.transform(frame)
    assert out.shape[1] < 10  # not 5000 columns


def test_duplicate_rows_are_reported_not_removed() -> None:
    frame = pd.DataFrame({"a": [1.0, 1.0, 2.0] * 50, "y": [0, 0, 1] * 50})
    pipe = AutoPipeline(target="y", random_state=0).fit(frame)
    assert len(pipe.transform(frame)) == len(frame)
    assert any("duplicate rows" in note for note in pipe.plan_.notes)


def test_unseen_categories_at_transform_time() -> None:
    gen = np.random.default_rng(5)
    train = pd.DataFrame(
        {"c": gen.choice(list("abc"), 300), "y": gen.integers(0, 2, 300)}
    )
    test = pd.DataFrame({"c": ["a", "z", "q"], "y": [0, 1, 0]})
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0).fit(train)
    out = pipe.transform(test)
    assert out.isna().sum().sum() == 0
    assert list(out.columns) == list(pipe.transform(train).columns)


def test_sklearn_pipeline_interoperability() -> None:
    """edaprep transformers must drop into an sklearn Pipeline."""
    sk_pipeline = pytest.importorskip("sklearn.pipeline")
    from sklearn.linear_model import LogisticRegression

    gen = np.random.default_rng(6)
    X = pd.DataFrame({"a": gen.normal(size=200), "b": gen.normal(size=200)})
    y = (gen.random(200) < 0.4).astype(int)

    model = sk_pipeline.Pipeline(
        [("impute", MissingValueHandler()), ("scale", Scaler()), ("clf", LogisticRegression())]
    )
    model.fit(X, y)
    assert model.predict(X).shape == (200,)
    assert "scale__strategy" in model.get_params()


def test_sklearn_tag_protocol_is_implemented() -> None:
    """scikit-learn 1.6+ asks estimators for tags; 1.8 raises if they cannot answer.

    Without ``__sklearn_tags__`` this warns on 1.6/1.7 and breaks outright on 1.8, so
    the advertised sklearn interoperability would silently rot. Asserted here rather
    than left to a deprecation warning nobody reads.
    """
    pytest.importorskip("sklearn", minversion="1.6")
    from sklearn.utils import get_tags

    tags = get_tags(MissingValueHandler())
    assert tags.estimator_type == "transformer"
    assert tags.input_tags.allow_nan
    # `required` must track uses_target, or the two can drift apart.
    assert get_tags(edaprep.TargetEncoder()).target_tags.required is True
    assert get_tags(Scaler()).target_tags.required is False


def test_sklearn_interop_emits_no_deprecation_warnings() -> None:
    """The interop path must be clean, not merely working."""
    sk_pipeline = pytest.importorskip("sklearn.pipeline")
    pytest.importorskip("sklearn", minversion="1.6")
    import warnings

    from sklearn.linear_model import LogisticRegression

    gen = np.random.default_rng(11)
    X = pd.DataFrame({"a": gen.normal(size=120), "b": gen.normal(size=120)})
    y = (gen.random(120) < 0.4).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sk_pipeline.Pipeline(
            [("scale", Scaler()), ("clf", LogisticRegression(max_iter=200))]
        ).fit(X, y)


def test_public_api_surface() -> None:
    for name in edaprep.__all__:
        assert hasattr(edaprep, name), f"{name} is exported but missing"


def test_visualization_import_error_is_helpful(monkeypatch) -> None:
    """Without matplotlib, plotting must give an install hint, not a bare ImportError.

    The subpackage itself imports fine without matplotlib -- pyplot is loaded on
    demand -- so the error belongs at the point of drawing, which is what this checks.
    """
    import sys

    # A sys.modules entry set to None makes any `import matplotlib` raise ImportError.
    # Patching builtins.__import__ would not work: importlib goes through the import
    # machinery directly and never calls it.
    for name in list(sys.modules):
        if name.startswith("matplotlib"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    from edaprep.visualization import plots

    with pytest.raises(ImportError, match=r"pip install"):
        plots.missing_bar(pd.DataFrame({"a": [1.0, np.nan]}))


def test_visualization_renders_without_showing() -> None:
    """Plot helpers return an Axes and never call plt.show()."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "num": np.where(gen.random(200) < 0.1, np.nan, gen.lognormal(size=200)),
            "cat": gen.choice(list("abc"), 200),
            "y": gen.integers(0, 2, 200),
        }
    )
    viz = edaprep.visualization
    for call in (
        lambda: viz.missing_bar(data),
        lambda: viz.missing_matrix(data),
        lambda: viz.histogram(data, "num"),
        lambda: viz.boxplot(data, ["num"]),
        lambda: viz.category_bar(data, "cat"),
        lambda: viz.target_distribution(data, "y"),
        lambda: viz.feature_target(data, "num", "y"),
        lambda: viz.correlation_heatmap(data[["num"]].corr()),
    ):
        ax = call()
        assert ax is not None
        plt.close("all")

    figure = viz.plot_profile(data, target="y")
    assert figure is not None
    plt.close("all")


def test_cast_introduced_nan_is_imputed() -> None:
    """Placeholder strings become NaN at Stage.CAST, so Stage.MISSING must plan for them.

    Regression: the profile measures ``n_missing`` on the raw frame, where a blank
    string is still a string, so the column reports 0% missing.  The cast then parses
    it to float and those blanks become NaN.  The imputation rule used to decline on
    ``n_missing == 0`` and leave NaN in supposedly ML-ready output, which crashes any
    estimator that does not accept them.

    Shape taken from Telco Customer Churn's ``TotalCharges`` column.
    """
    gen = np.random.default_rng(7)
    n = 300
    charges = [f"{v:.2f}" for v in gen.uniform(20, 8000, size=n)]
    for i in range(0, 30, 3):          # 10 blanks, as strings
        charges[i] = " "
    frame = pd.DataFrame(
        {
            "total_charges": charges,          # object dtype purely because of the blanks
            "tenure": gen.integers(1, 72, size=n),
            "churn": gen.integers(0, 2, size=n),
        }
    )
    # Not a dtype equality check: pandas 3 stores text as StringDtype where pandas 2
    # used object, and the precondition here is only that the column is non-numeric
    # and that its blanks are still strings rather than NaN.
    assert not pd.api.types.is_numeric_dtype(frame["total_charges"])
    assert frame["total_charges"].isna().sum() == 0, "blanks are strings, not NaN"

    pipe = AutoPipeline(target="churn", model_family="linear", random_state=0)
    out = pipe.fit_transform(frame)

    assert out.isna().sum().sum() == 0, "cast-introduced NaN reached the output"

    decisions = [d for d in pipe.plan_.decisions if d.column == "total_charges"]
    actions = {d.action for d in decisions}
    assert "impute_median" in actions, f"no imputation planned; got {actions}"

    impute = next(d for d in decisions if d.action == "impute_median")
    assert impute.params.get("cast_missing") == 10
    assert "placeholder" in impute.rationale
    assert "0.0% missing" in impute.rationale, (
        "the rationale must name the placeholder count rather than silently "
        "reporting 0% missing while imputing anyway"
    )


def test_cast_introduced_nan_imputed_on_unseen_test_rows() -> None:
    """The fitted statistic must carry to transform, not be recomputed per batch."""
    gen = np.random.default_rng(11)
    def make(n: int, blanks: int) -> pd.DataFrame:
        vals = [f"{v:.2f}" for v in gen.uniform(10, 500, size=n)]
        for i in range(blanks):
            vals[i] = ""
        return pd.DataFrame(
            {"amount": vals, "y": gen.integers(0, 2, size=n)}
        )

    train, test = make(200, 8), make(80, 5)
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0).fit(train)

    assert pipe.transform(train).isna().sum().sum() == 0
    assert pipe.transform(test).isna().sum().sum() == 0


def test_column_with_no_missing_and_no_placeholders_plans_no_imputation() -> None:
    """The fix must not schedule imputation on every cast column."""
    gen = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "clean_text_number": [f"{v:.1f}" for v in gen.uniform(1, 100, size=200)],
            "y": gen.integers(0, 2, size=200),
        }
    )
    pipe = AutoPipeline(target="y", model_family="linear", random_state=0).fit(frame)
    actions = {d.action for d in pipe.plan_.decisions if d.column == "clean_text_number"}
    assert not any(a.startswith("impute_") for a in actions), actions
