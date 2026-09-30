"""Regression tests for the results-table contract and the direct chart API.

Three groups of tests here, all guarding things that were broken or absent:

* **Gini through `ratios`** -- both `gini_args` paths raised and had their
  exception swallowed into the study's `errors` key, so a caller asking for a
  decomposition silently got no decomposition.
* **Presentation contract** -- column order used to follow whatever order the
  caller listed `indicators` in, and missing cells came out as NaN, which
  reads as a result rather than as "not computed here".
* **Chart API** -- the frame-first chart functions, which exist so nobody has
  to build an 'asr' column or pre-run `vei_analysis` to draw a picture.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import ratio_study as rs
from ratio_study.ratio_study import ratios

N = 400


@pytest.fixture
def sales():
    """400 sales with a grouping column and two characteristics."""
    rng = np.random.default_rng(11)
    mv = rng.lognormal(12.4, 0.5, N)
    return pd.DataFrame({
        "sale": mv,
        "assess": mv * rng.normal(1.0, 0.10, N),
        "nbhd": rng.choice(["A", "B", "C"], N),
        "quality": rng.integers(3, 8, N),
        "area": rng.integers(700, 3500, N),
    })


def _study(sales, **kwargs):
    return ratios(input_df=sales, market_value_header="sale",
                  assessment_header="assess", **kwargs)


# --------------------------------------------------------------------------
# Gini paths through `ratios`
# --------------------------------------------------------------------------

def test_gini_decomposition_returns_results_not_an_error(sales):
    """`gini_decomp` was called with a `model_name` it does not accept, and
    with the ratios dict where it wanted a (gini package, ungrouped package)
    pair. Both raised, and `ratios` caught the exception into 'errors'."""
    out = _study(sales, group_by="nbhd",
                 gini_args={"gini_decomposition": True}, return_dict=True)

    assert "errors" not in out, out.get("errors")
    decomposition = out["dict_gini_decomposition"]["nbhd"]
    assert {"Gini_within_groups", "Gini_between_groups", "Gini_residual"} <= set(decomposition)

    table = decomposition["dataframe"]
    assert isinstance(table, pd.DataFrame)
    # Quintos (2021) layout: a Total row, the within-group block, Between,
    # and the residual interaction term.
    effects = set(table["Effect"].dropna())
    assert {"Total", "Within", "Between", "Residual"} <= effects


def test_gini_factors_without_group_by_returns_results_not_an_error(sales):
    """`gini_decomp_results` was only bound inside the `group_by` branch, so
    asking for factors alone hit an UnboundLocalError."""
    out = _study(sales, gini_args={"gini_factors_variables": ["area", "quality"]},
                 return_dict=True)

    assert "errors" not in out, out.get("errors")
    factors = out["dict_gini_decomposition"]["gini_factors"]
    assert isinstance(factors, pd.DataFrame)
    assert set(factors["Variable"]) == {"area", "quality"}


# --------------------------------------------------------------------------
# Column order
# --------------------------------------------------------------------------

def test_identification_columns_come_first(sales):
    out = _study(sales, group_by="nbhd")
    assert list(out.columns)[:3] == ["Level", "Group", "Count"]


def test_indicator_order_is_canonical_not_caller_order(sales):
    """Two callers asking for the same indicators in different orders get the
    same table, so studies in a series stay comparable."""
    forward = _study(sales, indicators=["Median", "CoD", "PRD", "PRB"])
    backward = _study(sales, indicators=["PRB", "PRD", "CoD", "Median"])
    assert list(forward.columns) == list(backward.columns)


def test_level_then_uniformity_then_vertical_equity(sales):
    out = _study(sales, indicators=["Median", "Mean", "Weighted_Mean", "CoD",
                                    "CoV", "PRD", "PRB", "VEI", "RMSE"])
    order = list(out.columns)
    position = {name: order.index(name) for name in
                ("Median", "Mean", "Weighted_Mean", "CoD", "CoV",
                 "PRB", "PRD", "VEI", "RMSE")}

    assert position["Median"] < position["Mean"] < position["Weighted_Mean"]
    assert position["Weighted_Mean"] < position["CoD"] < position["CoV"]
    # PRB before the retiring PRD, and both before the preferred VEI.
    assert position["CoV"] < position["PRB"] < position["PRD"] < position["VEI"]
    # Accuracy measures trail the compliance indicators.
    assert position["VEI"] < position["RMSE"]


def test_each_interval_sits_beside_its_indicator(sales):
    out = _study(sales, ci_mode='bootstrap')
    order = list(out.columns)
    for indicator in ("Median", "CoD", "PRB", "PRD"):
        assert order.index(f"{indicator}_ci") == order.index(indicator) + 1


def test_the_unformatted_frame_shares_the_display_order(sales):
    out = _study(sales, group_by="nbhd", return_dict=True)
    raw = list(out["df_all_ratios"].columns)
    formatted = list(out["df_all_ratios_formatted"].columns)
    # The formatted frame adds comment/compliance columns, but everything it
    # shares with the raw frame appears in the same relative order.
    shared = [c for c in formatted if c in raw]
    assert [c for c in raw if c in shared] == shared


# --------------------------------------------------------------------------
# Blanks rather than NaN
# --------------------------------------------------------------------------

def test_no_nan_reaches_the_formatted_frame():
    """A group too small for an interval leaves a blank cell, not 'NaN'.

    Group 'A' has 8 sales, below the 20-sale bootstrap minimum, so its
    interval columns have nothing to carry.
    """
    rng = np.random.default_rng(3)
    mv = rng.lognormal(12.4, 0.5, 40)
    frame = pd.DataFrame({"sale": mv, "assess": mv * rng.normal(1.0, 0.1, 40),
                          "grp": ["A"] * 8 + ["B"] * 32})
    out = ratios(input_df=frame, market_value_header="sale",
                 assessment_header="assess", group_by="grp",
                 descriptive=True, confidence_intervals_aggs="both",
                 ci_mode='bootstrap')

    assert not out.isna().any().any()
    rendered = out.astype(str)
    assert not rendered.isin(["nan", "NaN", "<NA>", "None"]).any().any()

    small = out.loc[out["Group"] == "A"].iloc[0]
    assert small["Median_ci"] == ""


def test_blanks_survive_the_excel_export(tmp_path):
    """xlsx + group_by + descriptive used to raise `Invalid value '' for
    dtype boolean` in the Excel writer, and took the PDF down with it."""
    rng = np.random.default_rng(5)
    mv = rng.lognormal(12.4, 0.5, 60)
    frame = pd.DataFrame({"sale": mv, "assess": mv * rng.normal(1.0, 0.1, 60),
                          "grp": ["A"] * 10 + ["B"] * 50})

    rs.ratios_report(frame, assessment_header="assess", market_value_header="sale",
                     group_by="grp", descriptive=True, model_name="T",
                     export_directory=tmp_path, export_ratios="xlsx",
                     export_in_compacted_file="", return_dict=False)

    written = sorted(p.name for p in tmp_path.iterdir())
    assert any(name.endswith(".xlsx") for name in written), written
    # The PDF is built independently and must not be collateral damage.
    assert any(name.endswith(".pdf") for name in written), written
    assert "errors_in_ratio_study.txt" not in written


# --------------------------------------------------------------------------
# Chart API
# --------------------------------------------------------------------------

CHART_CALLS = [
    ("asr_scatter", lambda df, **kw: rs.asr_scatter(df, "assess", "sale", **kw)),
    ("asr_scatter_no_marginals",
     lambda df, **kw: rs.asr_scatter(df, "assess", "sale", x="area",
                                     marginals=False, lowess=False, **kw)),
    ("asr_histogram", lambda df, **kw: rs.asr_histogram(df, "assess", "sale", **kw)),
    ("asr_boxplot", lambda df, **kw: rs.asr_boxplot(df, "assess", "sale", by="nbhd", **kw)),
    ("asr_facet", lambda df, **kw: rs.asr_facet(df, "assess", "sale", col="nbhd",
                                                lowess=False, **kw)),
    ("vei_boxplot", lambda df, **kw: rs.vei_boxplot(df, assessment_header="assess",
                                                    market_value_header="sale", **kw)),
]


@pytest.mark.parametrize("name, call", CHART_CALLS, ids=[c[0] for c in CHART_CALLS])
def test_charts_take_a_frame_and_return_a_figure(sales, name, call):
    """No caller should have to build an 'asr' column or pre-run an analysis."""
    import matplotlib.pyplot as plt
    figure = call(sales, show=False) if name != "vei_boxplot" else call(sales)
    assert isinstance(figure, plt.Figure)
    plt.close("all")


@pytest.mark.parametrize("name, call", CHART_CALLS, ids=[c[0] for c in CHART_CALLS])
def test_charts_save_a_png_when_asked(sales, tmp_path, name, call):
    target = tmp_path / f"{name}.png"
    saved = call(sales, graph_filepath=str(target))
    assert str(saved) == str(target)
    assert target.exists() and target.stat().st_size > 0


@pytest.mark.parametrize("name, call", CHART_CALLS, ids=[c[0] for c in CHART_CALLS])
def test_charts_return_a_report_flowable_when_asked(sales, name, call):
    result = call(sales, return_object=True)
    assert isinstance(result, list) and len(result) == 2
    assert isinstance(result[0], str)


def test_charts_accept_an_existing_ratio_column(sales):
    import matplotlib.pyplot as plt
    frame = sales.assign(my_ratio=sales["assess"] / sales["sale"])
    figure = rs.asr_histogram(frame, asr_col="my_ratio", show=False)
    assert isinstance(figure, plt.Figure)
    plt.close("all")


def test_missing_value_columns_raise_a_useful_message(sales):
    with pytest.raises(ValueError, match="assessment_header"):
        rs.asr_scatter(sales)


def test_a_chart_names_what_it_left_out(sales, tmp_path):
    """Silent truncation reads as full coverage. `max_groups` has to say so."""
    figure = rs.asr_boxplot(sales, "assess", "sale", by="quality",
                            max_groups=2, show=False)
    notes = [t.get_text() for t in figure.texts]
    assert any("Not shown" in note for note in notes), notes


def test_lorenz_plot_needs_no_precomputed_gini(sales, tmp_path):
    target = tmp_path / "lorenz.png"
    assert str(rs.lorenz_plot(sales, "assess", "sale",
                              graph_filepath=str(target))) == str(target)
    assert target.exists()
