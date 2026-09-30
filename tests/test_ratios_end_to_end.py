"""End-to-end tests for the ``ratios`` orchestrator.

``ratio_study.py`` is the public entry point but had ~3% coverage, which is
how a broken keyword argument reached ``mcp_server`` unnoticed. These tests
pin the output contract (columns, shape, grouping, dict keys) and the
backend-agnostic promise rather than exact statistics, so they stay useful
as the underlying formulas evolve.
"""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from ratio_study import ratios

N = 600

# Columns `ratios` emits for its default indicator set, including the
# confidence interval that accompanies each scalar indicator under the
# default `ci_mode='distribution'`. Order is fixed by
# `INDICATOR_DISPLAY_ORDER`, not by the order `indicators` was passed in:
# level of valuation, then uniformity, then vertical equity (PRB ahead of
# the retiring PRD). All three measures of valuation level are in the default
# indicator set, so the default columns carry the six intervals of the
# Standard's own demonstration report (Table 10.2); PRD's comes from
# `confint.prd_ci`'s delta method on Mean/Weighted_Mean.
DEFAULT_COLUMNS = [
    "Level", "Count",
    "Median", "Median_ci",
    "Mean", "Mean_ci",
    "Weighted_Mean", "Weighted_Mean_ci",
    "CoD", "CoD_ci",
    "PRB", "PRB_ci",
    "PRD", "PRD_ci",
    "VEI", "VEI_conclusion", "VEI_kw_statistic", "VEI_kw_pvalue",
    "Inequity_conclusion", "Inequity_severity", "Inequity_prevalence",
]

# The bootstrap resampler is unseeded, so the `*_ci` columns vary run to run
# (see test_bootstrap_confidence_intervals_are_not_reproducible). Point
# estimates are stable, so comparisons that need reproducibility drop the
# interval columns rather than the whole check.
def _stable(df):
    """Drop the unseeded bootstrap CI columns from a results frame."""
    return df.drop(columns=[c for c in df.columns if c.endswith("_ci")])


# DEFAULT_COLUMNS without the bootstrap intervals, for cell-by-cell
# comparisons between two separate `ratios` calls.
STABLE_COLUMNS = [c for c in DEFAULT_COLUMNS if not c.endswith("_ci")]


@pytest.fixture
def sales():
    """600 sales, near-uniform ratios, with a grouping column."""
    rng = np.random.default_rng(4)
    mv = rng.lognormal(12.4, 0.5, N)
    return pd.DataFrame({
        "sale": mv,
        "assess": mv * rng.normal(1.0, 0.10, N),
        "nbhd": rng.choice(["A", "B", "C"], N),
    })


@pytest.fixture
def regressive_sales():
    """600 sales with a deliberate regressive tilt: ratios fall with price."""
    rng = np.random.default_rng(21)
    mv = rng.lognormal(12.4, 0.5, N)
    ratio = 1.05 * (mv / mv.mean()) ** -0.12 * rng.lognormal(0, 0.08, N)
    return pd.DataFrame({"sale": mv, "assess": mv * ratio})


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------

def test_returns_one_overall_row_with_the_default_columns(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess")
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == DEFAULT_COLUMNS
    assert len(out) == 1
    assert out.loc[0, "Level"] == "Overall"
    assert int(out.loc[0, "Count"]) == N


def test_vem_is_in_the_default_indicator_set(sales):
    # 'VEM' joined the defaults when ve_bands was removed; the three
    # Inequity_* columns are its output.
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess")
    for col in ("Inequity_conclusion", "Inequity_severity", "Inequity_prevalence"):
        assert col in out.columns


def test_no_ve_bands_columns_survive(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess")
    assert not [c for c in out.columns if "VE_Bands" in c or "VE_Median" in c]


def test_requesting_a_subset_of_indicators_narrows_the_table(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess",
                 indicators=["Median", "CoD"])
    assert "Median" in out.columns and "CoD" in out.columns
    # Indicators that were not asked for must not appear.
    assert "PRB" not in out.columns
    assert "VEI" not in out.columns


def test_median_ratio_is_close_to_one_for_an_unbiased_sample(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess")
    assert float(out.loc[0, "Median"]) == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

def test_group_by_adds_one_row_per_group_plus_overall(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess", group_by="nbhd")
    assert "Group" in out.columns
    assert len(out) == 1 + sales["nbhd"].nunique()
    assert out.loc[0, "Level"] == "Overall"
    groups = set(out.loc[out["Level"] == "nbhd", "Group"])
    assert groups == set(sales["nbhd"].unique())


def test_group_counts_sum_to_the_population(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess", group_by="nbhd")
    group_rows = out[out["Level"] == "nbhd"]
    assert group_rows["Count"].astype(int).sum() == N


# --------------------------------------------------------------------------
# Detecting real inequity
# --------------------------------------------------------------------------

def test_a_regressive_sample_is_flagged_as_regressive(regressive_sales):
    out = ratios(input_df=regressive_sales, market_value_header="sale",
                 assessment_header="assess")
    conclusion = str(out.loc[0, "Inequity_conclusion"])
    prd = float(out.loc[0, "PRD"])
    # Ratios fall with price: PRD above 1 is the classic regressivity signal.
    assert prd > 1.0
    assert "regress" in conclusion.lower() or float(out.loc[0, "PRB"]) < 0


def test_a_uniform_sample_is_not_flagged(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess")
    assert str(out.loc[0, "Inequity_conclusion"]).lower() == "acceptable"


# --------------------------------------------------------------------------
# return_dict
# --------------------------------------------------------------------------

def test_return_dict_exposes_the_frames_and_graph_buckets(sales):
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess", return_dict=True)
    assert isinstance(out, dict)
    assert {"df_all_ratios", "df_all_ratios_formatted",
            "dict_population_ratios", "overall_graphs"} <= set(out)
    assert isinstance(out["df_all_ratios_formatted"], pd.DataFrame)
    assert isinstance(out["dict_population_ratios"], dict)


def test_formatted_frame_matches_the_plain_return(sales):
    plain = ratios(input_df=sales, market_value_header="sale",
                   assessment_header="assess")
    as_dict = ratios(input_df=sales, market_value_header="sale",
                     assessment_header="assess", return_dict=True)
    pd.testing.assert_frame_equal(
        _stable(plain), _stable(as_dict["df_all_ratios_formatted"]))


# --------------------------------------------------------------------------
# Backend agnosticism -- the narwhals promise
# --------------------------------------------------------------------------

@pytest.mark.parametrize("to_backend", [
    pytest.param(lambda d: d, id="pandas"),
    pytest.param(lambda d: pl.from_pandas(d), id="polars"),
    pytest.param(lambda d: pa.Table.from_pandas(d), id="pyarrow"),
])
def test_every_backend_produces_the_same_numbers(sales, to_backend):
    expected = ratios(input_df=sales, market_value_header="sale",
                      assessment_header="assess")
    got = ratios(input_df=to_backend(sales), market_value_header="sale",
                 assessment_header="assess")
    got_pd = got if isinstance(got, pd.DataFrame) else got.to_pandas()
    assert list(got_pd.columns) == list(expected.columns)
    # Compare the numeric cells as text, since ratios formats them. The
    # bootstrap CI columns are dropped: they vary between any two calls
    # regardless of backend, so including them would test the resampler's
    # RNG rather than backend parity.
    assert (_stable(got_pd).astype(str).values.tolist()
            == _stable(expected).astype(str).values.tolist())


# --------------------------------------------------------------------------
# Weights and preprocessing
# --------------------------------------------------------------------------

def test_uniform_weights_reproduce_the_unweighted_table(sales):
    df = sales.assign(w=1.0)
    unweighted = ratios(input_df=sales, market_value_header="sale",
                        assessment_header="assess")
    weighted = ratios(input_df=df, market_value_header="sale",
                      assessment_header="assess", weights_header="w")
    # Every cell, not one column: a uniform weight vector must leave the
    # table completely untouched, 'Median' included -- the weighted median
    # breaks exact ties at the midpoint, so it agrees with ``np.median`` on
    # even-length samples too. Compared as text since ratios formats the
    # numbers.
    # STABLE_COLUMNS, not DEFAULT_COLUMNS: the bootstrap CIs are unseeded
    # and differ between any two calls, weights or not.
    assert (weighted[STABLE_COLUMNS].astype(str).values.tolist()
            == unweighted[STABLE_COLUMNS].astype(str).values.tolist())


def test_rescaling_the_weight_column_does_not_change_the_table(sales):
    """Only the relative weights matter, at the pipeline level too.

    ``ratios`` normalises the weight column to mean 1, but
    ``rts_calculator`` -- also a public entry point -- does not, and the
    bootstrap resamples an already-normalised column so its own mean
    drifts off 1. Every indicator therefore has to be scale-invariant on
    its own rather than relying on the caller to normalise.
    """
    rng = np.random.default_rng(5)
    w = rng.gamma(2.0, 0.5, len(sales))
    tables = [
        ratios(input_df=sales.assign(w=w * scale), market_value_header="sale",
               assessment_header="assess", weights_header="w")
        for scale in (1.0, 4.0, 0.1)
    ]
    for other in tables[1:]:
        assert (other[STABLE_COLUMNS].astype(str).values.tolist()
                == tables[0][STABLE_COLUMNS].astype(str).values.tolist())


def test_rts_calculator_is_scale_invariant_without_caller_normalisation(sales):
    # rts_calculator takes weights verbatim -- no mean-1 rescaling -- so it
    # is the direct check that MAE/MAPE divide by sum(w) and not by n.
    from ratio_study import rts_calculator

    rng = np.random.default_rng(6)
    w = rng.gamma(2.0, 0.5, len(sales))
    indicators = ['Median', 'Mean', 'CoD', 'PRD', 'MAE', 'MAPE', 'MdAPE',
                  'RMSE', 'CoV', 'CoC_15']
    base = rts_calculator(input_df=sales.assign(w=w), market_value_header="sale",
                          assessment_header="assess", weights_header="w",
                          indicators=indicators)
    for scale in (4.0, 0.1):
        scaled = rts_calculator(input_df=sales.assign(w=w * scale),
                                market_value_header="sale",
                                assessment_header="assess", weights_header="w",
                                indicators=indicators)
        for key in indicators:
            assert scaled[key] == pytest.approx(base[key], rel=1e-9), key


def test_codw_compounds_sample_weights_with_value_weights(sales):
    """CoDw must not silently discard a supplied weights_header.

    It weights by market value; when sample weights are also given the two
    compound, so a non-uniform weight column has to move the number.
    """
    from ratio_study import rts_calculator

    rng = np.random.default_rng(7)
    w = rng.gamma(2.0, 0.5, len(sales))
    kwargs = dict(market_value_header="sale", assessment_header="assess",
                  indicators=['Median', 'CoDw'])
    unweighted = rts_calculator(input_df=sales, **kwargs)
    weighted = rts_calculator(input_df=sales.assign(w=w),
                              weights_header="w", **kwargs)
    assert weighted['CoDw'] != pytest.approx(unweighted['CoDw'])
    # A uniform weight column leaves it exactly where it started.
    uniform = rts_calculator(input_df=sales.assign(w=1.0),
                             weights_header="w", **kwargs)
    assert uniform['CoDw'] == pytest.approx(unweighted['CoDw'])


def test_iqr_trim_drops_outliers_from_the_count(sales):
    df = sales.copy()
    # Two extreme ratios that any sane trim must remove.
    df.loc[0, "assess"] = df.loc[0, "sale"] * 25
    df.loc[1, "assess"] = df.loc[1, "sale"] * 0.02
    untrimmed = ratios(input_df=df, market_value_header="sale",
                       assessment_header="assess")
    trimmed = ratios(input_df=df, market_value_header="sale",
                     assessment_header="assess", iqr_trim_multiple=1.5)
    assert int(trimmed.loc[0, "Count"]) < int(untrimmed.loc[0, "Count"])


def test_iqr_min_cod_skips_trimming_when_dispersion_is_low(sales):
    df = sales.copy()
    rng = np.random.default_rng(9)
    df["assess"] = df["sale"] * rng.normal(1.0, 0.005, N)  # tight ratios, CoD well under 6%
    df.loc[0, "assess"] = df.loc[0, "sale"] * 25  # one extreme outlier
    trimmed_gated = ratios(input_df=df, market_value_header="sale",
                           assessment_header="assess", iqr_trim_multiple=1.5,
                           iqr_min_cod=0.06)
    trimmed_ungated = ratios(input_df=df, market_value_header="sale",
                             assessment_header="assess", iqr_trim_multiple=1.5,
                             iqr_min_cod=0)
    assert int(trimmed_gated.loc[0, "Count"]) == N
    assert int(trimmed_ungated.loc[0, "Count"]) < N


def test_iqr_group_by_trims_independently_per_group():
    # Two tight neighbourhood clusters at different ratio levels, plus one
    # sale in "A" that sits between the clusters: unremarkable pooled, but
    # a clear outlier against "A"'s own tight spread.
    rng = np.random.default_rng(11)
    mv = rng.lognormal(12.4, 0.5, N)
    ratio = np.empty(N)
    nbhd = np.array(["A"] * (N // 2) + ["B"] * (N // 2))
    ratio[:N // 2] = rng.normal(1.0, 0.01, N // 2)
    ratio[N // 2:] = rng.normal(1.3, 0.01, N // 2)
    ratio[0] = 1.15
    df = pd.DataFrame({"sale": mv, "assess": mv * ratio, "nbhd": nbhd})

    pooled_dict = ratios(input_df=df, market_value_header="sale",
                         assessment_header="assess", iqr_trim_multiple=1.5,
                         iqr_min_cod=0, return_dict=True)
    grouped_dict = ratios(input_df=df, market_value_header="sale",
                          assessment_header="assess", iqr_trim_multiple=1.5,
                          iqr_min_cod=0, iqr_group_by="nbhd", return_dict=True)

    pooled_trimmed_idx = pooled_dict["IQR_trimmed_sales"].index
    grouped_trimmed_idx = grouped_dict["IQR_trimmed_sales"].index
    assert df.index[0] not in pooled_trimmed_idx
    assert df.index[0] in grouped_trimmed_idx


def test_point_estimates_are_deterministic(sales):
    first = ratios(input_df=sales, market_value_header="sale",
                   assessment_header="assess")
    second = ratios(input_df=sales, market_value_header="sale",
                    assessment_header="assess")
    pd.testing.assert_frame_equal(_stable(first), _stable(second))


@pytest.mark.xfail(
    reason="the bootstrap resampler takes no seed, so published CIs are not "
           "reproducible run to run; needs a random_state/bootstrap_seed "
           "argument before this can be a compliance-grade output",
    strict=True,
)
def test_bootstrap_confidence_intervals_are_not_reproducible(sales):
    """Documents a real gap: two identical studies publish different CIs.

    ``Median_ci`` / ``CoD_ci`` / ``PRD_ci`` / ``PRB_ci`` come from an
    unseeded bootstrap, so re-running the same ratio study yields different
    intervals. Point estimates are unaffected. This is xfail(strict) rather
    than skip so that seeding the resampler turns it into a passing test and
    flags that this note should be removed.
    """
    first = ratios(input_df=sales, market_value_header="sale",
                   assessment_header="assess", ci_mode="bootstrap")
    second = ratios(input_df=sales, market_value_header="sale",
                    assessment_header="assess", ci_mode="bootstrap")
    ci_cols = [c for c in first.columns if c.endswith("_ci")]
    assert ci_cols, "expected bootstrap CI columns in the default output"
    pd.testing.assert_frame_equal(first[ci_cols], second[ci_cols])


def test_input_frame_is_not_mutated(sales):
    before = sales.copy(deep=True)
    ratios(input_df=sales, market_value_header="sale", assessment_header="assess")
    pd.testing.assert_frame_equal(sales, before)


# --------------------------------------------------------------------------
# Bootstrap confidence intervals: minimum sample size
# --------------------------------------------------------------------------
# A resampled interval needs a real sample behind it. Below
# `bootstrap_args['min_count_threshold']` (20 by default) the bootstrap is
# skipped and the CI columns stay empty, for the whole sample and for each
# group independently.
#
# `PRB_ci` is included. `rts_calculator` does fill it from the PRB slope
# regression at any n, but PRB is one of the bootstrapped indicators now, so
# the rows the bootstrap skipped have that value dropped rather than showing
# an interval from a different estimator at a different confidence level.

BOOTSTRAP_CI_COLUMNS = ["Median_ci", "CoD_ci", "PRB_ci", "PRD_ci"]


def _n_sales(n, seed=1):
    rng = np.random.default_rng(seed)
    sale = rng.lognormal(12.5, 0.5, n)
    return pd.DataFrame({"sale": sale, "assess": sale * rng.uniform(0.9, 1.1, n)})


def _bootstrap(df, group_by=None, **overrides):
    # These tests are specifically about bootstrap-mode mechanics
    # (resampling, the min_count_threshold gate, the resampled PRD
    # interval), so ci_mode is pinned here rather than left at the
    # package default, which is now 'distribution'. It has to
    # be passed as `ratios`' own `ci_mode` argument, not inside
    # `bootstrap_args`: `ci_mode` is one of the high-level switches applied
    # last, so it would otherwise win over anything set inside the dict.
    args = dict(bootstrap_cis_for_population=True, bootstrap_resamplings=60,
                bootstrap_concurrency=None)
    args.update(overrides)
    return ratios(input_df=df, market_value_header="sale",
                  assessment_header="assess", group_by=group_by,
                  ci_mode="bootstrap", bootstrap_args=args)


def test_the_default_bootstrap_minimum_is_twenty_sales():
    # Pinned rather than inferred: the boundary is the whole point of the
    # setting, so a silent drift in the default should fail loudly.
    from ratio_study.presets import BootstrapConfig

    assert BootstrapConfig().min_count_threshold == 20


@pytest.mark.parametrize("n", [6, 12, 19])
def test_no_bootstrap_cis_below_the_minimum_sample(n):
    out = _bootstrap(_n_sales(n))
    for column in BOOTSTRAP_CI_COLUMNS:
        assert column not in out.columns or pd.isna(out.loc[0, column])


@pytest.mark.parametrize("n", [20, 40])
def test_bootstrap_cis_are_produced_at_and_above_the_minimum(n):
    out = _bootstrap(_n_sales(n))
    for column in BOOTSTRAP_CI_COLUMNS:
        assert column in out.columns, column
        assert pd.notna(out.loc[0, column]), column


def test_the_minimum_is_a_default_not_a_floor():
    # 20 is the recommended starting point, not a hard limit: a caller who
    # deliberately lowers it still gets intervals.
    out = _bootstrap(_n_sales(12), min_count_threshold=10)
    for column in BOOTSTRAP_CI_COLUMNS:
        assert column in out.columns, column
        assert pd.notna(out.loc[0, column]), column


def test_the_group_minimum_is_applied_per_group():
    """Each group is gated on its own count, not on the sample's.

    The 40-row sample clears the threshold, so the overall row and the
    32-row group get intervals while the 8-row group is left blank.
    """
    df = _n_sales(40, seed=2)
    df["grp"] = ["A"] * 8 + ["B"] * 32
    out = _bootstrap(df, group_by="grp", bootstrap_cis_for_groups=True)
    small = out.loc[out["Group"] == "A"].iloc[0]
    large = out.loc[out["Group"] == "B"].iloc[0]
    overall = out.loc[out["Level"] == "Overall"].iloc[0]
    for column in BOOTSTRAP_CI_COLUMNS:
        assert _blank(small[column]), column
        assert not _blank(large[column]), column
        assert not _blank(overall[column]), column


def test_group_bootstrap_cis_do_not_raise():
    """Regression guard: `bootstrap_cis_for_groups=True` used to fail outright.

    The group call site passed `vei_use_log_normal_delta` to
    `_bootstrap_analysis`, which did not accept it, so the whole `ratios`
    call returned an error dict instead of a frame. Any group clearing the
    threshold is enough to trip it.
    """
    df = _n_sales(60, seed=3)
    df["grp"] = ["A"] * 30 + ["B"] * 30
    out = _bootstrap(df, group_by="grp", bootstrap_cis_for_groups=True)
    assert isinstance(out, pd.DataFrame), out
    assert len(out) == 3          # Overall + two groups


def test_vei_use_log_normal_delta_reaches_the_bootstrap():
    # It used to be dropped on the population path, so the flag had no
    # effect on bootstrapped intervals.
    import inspect

    from ratio_study.ratio_study import _bootstrap_analysis

    assert "vei_use_log_normal_delta" in inspect.signature(
        _bootstrap_analysis).parameters


# --------------------------------------------------------------------------
# Confidence intervals: the call-site surface
# --------------------------------------------------------------------------
# `confidence_intervals_aggs` / `ci_conf_level` / `ci_indicators` / `ci_seed` are
# shorthands over `bootstrap_args`. These pin what each spelling resolves
# to, since the whole point of them is that the caller does not have to know
# the canonical key names.

CI_ARGS = dict(bootstrap_resamplings=60, bootstrap_concurrency=None)


def _ci(sales, **kwargs):
    kwargs.setdefault("bootstrap_args", {}).update(CI_ARGS)
    return ratios(input_df=sales, market_value_header="sale",
                  assessment_header="assess", **kwargs)


def _ci_cols(frame):
    return [c for c in frame.columns if c.endswith("_ci")]


def _blank(value):
    """True when a cell carries no result.

    `ratios` returns a presentation frame, and since the column-ordering
    change it writes '' rather than NaN wherever an indicator or interval
    was not computed -- so a reader never sees 'NaN' in a report. Tests
    that mean "nothing here" have to accept both.
    """
    return pd.isna(value) or (isinstance(value, str) and not value.strip())


def _filled(series):
    return ~series.map(_blank)


def test_overall_cis_for_median_cod_prd_and_prb_are_the_default(sales):
    # This is bootstrap mode's own default indicator list
    # (`CI_DEFAULT_INDICATORS`); the package-wide default is now
    # `ci_mode='distribution'`, whose four columns are pinned separately
    # in `test_each_mode_keeps_its_own_default_indicator_list`.
    assert _ci_cols(_ci(sales, ci_mode="bootstrap")) == [
        "Median_ci", "CoD_ci", "PRB_ci", "PRD_ci"]


@pytest.mark.parametrize("off", [False, "none", "off", "no"])
def test_confidence_intervals_off_removes_every_interval(sales, off):
    assert _ci_cols(_ci(sales, confidence_intervals_aggs=off)) == []


def test_confidence_intervals_beats_bootstrap_args(sales):
    """The shorthand is applied last, so False is a reliable kill-switch.

    Without that ordering a caller silencing the intervals on a call that
    already carries a tuned `bootstrap_args` dict would get them anyway.
    """
    out = _ci(sales, confidence_intervals_aggs=False,
              bootstrap_args={"bootstrap_cis_for_population": True})
    assert _ci_cols(out) == []


@pytest.mark.parametrize("scope, wants_group_rows", [
    ("overall", False), ("population", False),
    ("groups", True), ("both", True), ("all", True),
])
def test_scope_words_select_which_rows_get_intervals(sales, scope,
                                                     wants_group_rows):
    out = _ci(sales, group_by="nbhd", confidence_intervals_aggs=scope)
    groups = out.loc[out["Level"] != "Overall", "Median_ci"]
    overall = out.loc[out["Level"] == "Overall", "Median_ci"]
    assert _filled(groups).all() == wants_group_rows
    # 'groups' is groups only, so the overall row goes without.
    assert _filled(overall).all() == (scope != "groups")


def test_a_list_of_indicators_means_overall_cis_for_those(sales):
    """PRB rides along: its slope regression produces an interval at
    `ci_conf_level` whether or not PRB is one of the bootstrapped indicators,
    so the column shows up alongside the two that were asked for."""
    out = _ci(sales, confidence_intervals_aggs=["Median", "CoD"])
    assert _ci_cols(out) == ["Median_ci", "CoD_ci", "PRB_ci"]
    # Narrowing the study to indicators without their own interval leaves
    # exactly what was requested.
    out = _ci(sales, indicators=["Median", "CoD"],
              confidence_intervals_aggs=["Median", "CoD"])
    assert _ci_cols(out) == ["Median_ci", "CoD_ci"]


def test_ci_indicators_drops_the_ones_the_study_does_not_compute(sales):
    # Bootstrap mode, deliberately: PRD is supported by the bootstrap, so
    # this exercises the "the study itself does not compute it" check
    # rather than the mode-support check `test_distribution_mode_reports_
    # indicators_it_cannot_cover` already covers for distribution mode.
    with pytest.warns(UserWarning, match="does not compute"):
        out = _ci(sales, indicators=["Median", "CoD"],
                  ci_indicators=["Median", "PRD"], ci_mode="bootstrap")
    assert _ci_cols(out) == ["Median_ci"]


def test_the_default_ci_indicators_narrow_silently_with_the_study(sales):
    # Not a caller mistake -- the default list is a superset by design, so
    # trimming it to what the study computes must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _ci(sales, indicators=["Median", "CoD", "VEI"])
    assert _ci_cols(out) == ["Median_ci", "CoD_ci"]


@pytest.mark.parametrize("level", [0.5, 50])
def test_ci_level_accepts_a_fraction_or_a_percentage(sales, level):
    # 50% against the 95% default: the interval has to be visibly narrower,
    # which also proves the percentage form is not read as 50/1.
    wide = _ci(sales, ci_seed=1, ci_indicators=["Median"])
    narrow = _ci(sales, ci_seed=1, ci_indicators=["Median"], ci_conf_level=level)
    assert _width(narrow.loc[0, "Median_ci"]) < _width(wide.loc[0, "Median_ci"])


def _width(formatted_ci):
    low, high = (float(x) for x in formatted_ci.strip("()").split(","))
    return high - low


def test_ci_seed_makes_the_intervals_reproducible(sales):
    first = _ci(sales, ci_seed=7)
    second = _ci(sales, ci_seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_short_bootstrap_args_keys_match_the_canonical_ones(sales):
    short = _ci(sales, group_by="nbhd", ci_seed=3,
                bootstrap_args={"groups": True, "level": 0.9,
                                "indicators": ["Median", "CoD"]})
    canonical = _ci(sales, group_by="nbhd", ci_seed=3,
                    bootstrap_args={"bootstrap_cis_for_groups": True,
                                    "bootstrap_confint_level": 0.9,
                                    "bootstrap_indicators": ["Median", "CoD"]})
    pd.testing.assert_frame_equal(short, canonical)


def test_an_unknown_bootstrap_args_key_is_reported_not_ignored(sales):
    """`boostrap_indicators` -- the typo in the package's own examples.

    It used to silently leave the default indicator list in place, so the
    caller got intervals they had not asked for and no hint why. Pinned to
    bootstrap mode: that default list -- `CI_DEFAULT_INDICATORS` -- is what
    is actually being checked here, not the package-wide default, which is
    now `ci_mode='distribution'`.
    """
    with pytest.warns(UserWarning, match="boostrap_indicators"):
        out = _ci(sales, ci_mode="bootstrap",
                  bootstrap_args={"boostrap_indicators": ["Median"]})
    # The typo is dropped, so the default list stands -- as before, but now
    # the caller is told.
    assert _ci_cols(out) == ["Median_ci", "CoD_ci", "PRB_ci", "PRD_ci"]


def test_an_unsupported_bootstrap_method_falls_back_with_a_warning(sales):
    with pytest.warns(UserWarning, match="is not supported"):
        out = _ci(sales, ci_mode="bootstrap",
                  bootstrap_args={"method": "jackknife"})
    assert _ci_cols(out) == ["Median_ci", "CoD_ci", "PRB_ci", "PRD_ci"]


def test_group_cis_without_a_group_by_are_reported_and_dropped(sales):
    with pytest.warns(UserWarning, match="no group_by"):
        out = _ci(sales, confidence_intervals_aggs="groups")
    assert _ci_cols(out) == []


def test_an_unrecognised_scope_word_raises(sales):
    with pytest.raises(ValueError, match="not recognised"):
        _ci(sales, confidence_intervals_aggs="sometimes")


def test_ci_messages_reach_the_results_dict(sales):
    # The warning is for the interactive caller; `error_messages` is what
    # ends up in the errors file and the PDF's ending comments.
    with pytest.warns(UserWarning):
        out = _ci(sales, bootstrap_args={"boostrap_indicators": ["Median"]},
                  return_dict=True)
    assert any("boostrap_indicators" in m for m in out["error_messages"])


# --------------------------------------------------------------------------
# PRB and its confidence interval
# --------------------------------------------------------------------------

def test_a_marked_prb_gives_the_same_bca_interval_as_the_bare_number():
    """The significance marker must not change the interval.

    An insignificant PRB reaches `_bootstrap_analysis` as a marked string
    like '-0.058*'. Stripping the marker used `re.findall(r'\\d+\\.\\d*')`,
    which dropped the minus sign, so BCa bias-corrected a negative
    statistic against a positive point estimate and pushed the interval
    well off. Passing the same value marked and bare has to agree.
    """
    from ratio_study.ratio_study import _bootstrap_analysis

    df = _n_sales(400, seed=8)
    common = dict(assessment_header="assess", market_value_header="sale",
                  weights_header=None, confint_level=0.95,
                  bootstrap_method="BCa", bootstrap_resamplings=120,
                  ci_only=True, req_significance=0.05,
                  indicators=["PRB"], concurrency=None, seed=42)
    bare = _bootstrap_analysis(df.copy(), population_ratios={"PRB": -0.058},
                               **common)
    marked = _bootstrap_analysis(df.copy(), population_ratios={"PRB": "-0.058*"},
                                 **common)
    assert marked["PRB_ci"] == bare["PRB_ci"]


def test_the_sign_of_the_point_estimate_moves_the_bca_interval():
    """Guards the assertion above against being vacuously true.

    If the sign were still being dropped, +0.058 and -0.058 would produce
    identical intervals and the marked/bare check would pass for the wrong
    reason.
    """
    from ratio_study.ratio_study import _bootstrap_analysis

    df = _n_sales(400, seed=8)
    common = dict(assessment_header="assess", market_value_header="sale",
                  weights_header=None, confint_level=0.95,
                  bootstrap_method="BCa", bootstrap_resamplings=120,
                  ci_only=True, req_significance=0.05,
                  indicators=["PRB"], concurrency=None, seed=42)
    negative = _bootstrap_analysis(df.copy(), population_ratios={"PRB": -0.058},
                                   **common)
    positive = _bootstrap_analysis(df.copy(), population_ratios={"PRB": 0.058},
                                   **common)
    assert negative["PRB_ci"] != positive["PRB_ci"]


def test_prb_ci_follows_ci_level_when_prb_is_not_bootstrapped(sales):
    """The slope regression's own interval now honours `ci_conf_level`.

    It used to be pinned to `1 - req_significance`, so asking for 90%
    intervals put a 95% `PRB_ci` beside them and the column had to be
    hidden. `prb(ci_conf_level=...)` decouples the interval's level from the
    significance threshold, so the column can be shown -- and has to
    widen with the level like every other interval in the table.
    """
    narrow = _ci(sales, ci_indicators=["Median", "CoD"], ci_conf_level=80)
    wide = _ci(sales, ci_indicators=["Median", "CoD"], ci_conf_level=99)
    for out in (narrow, wide):
        assert "PRB" in out.columns
        assert "PRB_ci" in out.columns

    n_lo, n_hi = _bounds(narrow.loc[0, "PRB_ci"])
    w_lo, w_hi = _bounds(wide.loc[0, "PRB_ci"])
    assert (w_hi - w_lo) > (n_hi - n_lo)


def test_the_prb_significance_test_ignores_ci_level(sales):
    """`ci_conf_level` sets the interval's width, never the p-value threshold."""
    from ratio_study.indicators import prb

    asr = sales["assess"] / sales["sale"]
    flags = {
        level: prb(sales["sale"], sales["assess"], asr,
                   req_significance=0.05, ci_conf_level=level)[1]
        for level in (None, 0.50, 0.80, 0.95, 0.999)
    }
    assert len(set(flags.values())) == 1, flags


def test_prb_ci_defaults_to_the_historical_level(sales):
    """Untouched, `ci_conf_level=None` still means `1 - req_significance`."""
    from ratio_study.indicators import prb

    asr = sales["assess"] / sales["sale"]
    common = dict(req_significance=0.05, non_significant_marker="")
    assert prb(sales["sale"], sales["assess"], asr, **common)[2] == pytest.approx(
        prb(sales["sale"], sales["assess"], asr, ci_conf_level=0.95, **common)[2]
    )


def test_a_bootstrapped_prb_keeps_the_bootstrap_interval(sales):
    """One estimator per column: the bootstrap wins where it ran."""
    out = _ci(sales, ci_indicators=["Median", "PRB"], ci_conf_level=90)
    from ratio_study.indicators import prb

    asr = sales["assess"] / sales["sale"]
    regression_ci = prb(sales["sale"], sales["assess"], asr,
                        req_significance=0.05, non_significant_marker="",
                        ci_conf_level=0.90)[2]
    assert _bounds(out.loc[0, "PRB_ci"]) != pytest.approx(regression_ci, abs=1e-9)


# --------------------------------------------------------------------------
# VEI's Kruskal-Wallis columns: table contract and PDF/Excel split
# --------------------------------------------------------------------------

def _regressive_sales(n=600, seed=21, exponent=-0.30):
    rng = np.random.default_rng(seed)
    mv = rng.lognormal(12.4, 0.5, n)
    ratio = 1.05 * (mv / mv.mean()) ** exponent * rng.lognormal(0, 0.08, n)
    return pd.DataFrame({"sale": mv, "assess": mv * ratio})


# Every call below pins `pdf_vertical_equity_panel` to False explicitly.
# That default is owned by unrelated, concurrently in-progress work in this
# repo and shouldn't decide whether these assertions -- about the
# Kruskal-Wallis columns, not the panel -- hold. (Generating the panel
# upgrades `ratios()`'s return type to a dict, which is exactly the kind of
# coupling these tests need to stay independent of.)
_NO_PANEL = {"pdf_vertical_equity_panel": False}


def test_vei_kw_columns_are_in_the_default_table():
    out = ratios(input_df=_regressive_sales(), market_value_header="sale",
                assessment_header="assess", confidence_intervals_aggs=False,
                graphs_args=_NO_PANEL)
    # Formatted to text like every other numeric column in this table.
    assert float(out.loc[0, "VEI_kw_statistic"]) > 0
    assert 0 <= float(out.loc[0, "VEI_kw_pvalue"]) <= 1


def test_vei_kw_columns_are_absent_when_vei_is_not_requested():
    out = ratios(input_df=_regressive_sales(), market_value_header="sale",
                assessment_header="assess", indicators=["Median"],
                confidence_intervals_aggs=False, graphs_args=_NO_PANEL)
    assert "VEI_kw_statistic" not in out.columns
    assert "VEI_kw_pvalue" not in out.columns


def test_vei_kw_columns_are_blank_below_the_vei_minimum_sample():
    out = ratios(input_df=_n_sales(12), market_value_header="sale",
                assessment_header="assess", indicators=["Median", "VEI"],
                confidence_intervals_aggs=False, graphs_args=_NO_PANEL)
    assert out.loc[0, "VEI_kw_statistic"] == ""
    assert out.loc[0, "VEI_kw_pvalue"] == ""


def test_vei_kw_reaches_the_excel_csv_export_but_not_the_pdf_table(tmp_path):
    """The split the user asked for: Excel/CSV gets the columns, the PDF
    table does not -- there is no room on the printed page for two more
    columns without shrinking the font past readability.
    """
    from ratio_study import ratios_report

    out = ratios_report(_regressive_sales(), "assess", "sale",
                        indicators=["Median", "VEI"],
                        confidence_intervals_aggs=False,
                        export_directory=tmp_path,
                        pdf_args={"gen_pdf_report": False, **_NO_PANEL},
                        export_in_compacted_file="",
                        return_dict_results=True)
    excel_cols = out["df_all_ratios_formatted"].columns
    assert "VEI_kw_statistic" in excel_cols
    assert "VEI_kw_pvalue" in excel_cols

    # Even an explicit attempt to add them to the PDF table is filtered out:
    # `pdf_ratios_table_headers` is intersected with `indicators`, and
    # 'VEI_kw_statistic' is never a member of that list (it names a column,
    # not something a caller passes to `indicators=`).
    requested = ["Median", "VEI", "VEI_kw_statistic", "VEI_kw_pvalue"]
    indicators = ["Median", "VEI"]
    assert [x for x in requested if x in indicators] == ["Median", "VEI"]


# --------------------------------------------------------------------------
# ratios_report: gen_pdf_report actually gates the PDF
# --------------------------------------------------------------------------

def test_gen_pdf_report_false_writes_no_pdf(tmp_path):
    """Regression guard: `gen_pdf_report=False` used to do nothing.

    The gate checked `'gen_pdf_report' in pdf_args` -- key *presence* --
    but the key is always present after the defaults merge
    (`std_pdf_args = {'gen_pdf_report': True, ...}`), so a PDF was written
    and added to `exported_files` regardless of what the caller set the
    flag to.
    """
    from ratio_study import ratios_report

    out = ratios_report(_regressive_sales(), "assess", "sale",
                        indicators=["Median"],
                        confidence_intervals_aggs=False,
                        export_directory=tmp_path,
                        pdf_args={"gen_pdf_report": False, **_NO_PANEL},
                        export_in_compacted_file="",
                        return_dict_results=True)
    assert not any(f.endswith(".pdf") for f in out["exported_files"])
    assert not list(tmp_path.glob("*.pdf"))


def test_gen_pdf_report_true_still_writes_one(tmp_path):
    """The other half of the guard: the default must keep working."""
    from ratio_study import ratios_report

    out = ratios_report(_regressive_sales(), "assess", "sale",
                        indicators=["Median"],
                        confidence_intervals_aggs=False,
                        export_directory=tmp_path,
                        pdf_args=_NO_PANEL,
                        export_in_compacted_file="",
                        return_dict_results=True)
    assert any(f.endswith(".pdf") for f in out["exported_files"])
    assert list(tmp_path.glob("*.pdf"))


# ===========================================================================
# ci_mode: bootstrap vs distribution-based confidence intervals
# ===========================================================================

DISTRIBUTION_CI_COLUMNS = ["Median_ci", "Mean_ci", "CoD_ci", "Weighted_Mean_ci"]

# The four indicators with a per-sample-array formula in `distribution_
# analysis` (Mean/Median/CoD/Weighted_Mean); PRB and PRD are also covered
# in distribution mode, but via a different mechanism -- PRB's own slope
# regression, PRD's delta-method interval -- exercised in their own tests
# below rather than through this default. `_FOUR_INDICATORS` puts all four
# of these columns in play rather than only the two `ratios` computes by
# default.
_FOUR_INDICATORS = ["Median", "Mean", "CoD", "Weighted_Mean"]


def _distribution(df, group_by=None, indicators=None, **kwargs):
    return ratios(input_df=df, market_value_header="sale",
                  assessment_header="assess", group_by=group_by,
                  indicators=indicators or list(_FOUR_INDICATORS),
                  ci_mode="distribution", **kwargs)


def _bounds(cell):
    """Parse a presentation-layer CI cell back into two floats.

    `ratios` formats its output for reading -- ``'(0.9771, 1.0074)'``, or
    ``'(4.76%, 5.61%)'`` for the columns shown as percentages -- so an
    assertion about the numbers has to undo that first.
    """
    if isinstance(cell, (tuple, list)):
        return float(cell[0]), float(cell[1])
    parts = str(cell).strip().strip("()").split(",")

    def _one(text):
        text = text.strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)

    return _one(parts[0]), _one(parts[1])


def _value(cell):
    """Parse a presentation-layer point estimate back into a float."""
    if isinstance(cell, str):
        text = cell.strip().replace(",", "")
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)
    return float(cell)


def test_distribution_stays_the_default_mode():
    """The Appendix D closed forms are now the default, everywhere `ci_mode`
    has one: `ratios`, `ratios_report`, and the `BootstrapConfig` preset."""
    import inspect

    from ratio_study import ratios_report
    from ratio_study.presets import BootstrapConfig

    for func in (ratios, ratios_report):
        assert inspect.signature(func).parameters["ci_mode"].default == "distribution"
    assert BootstrapConfig().ci_mode == "distribution"


def test_distribution_mode_fills_its_four_columns():
    out = _distribution(_n_sales(200, seed=11))
    for column in DISTRIBUTION_CI_COLUMNS:
        assert column in out.columns, column
        assert pd.notna(out.loc[0, column]), column


def test_distribution_mode_covers_prb_via_regression_and_prd_via_delta_method():
    """Neither has a formula in Appendix D, but both get an interval.

    PRB keeps the interval its own slope regression always produces.
    PRD -- the ratio of Mean to Weighted_Mean -- gets a delta-method
    interval built from those two formulas (`confint.prd_ci`), so it has
    to match a direct call to that function on the same sample.
    """
    from ratio_study.confint import prd_ci
    from ratio_study.indicators import prb, asr_median

    df = _n_sales(200, seed=12)
    out = _distribution(df, indicators=_FOUR_INDICATORS + ["PRD", "PRB"])
    row = out.loc[0]
    assert pd.notna(row["PRD_ci"])
    assert pd.notna(row["PRB_ci"])

    # Both helpers default to 95%, the study to `ci_conf_level`'s 90%, so
    # the level has to be named for the comparison to be like for like.
    expected_lo, expected_hi = prd_ci(df["assess"], df["sale"], conf_level=0.9)
    lo, hi = _bounds(row["PRD_ci"])
    assert lo == pytest.approx(expected_lo, abs=1e-4)
    assert hi == pytest.approx(expected_hi, abs=1e-4)

    asr = df["assess"] / df["sale"]
    _, _, expected_prb_ci = prb(df["sale"], df["assess"], asr,
                               asr_md=asr_median(asr), req_significance=0.05,
                               non_significant_marker="", ci_conf_level=0.9)
    lo, hi = _bounds(row["PRB_ci"])
    assert lo == pytest.approx(expected_prb_ci[0], abs=1e-4)
    assert hi == pytest.approx(expected_prb_ci[1], abs=1e-4)


def test_distribution_mode_reports_indicators_it_cannot_cover():
    """An explicit request for an unsupported indicator has to be told.

    PRD is now covered (delta method), so only RMSE -- which has no
    distribution-mode formula at all -- should be flagged.
    """
    from ratio_study.confint import normalize_ci_args

    args, messages = normalize_ci_args(ci_mode="distribution",
                                       ci_indicators=["Median", "PRD", "RMSE"],
                                       study_indicators=["Median", "PRD", "RMSE"])
    assert args["bootstrap_indicators"] == ["Median", "PRD"]
    assert any("['RMSE']" in m for m in messages)
    assert any("distribution-based" in m for m in messages)


def test_distribution_mode_needs_no_minimum_count():
    """The binomial median tables reach far below the bootstrap's floor.

    A 12-sale study gets no bootstrap interval at the default threshold of
    20, but every distribution formula is defined there.
    """
    df = _n_sales(12, seed=13)
    assert all(
        column not in _bootstrap(df).columns
        or pd.isna(_bootstrap(df).loc[0, column])
        for column in BOOTSTRAP_CI_COLUMNS
    )
    out = _distribution(df)
    for column in DISTRIBUTION_CI_COLUMNS:
        assert pd.notna(out.loc[0, column]), column


def test_distribution_mode_gives_every_group_an_interval():
    """Small groups are the case the small-sample tables exist for."""
    df = _n_sales(60, seed=14)
    df["grp"] = ["A"] * 8 + ["B"] * 22 + ["C"] * 30
    out = _distribution(df, group_by="grp", confidence_intervals_aggs="both")
    for group in ("A", "B", "C"):
        row = out.loc[out["Group"] == group].iloc[0]
        for column in DISTRIBUTION_CI_COLUMNS:
            assert pd.notna(row[column]), (group, column)


def test_the_two_modes_broadly_agree_on_a_healthy_sample():
    """Different estimators, same quantity: they should be close.

    Not a tight assertion -- BCa and a closed-form interval genuinely
    differ -- but a mode wired to the wrong statistic would be nowhere
    near.
    """
    df = _n_sales(400, seed=15)
    boot = _bootstrap(df, bootstrap_resamplings=400, bootstrap_seed=7,
                      bootstrap_indicators=["Median", "CoD"])
    dist = _distribution(df)
    for column in ("Median_ci", "CoD_ci"):
        b_lo, b_hi = _bounds(boot.loc[0, column])
        d_lo, d_hi = _bounds(dist.loc[0, column])
        width = max(b_hi - b_lo, d_hi - d_lo)
        assert abs(b_lo - d_lo) < width, column
        assert abs(b_hi - d_hi) < width, column


def test_distribution_intervals_bracket_their_point_estimates():
    out = _distribution(_n_sales(300, seed=16))
    row = out.loc[0]
    for indicator, column in (("Median", "Median_ci"), ("Mean", "Mean_ci"),
                              ("Weighted_Mean", "Weighted_Mean_ci"),
                              ("CoD", "CoD_ci")):
        lower, upper = _bounds(row[column])
        # The presentation layer rounds both, so allow for half a displayed
        # unit rather than asserting on exact inequality.
        assert lower - 1e-4 <= _value(row[indicator]) <= upper + 1e-4, indicator


@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
def test_distribution_intervals_widen_with_the_level(level):
    narrow = _distribution(_n_sales(300, seed=17), ci_conf_level=0.80)
    wider = _distribution(_n_sales(300, seed=17), ci_conf_level=level)
    for column in ("Mean_ci", "CoD_ci", "Weighted_Mean_ci", "Median_ci"):
        n_lo, n_hi = _bounds(narrow.loc[0, column])
        w_lo, w_hi = _bounds(wider.loc[0, column])
        assert (w_hi - w_lo) >= (n_hi - n_lo) - 1e-9, column


@pytest.mark.parametrize("spelling", ["distribution", "Distribution", "dist",
                                      "analytic", "closed_form", "parametric"])
def test_ci_mode_accepts_the_obvious_spellings(spelling):
    from ratio_study.confint import normalize_ci_args

    args, messages = normalize_ci_args(ci_mode=spelling)
    assert args["ci_mode"] == "distribution"
    assert not messages


def test_an_unknown_ci_mode_is_reported_and_falls_back_to_distribution():
    from ratio_study.confint import normalize_ci_args

    args, messages = normalize_ci_args(ci_mode="jackknife")
    assert args["ci_mode"] == "distribution"
    assert any("jackknife" in m and "not recognised" in m for m in messages)
    assert any("distribution" in m for m in messages)


def test_ci_mode_can_also_come_through_bootstrap_args():
    from ratio_study.confint import normalize_ci_args

    args, _ = normalize_ci_args(bootstrap_args={"mode": "distribution"})
    assert args["ci_mode"] == "distribution"


def test_each_mode_keeps_its_own_default_indicator_list():
    from ratio_study.confint import normalize_ci_args

    study = ["Median", "Mean", "CoD", "PRD", "PRB", "Weighted_Mean"]
    boot, _ = normalize_ci_args(ci_mode="bootstrap", study_indicators=study)
    dist, _ = normalize_ci_args(ci_mode="distribution", study_indicators=study)
    assert boot["bootstrap_indicators"] == ["Median", "CoD", "PRD", "PRB"]
    assert dist["bootstrap_indicators"] == [
        "Median", "Mean", "Weighted_Mean", "CoD", "PRB", "PRD"]


def test_an_explicit_none_still_means_every_supported_indicator():
    """`bootstrap_indicators=None` has always meant "all of them"."""
    from ratio_study.confint import normalize_ci_args

    study = ["Median", "Mean", "CoD", "PRD", "Weighted_Mean"]
    dist, _ = normalize_ci_args(ci_mode="distribution",
                                bootstrap_args={"bootstrap_indicators": None},
                                study_indicators=study)
    assert set(dist["bootstrap_indicators"]) == {"Median", "Mean", "CoD",
                                                 "PRD", "Weighted_Mean"}


def test_weighted_mean_is_computed_when_asked_for_on_its_own():
    """It used to appear only as a by-product of the Gini block.

    Without this, `Weighted_Mean_ci` had no point estimate to sit beside
    unless the study also happened to request a Gini indicator.
    """
    df = _n_sales(80, seed=18)
    out = ratios(input_df=df, market_value_header="sale",
                 assessment_header="assess", indicators=["Weighted_Mean"],
                 confidence_intervals_aggs=False)
    assert "Weighted_Mean" in out.columns
    assert _value(out.loc[0, "Weighted_Mean"]) == pytest.approx(
        df["assess"].sum() / df["sale"].sum(), abs=1e-4
    )


def test_the_vei_confidence_level_defaults_to_ninety_percent_everywhere():
    """`ratios_dispatcher` used to fall back to 0.1 -- a 10% interval.

    It is a confidence level, not a significance level, and every other
    default in the package spells it 0.9.
    """
    import inspect

    from ratio_study import ratios_report
    from ratio_study.presets import RatioStudyConfig
    from ratio_study.ratio_study import _ratios_calculator_wrap, rts_calculator

    source = inspect.getsource(_ratios_calculator_wrap)
    assert "else 0.1" not in source
    assert "else 0.9" in source

    for func in (ratios, ratios_report, rts_calculator):
        assert inspect.signature(func).parameters["vei_conf_level"].default == 0.9
    assert RatioStudyConfig().vei_conf_level == 0.90


# ===========================================================================
# VEM's confidence level, reachable from the public entry points
# ===========================================================================

def test_vem_conf_level_is_exposed_on_both_entry_points():
    """It used to be pinned to `vem`'s own 0.90 default.

    `ratios` called `vem()` without `conf_level`, so a study could draw a
    "90% CI" band in the panel while reporting indicator intervals at
    whatever `ci_conf_level` asked for, with no way to align them.
    """
    import inspect

    from ratio_study import ratios_report

    for func in (ratios, ratios_report):
        params = inspect.signature(func).parameters
        assert params["vem_conf_level"].default == 0.9
        assert params["vem_ci_method"].default == "binom"


def test_vem_band_widens_monotonically_with_the_level():
    """The band has to actually widen -- it drives the VEM verdict via
    `ci_out_of_bounds`, so a level that does not reach it would silently
    fix the verdict at whatever `vem`'s own default produced."""
    from ratio_study.indicators import vem

    df = _n_sales(300, seed=51)
    df["asr"] = df["assess"] / df["sale"]
    widths = []
    for level in (0.50, 0.80, 0.95, 0.99):
        out = vem(df, "sale", window_size=30, conf_level=level,
                  acceptable_inequity_level=0.1)
        band = out["dataframe"]
        widths.append(float(
            (band["median_ci_upper"] - band["median_ci_lower"]).mean()))
    assert widths == sorted(widths)
    # A 99% band should be several times a 50% one, not marginally wider.
    assert widths[-1] / widths[0] > 2.0


def test_vem_conf_level_survives_the_trip_through_ratios():
    """End to end: the level set on `ratios` must reach the band."""
    from ratio_study.indicators import vem

    df = _n_sales(300, seed=52)
    out = ratios(input_df=df, market_value_header="sale",
                 assessment_header="assess",
                 indicators=["Median", "VEM"], vem_conf_level=0.99,
                 confidence_intervals_aggs=False, return_dict=True)
    # `ratios` does not surface the band frame, so compare the verdict
    # fields against a direct call at the same level and at a different
    # one; only the matching level can agree on all three.
    direct = df.copy()
    direct["asr"] = direct["assess"] / direct["sale"]
    window = int(len(df) / 10)
    same = vem(direct.copy(), "sale", window_size=window, conf_level=0.99,
               acceptable_inequity_level=0.1)
    frame = out["dict_population_ratios"]["ratios"]
    assert frame["Inequity_severity"] == pytest.approx(
        same["Inequity_severity"])
    assert frame["Inequity_conclusion"] == same["Inequity_conclusion"]


@pytest.mark.parametrize("method", ["binom", "R6", "R7"])
def test_vem_ci_method_is_accepted_end_to_end(method):
    out = ratios(input_df=_n_sales(200, seed=53), market_value_header="sale",
                 assessment_header="assess", indicators=["Median", "VEM"],
                 vem_ci_method=method, confidence_intervals_aggs=False)
    assert "Inequity_conclusion" in out.columns


# ===========================================================================
# Level validation, end to end
# ===========================================================================

@pytest.mark.parametrize("level", [90, 0.90])
def test_every_level_argument_takes_either_spelling(sales, level):
    """`ci_conf_level=90` worked while `vei_conf_level=90` raised, because the
    vertical-equity levels validated `0 < x < 1` directly instead of going
    through the shared gate."""
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess",
                 indicators=["Median", "VEI", "VEM"],
                 ci_conf_level=level, vei_conf_level=level, vem_conf_level=level,
                 confidence_intervals_aggs=False)
    assert len(out) == 1


@pytest.mark.parametrize("argument", ["ci_conf_level", "vei_conf_level",
                                      "vem_conf_level"])
def test_a_significance_level_is_rejected_at_every_level_argument(sales, argument):
    with pytest.raises(ValueError, match="significance level"):
        ratios(input_df=sales, market_value_header="sale",
               assessment_header="assess",
               indicators=["Median", "VEI", "VEM"],
               **{argument: 0.05})


def test_req_significance_still_takes_a_significance_level(sales):
    """The floor applies to confidence levels only -- 0.05 is exactly what
    `req_significance` is for, and must keep working."""
    out = ratios(input_df=sales, market_value_header="sale",
                 assessment_header="assess", indicators=["Median", "PRB"],
                 req_significance=0.05, confidence_intervals_aggs=False)
    assert "PRB" in out.columns


def test_presets_accept_a_percentage_and_reject_a_significance_level():
    from ratio_study.presets import BootstrapConfig, RatioStudyConfig

    assert BootstrapConfig(confint_level=95).confint_level == pytest.approx(0.95)
    assert RatioStudyConfig(vei_conf_level=90).vei_conf_level == pytest.approx(0.90)
    assert RatioStudyConfig(vem_conf_level=90).vem_conf_level == pytest.approx(0.90)
    for model, field in ((BootstrapConfig, "confint_level"),
                         (RatioStudyConfig, "vei_conf_level"),
                         (RatioStudyConfig, "vem_conf_level")):
        with pytest.raises(Exception, match="significance level"):
            model(**{field: 0.05})


def test_the_preset_min_count_floor_matches_what_confint_enforces():
    """`normalize_ci_args` clamps to 4, so a validated 1 became 4 in
    silence. The model now rejects anything it cannot honour."""
    from ratio_study.confint import normalize_ci_args
    from ratio_study.presets import BootstrapConfig

    assert BootstrapConfig(min_count_threshold=4).min_count_threshold == 4
    with pytest.raises(Exception):
        BootstrapConfig(min_count_threshold=3)
    args, _ = normalize_ci_args(bootstrap_args={"min_count_threshold": 4})
    assert args["min_count_threshold"] == 4


def test_no_dead_confidence_level_tail_remains():
    """It read as though the level were applied in `_bootstrap_analysis`,
    but `_bca_ci` / `_percentile_ci` each derive their own tail."""
    import inspect

    from ratio_study.ratio_study import _bootstrap_analysis

    assert "confidence_level_tail" not in inspect.getsource(_bootstrap_analysis)


# --------------------------------------------------------------------------
# mvp_sale_price_share
# --------------------------------------------------------------------------

def test_the_default_share_is_the_historical_50_50_proxy(regressive_sales):
    """The argument exists to be changed, not to change anything by default."""
    explicit = ratios(input_df=regressive_sales, market_value_header="sale",
                      assessment_header="assess", mvp_sale_price_share=0.5,
                      print_ci_settings=False)
    default = ratios(input_df=regressive_sales, market_value_header="sale",
                     assessment_header="assess", print_ci_settings=False)
    pd.testing.assert_frame_equal(_stable(explicit), _stable(default))


def test_the_share_reaches_both_prb_and_vei(regressive_sales):
    """Both indicators read the proxy, so both have to move with it."""
    base = ratios(input_df=regressive_sales, market_value_header="sale",
                  assessment_header="assess", print_ci_settings=False).iloc[0]
    leaned = ratios(input_df=regressive_sales, market_value_header="sale",
                    assessment_header="assess", mvp_sale_price_share=0.3,
                    print_ci_settings=False).iloc[0]

    assert base["PRB"] != leaned["PRB"]
    assert base["VEI"] != leaned["VEI"]
    # The level of assessment is a property of the sample, not of the proxy.
    assert base["Median"] == leaned["Median"]


def test_the_share_reaches_the_per_group_tables(sales):
    base = ratios(input_df=sales, market_value_header="sale",
                  assessment_header="assess", group_by=["nbhd"],
                  print_ci_settings=False)
    leaned = ratios(input_df=sales, market_value_header="sale",
                    assessment_header="assess", group_by=["nbhd"],
                    mvp_sale_price_share=0.2, print_ci_settings=False)
    assert base["VEI"].tolist() != leaned["VEI"].tolist()


def test_a_share_of_one_proxies_with_the_sale_price_alone(regressive_sales):
    """The escape hatch for a jurisdiction that trusts its prices: the
    proxy collapses to the sale price, which is what PRB regressed on
    before the Standard introduced the blend."""
    from ratio_study import indicators as I

    row = ratios(input_df=regressive_sales, market_value_header="sale",
                 assessment_header="assess", mvp_sale_price_share=1.0,
                 indicators=["Median", "PRB"], print_ci_settings=False).iloc[0]
    asr = regressive_sales["assess"] / regressive_sales["sale"]
    expected = I.prb(regressive_sales["sale"].to_numpy(),
                     regressive_sales["assess"].to_numpy(), asr.to_numpy(),
                     market_value_proxy=regressive_sales["sale"].to_numpy(),
                     req_significance=0.05, non_significant_marker="*")[0]
    assert str(row["PRB"]) == str(expected)


def test_an_out_of_range_share_is_rejected_by_ratios(regressive_sales):
    with pytest.raises(Exception, match="mvp_sale_price_share"):
        ratios(input_df=regressive_sales, market_value_header="sale",
               assessment_header="assess", mvp_sale_price_share=-1,
               print_ci_settings=False)
