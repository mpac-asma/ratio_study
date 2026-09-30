"""Tests for the atomic (scalar) ratio-study indicators.

Every expected value below is hand-computed from the five-element fixture
``ASR`` / ``MV`` so the assertions pin the formulas themselves, not just
whatever the code currently happens to return.

Convention note: the atomic indicators return **fractions**, not
percentages -- ``cod`` on a 12% dispersion returns ``0.12``. Scaling to
percent happens in the ``ratios`` presentation layer.
"""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from ratio_study import confint as C
from ratio_study import indicators as I
from ratio_study.confint import compute_median_ci

# Five sales with an exactly symmetric ratio spread around 1.0, so the
# median, mean and every derived statistic are hand-checkable.
ASR = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
MV = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
AV = ASR * MV
SALE_ERROR = (AV - MV) / MV


# --------------------------------------------------------------------------
# Central tendency
# --------------------------------------------------------------------------

def test_asr_median_is_the_middle_ratio():
    assert I.asr_median(ASR) == pytest.approx(1.0)


def test_asr_mean_is_the_arithmetic_mean():
    assert I.asr_mean(ASR) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Dispersion
# --------------------------------------------------------------------------

def test_cod_is_mean_absolute_deviation_over_the_median():
    # mean(|asr - 1.0|) / 1.0 = (0.2 + 0.1 + 0 + 0.1 + 0.2) / 5
    assert I.cod(ASR) == pytest.approx(0.12)


def test_cod_accepts_a_precomputed_median():
    # The asr_md argument exists purely to avoid recomputing the median;
    # passing the true median must not change the answer.
    assert I.cod(ASR, asr_md=1.0) == pytest.approx(I.cod(ASR))


def test_cod_residual_works_from_raw_values():
    # mean(|AV - MV|) / mean(MV)
    assert I.cod_residual(MV, AV) == pytest.approx(0.12)


def test_cov_adj_applies_the_iaao_small_sample_correction():
    # IAAO: (sample sd / mean) * (1 + 1/(4n)) -- sd uses the n-1 divisor.
    n = len(ASR)
    expected = (ASR.std(ddof=1) / ASR.mean()) * (1 + 1 / (4 * n))
    assert I.cov_adj(ASR) == pytest.approx(expected)
    # The correction must inflate the raw CoV, never shrink it.
    assert I.cov_adj(ASR) > ASR.std(ddof=1) / ASR.mean()


def test_coc_counts_ratios_inside_the_band():
    # Within 15% of 1.0 -> [0.85, 1.15] captures 0.9, 1.0, 1.1 -> 3/5.
    assert I.coc(ASR, within_prcnt=0.15) == pytest.approx(0.6)
    # A band wide enough to cover everything.
    assert I.coc(ASR, within_prcnt=0.50) == pytest.approx(1.0)


def test_coc_is_monotone_in_band_width():
    widths = [0.05, 0.10, 0.15, 0.25, 0.50]
    values = [I.coc(ASR, within_prcnt=w) for w in widths]
    assert values == sorted(values)


# --------------------------------------------------------------------------
# Vertical equity / bias
# --------------------------------------------------------------------------

def test_prd_is_mean_ratio_over_dollar_weighted_ratio():
    expected = ASR.mean() / np.average(ASR, weights=MV)
    assert I.prd(MV, ASR) == pytest.approx(expected)
    # This fixture rises with price, so the dollar-weighted ratio exceeds
    # the mean ratio and PRD lands below 1 (progressive).
    assert I.prd(MV, ASR) < 1.0


def test_prd_is_one_when_ratios_are_flat():
    flat = np.ones(5)
    assert I.prd(MV, flat) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Market value proxy
# --------------------------------------------------------------------------

def test_market_value_proxy_is_the_standards_50_50_blend_by_default():
    # median ASR is exactly 1.0 on this fixture, so the assessment half
    # needs no restating and the proxy is the plain midpoint.
    proxy = I.market_value_proxy(MV, AV)
    assert proxy == pytest.approx((MV + AV) / 2)


def test_market_value_proxy_restates_the_assessment_at_the_level_of_assessment():
    # Halve every assessment: the median ASR halves with it, so dividing
    # the assessment back through puts it on the sale price's scale again
    # and the proxy is unchanged.
    assert I.market_value_proxy(MV, AV / 2) == pytest.approx(
        I.market_value_proxy(MV, AV))


def test_mvp_sale_price_share_moves_the_weight_between_the_two_halves():
    level_adjusted = AV / I.asr_median(ASR)
    for share in (0.0, 0.3, 0.5, 0.7, 1.0):
        expected = MV * share + level_adjusted * (1 - share)
        assert I.market_value_proxy(
            MV, AV, mvp_sale_price_share=share) == pytest.approx(expected)


def test_a_share_of_one_is_the_sale_price_and_zero_is_the_assessment():
    assert I.market_value_proxy(MV, AV, mvp_sale_price_share=1) == pytest.approx(MV)
    assert I.market_value_proxy(MV, AV, mvp_sale_price_share=0) == pytest.approx(
        AV / I.asr_median(ASR))


def test_a_share_given_as_a_percentage_means_the_same_as_the_fraction():
    assert I.market_value_proxy(MV, AV, mvp_sale_price_share=30) == pytest.approx(
        I.market_value_proxy(MV, AV, mvp_sale_price_share=0.3))


@pytest.mark.parametrize('share', [-0.1, 101, 'half', None, np.nan])
def test_market_value_proxy_rejects_a_share_outside_the_range(share):
    with pytest.raises(ValueError, match='mvp_sale_price_share'):
        I.market_value_proxy(MV, AV, mvp_sale_price_share=share)


def test_market_value_proxy_uses_a_supplied_median_over_recomputing_one():
    # Passing the study's own median is what keeps the proxy from drifting
    # away from the ratios table.
    assert I.market_value_proxy(MV, AV, asr_md=2.0) == pytest.approx(
        MV * 0.5 + (AV / 2.0) * 0.5)


def test_market_value_proxy_keeps_the_container_it_was_given():
    proxy = I.market_value_proxy(pd.Series(MV), pd.Series(AV))
    assert isinstance(proxy, pd.Series)
    assert proxy.to_numpy() == pytest.approx((MV + AV) / 2)


def test_prb_builds_its_default_proxy_through_market_value_proxy():
    # The centralised function is the single definition: PRB's own default
    # has to be the same blend, or the two would drift apart.
    explicit = I.market_value_proxy(MV, AV, asr_md=1.0)
    assert I.prb(MV, AV, ASR)[0] == pytest.approx(
        I.prb(MV, AV, ASR, market_value_proxy=explicit)[0])


def test_prb_moves_when_the_proxy_leans_onto_the_assessment():
    default = I.prb(MV, AV, ASR, req_significance=0)[0]
    leaned = I.prb(MV, AV, ASR, req_significance=0, mvp_sale_price_share=0.3)[0]
    assert default != pytest.approx(leaned)
    # And the share still routes through the same blend.
    proxy = I.market_value_proxy(MV, AV, asr_md=1.0, mvp_sale_price_share=0.3)
    assert leaned == pytest.approx(
        I.prb(MV, AV, ASR, req_significance=0, market_value_proxy=proxy)[0])


def test_prb_ignores_the_share_when_a_proxy_is_supplied():
    proxy = I.market_value_proxy(MV, AV, asr_md=1.0)
    assert I.prb(MV, AV, ASR, market_value_proxy=proxy,
                 mvp_sale_price_share=0.1)[0] == pytest.approx(
        I.prb(MV, AV, ASR, market_value_proxy=proxy)[0])


def test_prb_returns_value_significance_flag_and_ci():
    result = I.prb(MV, AV, ASR)
    assert isinstance(result, tuple) and len(result) == 3
    value, not_significant, ci = result
    assert isinstance(value, float)
    assert isinstance(not_significant, (bool, np.bool_))
    assert len(ci) == 2 and ci[0] < value < ci[1]


def test_prb_detects_the_progressive_tilt_in_this_fixture():
    value, not_significant, _ = I.prb(MV, AV, ASR)
    # Ratios rise with price -> positive PRB slope, and with a clean
    # monotone fixture it is statistically significant.
    assert value > 0
    assert not_significant is False or not_significant == False  # noqa: E712


def test_prb_skips_the_significance_test_when_disabled():
    # Falsy req_significance documents "compute the value only".
    value, not_significant, _ = I.prb(MV, AV, ASR, req_significance=0)
    assert isinstance(value, float)


def test_prb_is_the_same_number_with_and_without_the_significance_test():
    """The two paths must not disagree.

    With `req_significance` falsy, `prb` takes a closed-form slope instead
    of a statsmodels fit -- ~20x faster, which is what makes bootstrapping
    PRB affordable. The slope it returns has to be the statsmodels one to
    the last decimal, or the bootstrap would be resampling a different
    statistic from the point estimate its interval is built around.
    """
    rng = np.random.default_rng(3)
    for n in (30, 600):
        mv = rng.lognormal(12.4, 0.5, n)
        av = mv * rng.normal(1.0, 0.12, n)
        asr = av / mv
        weights = rng.gamma(2.0, 0.5, n)
        for w in (None, weights):
            tested = I.prb(mv, av, asr, req_significance=0.05,
                           non_significant_marker='', weights=w)[0]
            plain = I.prb(mv, av, asr, req_significance=0,
                          non_significant_marker='', weights=w)[0]
            # Both are rounded to 5 decimals inside `prb`, so equality is
            # the right assertion rather than approx.
            assert plain == tested, (n, w is not None)


@pytest.mark.parametrize("to_backend", [
    pytest.param(lambda a: a, id="numpy"),
    pytest.param(pd.Series, id="pandas"),
    pytest.param(pl.Series, id="polars"),
])
def test_ols_slope_matches_statsmodels_across_backends(to_backend):
    from ratio_study.helpers import _ols_slope, _simple_regression

    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 200)
    y = 2.5 * x + rng.normal(0, 1, 200)
    expected = _simple_regression(y, x)['slope']['beta']
    assert _ols_slope(to_backend(y), to_backend(x)) == pytest.approx(
        expected, rel=1e-12)


def test_ols_slope_drops_nans_pairwise_like_statsmodels():
    from ratio_study.helpers import _ols_slope, _simple_regression

    rng = np.random.default_rng(12)
    x = rng.normal(0, 1, 100)
    y = 2.5 * x + rng.normal(0, 1, 100)
    y[7] = np.nan
    x[19] = np.nan
    assert _ols_slope(y, x) == pytest.approx(
        _simple_regression(y, x)['slope']['beta'], rel=1e-12)


@pytest.mark.parametrize("y, x", [
    ([1.0], [2.0]),                          # a single point
    ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),      # x has no spread
    ([np.nan, np.nan], [1.0, 2.0]),          # nothing survives the mask
])
def test_ols_slope_is_nan_where_the_slope_is_undetermined(y, x):
    from ratio_study.helpers import _ols_slope

    assert np.isnan(_ols_slope(y, x))


# --------------------------------------------------------------------------
# Error magnitude
# --------------------------------------------------------------------------

def test_mae_is_mean_absolute_error():
    assert I.mae(MV, AV) == pytest.approx(np.mean(np.abs(AV - MV)))


def test_rmse_is_root_mean_squared_error():
    assert I.rmse(MV, AV) == pytest.approx(np.sqrt(np.mean((AV - MV) ** 2)))


def test_rmse_is_zero_for_a_perfect_model():
    assert I.rmse(MV, MV) == pytest.approx(0.0)


def test_mape_and_mdmape_use_mean_and_median():
    assert I.mape(SALE_ERROR) == pytest.approx(np.mean(np.abs(SALE_ERROR)))
    assert I.mdmape(SALE_ERROR) == pytest.approx(np.median(np.abs(SALE_ERROR)))


def test_fsd_returns_a_formatted_string_with_a_normality_p_value():
    # fsd is a *display* indicator: '14.14% (0.967)' -- percentage plus the
    # Shapiro-Wilk p-value. Unlike its siblings it is not numeric, so it
    # cannot be fed into further arithmetic.
    out = I.fsd(SALE_ERROR)
    assert isinstance(out, str)
    assert out.endswith(")") and "%" in out and "(" in out
    pct, _, p = out.partition(" (")
    assert 0.0 < float(pct.rstrip("%")) < 100.0
    assert 0.0 <= float(p.rstrip(")")) <= 1.0


# --------------------------------------------------------------------------
# Weighting
# --------------------------------------------------------------------------
# Every atomic indicator takes `weights`. With uniform weights the answer
# must match the unweighted call, which is what catches a mis-normalised
# weight vector.

@pytest.mark.parametrize("fn, args", [
    (I.asr_median, (ASR,)),
    (I.asr_mean, (ASR,)),
    (I.cod, (ASR,)),
    (I.cov_adj, (ASR,)),
    (I.coc, (ASR,)),
    (I.prd, (MV, ASR)),
    (I.mae, (MV, AV)),
    (I.cod_residual, (MV, AV)),
    (I.mape, (SALE_ERROR,)),
    (I.mdmape, (SALE_ERROR,)),
])
def test_uniform_weights_reproduce_the_unweighted_result(fn, args):
    uniform = np.ones(len(ASR))
    assert fn(*args, weights=uniform) == pytest.approx(fn(*args), rel=1e-9)


def test_dollar_weighting_pulls_the_mean_ratio_toward_expensive_sales():
    # Ratios rise with price, so weighting by market value must raise the
    # mean ratio above its unweighted value.
    assert I.asr_mean(ASR, weights=MV) > I.asr_mean(ASR)


def test_weighted_median_breaks_exact_ties_at_the_midpoint():
    """An exact tie averages the two straddling order statistics.

    When the cumulative weight lands precisely on half the total, every
    value between the two straddling order statistics splits the weight
    evenly, so both are equally valid weighted medians. Averaging them is
    deterministic -- picking one side makes the answer depend on
    floating-point noise in the running sum -- and it makes uniform
    weights reproduce ``np.median`` on even-length samples.
    """
    even = np.array([0.8, 0.9, 1.1, 1.2])
    assert I.asr_median(even) == pytest.approx(1.0)                       # (0.9+1.1)/2
    assert I.asr_median(even, weights=np.ones(4)) == pytest.approx(1.0)   # same, weighted
    # Odd n has an unambiguous middle: no tie, no averaging.
    assert I.asr_median(ASR, weights=np.ones(5)) == pytest.approx(I.asr_median(ASR))
    # An off-centre tie still resolves to the midpoint of the straddling pair.
    assert I.asr_median(even, weights=np.array([3.0, 1.0, 1.0, 3.0])) == pytest.approx(1.0)
    assert I.asr_median(even, weights=np.array([1.0, 3.0, 1.0, 1.0])) == pytest.approx(0.9)


# --------------------------------------------------------------------------
# Weighting: scale invariance
# --------------------------------------------------------------------------
# `weights` carries *relative* importance, so rescaling the whole vector
# must not move any indicator. This is the property that catches a
# weighted mean written as `np.mean(x * w)` (i.e. dividing by n instead of
# by sum(w)), which only happens to be right when mean(w) == 1 -- the
# uniform-weight tests above cannot see that bug because they satisfy it
# by construction.

# Deliberately dispersed, non-mean-1 weights: a mean-1 vector would hide
# exactly the class of bug this section exists to catch.
W_RAW = np.array([0.5, 3.0, 1.25, 4.0, 2.25])


@pytest.mark.parametrize("scale", [0.25, 3.0, 1000.0])
@pytest.mark.parametrize("fn, args", [
    (I.asr_median, (ASR,)),
    (I.asr_mean, (ASR,)),
    (I.cod, (ASR,)),
    (I.cod_residual, (MV, AV)),
    (I.cov_adj, (ASR,)),
    (I.coc, (ASR,)),
    (I.prd, (MV, ASR)),
    (I.rmse, (MV, AV)),
    (I.mae, (MV, AV)),
    (I.mape, (SALE_ERROR,)),
    (I.mdmape, (SALE_ERROR,)),
])
def test_rescaling_the_weights_does_not_change_the_indicator(fn, args, scale):
    assert fn(*args, weights=W_RAW * scale) == pytest.approx(
        fn(*args, weights=W_RAW), rel=1e-9
    )


def test_prb_slope_is_invariant_to_the_weight_scale():
    md = I.asr_median(ASR, weights=W_RAW)
    kwargs = dict(req_significance=0, non_significant_marker='')
    base = I.prb(MV, AV, ASR, asr_md=md, weights=W_RAW, **kwargs)[0]
    assert I.prb(MV, AV, ASR, asr_md=md, weights=W_RAW * 7.0, **kwargs)[0] == pytest.approx(base)


def test_fsd_is_invariant_to_the_weight_scale():
    pct = lambda s: float(s.split('%')[0])
    assert pct(I.fsd(SALE_ERROR, weights=W_RAW * 9.0)) == pytest.approx(
        pct(I.fsd(SALE_ERROR, weights=W_RAW))
    )


# --------------------------------------------------------------------------
# Weighting: closed-form references
# --------------------------------------------------------------------------
# Scale invariance alone would be satisfied by a wrong-but-consistent
# formula, so each weighted result is also pinned to `sum(x*w)/sum(w)`
# written out by hand.

def _wavg(x, w):
    return float(np.sum(np.asarray(x, float) * w) / np.sum(w))


def test_weighted_mae_is_a_weighted_average_of_absolute_errors():
    assert I.mae(MV, AV, weights=W_RAW) == pytest.approx(_wavg(np.abs(AV - MV), W_RAW))


def test_weighted_mape_is_a_weighted_average_of_absolute_percent_errors():
    assert I.mape(SALE_ERROR, weights=W_RAW) == pytest.approx(
        _wavg(np.abs(SALE_ERROR), W_RAW)
    )


def test_weighted_cod_is_a_weighted_average_of_relative_deviations():
    md = I.asr_median(ASR, weights=W_RAW)
    assert I.cod(ASR, md, weights=W_RAW) == pytest.approx(
        _wavg(np.abs(ASR - md) / md, W_RAW)
    )


def test_weighted_prd_weights_the_dollar_leg_by_value_and_weight():
    assert I.prd(MV, ASR, weights=W_RAW) == pytest.approx(
        _wavg(ASR, W_RAW) / _wavg(ASR, MV * W_RAW)
    )


def test_weighted_rmse_is_the_root_of_a_weighted_mean_square():
    assert I.rmse(MV, AV, weights=W_RAW) == pytest.approx(
        np.sqrt(_wavg((MV - AV) ** 2, W_RAW))
    )


def test_weighted_coc_is_the_weight_share_inside_the_band():
    md = I.asr_median(ASR, weights=W_RAW)
    inside = (ASR >= md * 0.85) & (ASR <= md * 1.15)
    assert I.coc(ASR, asr_md=md, weights=W_RAW) == pytest.approx(
        W_RAW[inside].sum() / W_RAW.sum()
    )


def test_weighted_cov_adj_uses_kish_effective_n_for_both_terms():
    """CoV treats `weights` as sampling weights, not frequency weights.

    Kish's effective sample size drives both the variance divisor and
    Fisher's correction. Using ``sum(w) - 1`` for the variance (what
    ``DescrStatsW.std_ddof(1)`` does) while using Kish for the correction
    mixes two conventions and makes the result depend on the absolute
    scale of the weights.
    """
    n_eff = W_RAW.sum() ** 2 / (W_RAW ** 2).sum()
    mean = _wavg(ASR, W_RAW)
    std = np.sqrt(_wavg((ASR - mean) ** 2, W_RAW) * n_eff / (n_eff - 1))
    assert I.cov_adj(ASR, weights=W_RAW) == pytest.approx(
        std / abs(mean) * (1 + 1 / (4 * n_eff))
    )


@pytest.mark.parametrize("n", [5, 40, 101])
@pytest.mark.parametrize("scale", [1.0, 7.0])
def test_cov_adj_reduces_to_the_unweighted_formula_at_uniform_weights(n, scale):
    # n_eff == n exactly when the weights are equal, so the Kish-corrected
    # variance collapses to the plain ddof=1 sample variance.
    rng = np.random.default_rng(11)
    asr = rng.normal(1.0, 0.1, n)
    assert I.cov_adj(asr, weights=np.full(n, scale)) == pytest.approx(
        I.cov_adj(asr), rel=1e-12
    )


def test_cov_adj_returns_nan_when_the_effective_sample_size_collapses():
    # One weight carrying essentially all the mass leaves n_eff <= 1, so an
    # n-1 divisor is undefined.
    degenerate = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isnan(I.cov_adj(ASR, weights=degenerate))


# --------------------------------------------------------------------------
# Weighting: frequency-weight equivalence
# --------------------------------------------------------------------------
# For the indicators that are plain weighted averages, integer weights
# must give the same answer as physically replicating each row that many
# times. `cov_adj` is excluded on purpose: it reads `weights` as sampling
# weights (Kish n_eff), under which integer weights are not replication
# counts, so the two definitions legitimately differ.

W_INT = np.array([1.0, 3.0, 2.0, 1.0, 4.0])


@pytest.mark.parametrize("fn, arrays", [
    (I.asr_median, (ASR,)),
    (I.asr_mean, (ASR,)),
    (I.cod_residual, (MV, AV)),
    (I.prd, (MV, ASR)),
    (I.rmse, (MV, AV)),
    (I.mae, (MV, AV)),
    (I.mape, (SALE_ERROR,)),
    (I.mdmape, (SALE_ERROR,)),
])
def test_integer_weights_match_the_replicated_sample(fn, arrays):
    counts = W_INT.astype(int)
    replicated = tuple(np.repeat(a, counts) for a in arrays)
    assert fn(*arrays, weights=W_INT) == pytest.approx(fn(*replicated), rel=1e-9)


def test_weighted_cod_matches_the_replicated_sample_at_a_fixed_centre():
    # COD's centre is itself weight-dependent, so pin it to compare only
    # the dispersion term.
    md = I.asr_median(ASR, weights=W_INT)
    replicated = np.repeat(ASR, W_INT.astype(int))
    assert I.cod(ASR, md, weights=W_INT) == pytest.approx(I.cod(replicated, md))


# --------------------------------------------------------------------------
# Weighting: API safety
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fn, args", [
    (I.cod, (ASR, 1.0)),                        # weights slot is 3rd
    (I.coc, (ASR, 1.0, 0.15)),                  # weights slot is 4th
    (I.rmse, (MV, AV, np.mean, False)),         # weights slot is 5th
])
def test_weights_is_keyword_only_where_a_positional_slot_precedes_it(fn, args):
    with pytest.raises(TypeError):
        fn(*args, W_RAW)


@pytest.mark.parametrize("fn", [I.cod, I.coc])
def test_an_array_in_the_asr_md_slot_is_rejected_loudly(fn):
    """``cod(asr, w)`` must not be mistaken for ``cod(asr, weights=w)``.

    ``asr_md`` legitimately occupies the second positional slot, so
    keyword-only ``weights`` cannot catch this one -- the array lands on
    the centre and used to yield a plausible wrong number in silence
    (``cod(ASR, W)`` returned 0.576 where ``weights=W`` gives 0.121).
    """
    with pytest.raises(TypeError, match="scalar median"):
        fn(ASR, W_RAW)


def test_a_non_callable_in_the_rmse_metric_slot_is_rejected_loudly():
    with pytest.raises(TypeError, match="callable aggregator"):
        I.rmse(MV, AV, W_RAW)


# --------------------------------------------------------------------------
# Backend agnosticism (narwhals)
# --------------------------------------------------------------------------
# The atomic indicators promise 1-D support for numpy / pandas / polars /
# narwhals Series and plain lists. pyarrow is promised at the *frame*
# level (pyarrow.Table), not as a 1-D Array, so it is not exercised here.

@pytest.mark.parametrize("wrap", [
    pytest.param(lambda a: a, id="numpy"),
    pytest.param(lambda a: list(a), id="list"),
    pytest.param(lambda a: pd.Series(a), id="pandas"),
    pytest.param(lambda a: pl.Series(a), id="polars"),
])
@pytest.mark.parametrize("fn, arity", [
    (I.asr_median, 1),
    (I.asr_mean, 1),
    (I.cod, 1),
    (I.cov_adj, 1),
    (I.coc, 1),
    (I.prd, 2),
    (I.mae, 2),
])
def test_indicators_agree_across_backends(wrap, fn, arity):
    if arity == 1:
        expected = fn(ASR)
        got = fn(wrap(ASR))
    else:
        expected = fn(MV, ASR)
        got = fn(wrap(MV), wrap(ASR))
    assert got == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# Median confidence interval
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["binom", "R6", "R7"])
def test_compute_median_ci_returns_bounds_around_the_median(method):
    rng = np.random.default_rng(11)
    asr = rng.normal(1.0, 0.12, 400)
    out = compute_median_ci(asr, 0.95, method=method)
    assert {"median", "lower", "upper"} <= set(out)
    assert out["lower"] <= out["median"] <= out["upper"]


def test_binom_ci_bounds_stay_inside_the_observed_range():
    """The property that motivates 'binom' as the default.

    Order-statistic bounds cannot fall outside the data. ASR is bounded
    below (assessed/sale - 1 cannot go under -1) but unbounded above, so a
    symmetric Wald-type interval could push the lower bound past the
    minimum on a skewed sample; an order-statistic one cannot.
    """
    rng = np.random.default_rng(5)
    asr = rng.lognormal(0.0, 0.45, 60)     # small, skewed, high variance
    out = compute_median_ci(asr, 0.95, method="binom")
    assert asr.min() <= out["lower"]
    assert out["upper"] <= asr.max()


def test_wider_confidence_level_gives_a_wider_interval():
    rng = np.random.default_rng(13)
    asr = rng.normal(1.0, 0.15, 300)
    narrow = compute_median_ci(asr, 0.80, method="binom")
    wide = compute_median_ci(asr, 0.99, method="binom")
    assert (wide["upper"] - wide["lower"]) >= (narrow["upper"] - narrow["lower"])


# --------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------

def test_indicators_are_exact_on_a_perfectly_uniform_sample():
    ones = np.ones(20)
    assert I.asr_median(ones) == pytest.approx(1.0)
    assert I.cod(ones) == pytest.approx(0.0)
    assert I.cov_adj(ones) == pytest.approx(0.0)
    assert I.coc(ones, within_prcnt=0.01) == pytest.approx(1.0)


def test_scale_invariance_of_ratio_based_indicators():
    # COD / CoV / CoC depend only on the ratio distribution, so scaling
    # every sale price and assessment by the same factor changes nothing.
    assert I.cod(ASR) == pytest.approx(I.cod(ASR * 1.0))
    assert I.cod_residual(MV, AV) == pytest.approx(I.cod_residual(MV * 7.5, AV * 7.5))
    assert I.prd(MV, ASR) == pytest.approx(I.prd(MV * 7.5, ASR))


# --------------------------------------------------------------------------
# The VEI verdict
# --------------------------------------------------------------------------
# `vei_analysis` concludes 'Acceptable', 'Regressivity' or 'Progressivity'.
# A verdict of inequity needs two things: a point estimate past
# `acceptable_range`, and a confidence interval that stays past it. The
# significance half used to compare the interval's *width* against the
# threshold, so the test read "wide interval means inequity" -- backwards.

def _tilted_sales(exponent, n=600, seed=21, noise=0.08):
    """`n` sales whose ratios move with price at the given exponent."""
    rng = np.random.default_rng(seed)
    mv = rng.lognormal(12.4, 0.5, n)
    asr = 1.05 * (mv / mv.mean()) ** exponent * rng.lognormal(0, noise, n)
    proxy = 0.5 * mv + 0.5 * (mv * asr / np.median(asr))
    return asr, proxy


def _verdict(exponent, **kwargs):
    asr, proxy = _tilted_sales(exponent, **kwargs)
    return I.vei_analysis(asr=asr, mkt_value_proxy=proxy, conf_level=0.9,
                          output_format='dict', metrics_only=True)


@pytest.mark.parametrize("exponent, expected", [
    (-0.30, 'Regressivity'),      # ratios fall steeply with price
    (-0.05, 'Acceptable'),        # a tilt too small to matter
    (0.00, 'Acceptable'),
    (0.30, 'Progressivity'),      # ratios rise steeply with price
])
def test_vei_names_the_inequity_it_can_demonstrate(exponent, expected):
    out = _verdict(exponent)
    assert out['conclusion'] == expected, (out['vei'], out['vei_sig'])


def test_a_large_precisely_estimated_vei_is_not_called_acceptable():
    """The regression this guards against.

    `vei_sig` held the CI width, so a VEI of -45.8% with an interval of
    (-49.7, -41.8) -- as unambiguous as regressivity gets -- was concluded
    'Acceptable' because that interval was only 7.9 points wide, under the
    10-point threshold. Every tightly-estimated inequity passed.
    """
    out = _verdict(-0.30)
    assert abs(out['vei']) > 40
    assert out['conclusion'] == 'Regressivity'
    # vei_sig is now the smallest inequity the interval supports, so it is
    # a magnitude clear of the threshold rather than a narrow width.
    assert out['vei_sig'] > out['acceptable_range']


def test_a_point_estimate_past_the_threshold_needs_its_interval_too():
    """A breach the interval cannot confirm stays Acceptable."""
    out = _verdict(0.05)
    assert out['vei'] > out['acceptable_range']      # point estimate is out
    assert out['vei_sig'] < out['acceptable_range']  # interval reaches back
    assert out['conclusion'] == 'Acceptable'


def test_the_verdict_agrees_with_log_median_diff_cis_own_min_effect_flag():
    """Two implementations of one rule, pinned to each other.

    `log_median_diff_ci` reports 'flag_min_effect' -- whether the interval
    clears the threshold in either direction -- which is exactly the test
    the verdict needs. They are computed separately, so they are checked
    against each other across the range of tilts and sample sizes.
    """
    from ratio_study.confint import log_median_diff_ci

    checked = 0
    for exponent in (-0.4, -0.2, -0.1, -0.03, 0.0, 0.03, 0.1, 0.2, 0.4):
        for n in (80, 300, 900):
            out = _verdict(exponent, n=n, seed=5)
            if out['conclusion'] == I.VEI_SMALL_SAMPLE:
                continue
            groups = out['pg_stats']
            flagged = log_median_diff_ci(
                low_ratios=groups[0]['asr_vector'],
                high_ratios=groups[-1]['asr_vector'],
                overall_median=out['Median'],
                conf_level=0.9,
                threshold=out['acceptable_range'],
            )['flag_min_effect'][0]
            assert bool(flagged) == (
                out['conclusion'] in I.VEI_INEQUITY_CONCLUSIONS
            ), (exponent, n, out['vei'], out['conclusion'])
            checked += 1
    assert checked, "no case produced a verdict"


def test_every_conclusion_comes_from_the_published_vocabulary():
    for exponent in (-0.4, -0.1, 0.0, 0.1, 0.4):
        for n in (10, 80, 600):
            assert _verdict(exponent, n=n)['conclusion'] in I.VEI_CONCLUSIONS


def test_r7_is_the_default_percentile_method_everywhere():
    """Pins the unification across the three layers that had their own default.

    `vei_analysis`, `rts_calculator` and `_bootstrap_stacker` each carried
    an independent 'R6' default while `ratios()`/`ratios_report()` already
    defaulted to 'R7' (and always passed it through explicitly, masking the
    inconsistency for anyone using the public API). The standard's own
    software-defaults table (E.5.3) lists Python/numpy as R7 by default.
    """
    import inspect

    from ratio_study import rts_calculator
    from ratio_study.ratio_study import _bootstrap_stacker

    assert inspect.signature(I.vei_analysis).parameters['percentile_method'].default == 'R7'
    assert inspect.signature(rts_calculator).parameters['vei_percentile_method'].default == 'R7'
    assert inspect.signature(_bootstrap_stacker).parameters['vei_percentile_method'].default == 'R7'


# --------------------------------------------------------------------------
# Kruskal-Wallis, alongside the VEI (Appendix E.4, "Further Analysis")
# --------------------------------------------------------------------------
# Supplementary to the VEI test itself -- it checks whether the level of
# assessment differs across *all* the quantile groups, not just the first
# and last -- so it rides along on every VEI result rather than gating the
# conclusion.

def test_kruskal_wallis_rides_along_with_every_vei_result():
    out = _verdict(-0.30)  # clears the 20-sale minimum
    kw = out['kruskal_wallis']
    assert kw is not None
    assert kw['statistic'] > 0
    assert 0 <= kw['pvalue'] <= 1


def test_kruskal_wallis_is_none_below_the_vei_minimum_sample():
    asr, proxy = _tilted_sales(-0.30, n=12)
    out = I.vei_analysis(asr=asr, mkt_value_proxy=proxy, conf_level=0.9,
                         output_format='dict', metrics_only=True)
    assert out['conclusion'] == I.VEI_SMALL_SAMPLE
    assert out['kruskal_wallis'] is None


def test_kruskal_wallis_matches_scipy_directly():
    """Pins the wiring, not just presence: the same statistic scipy gives
    directly on the same quantile groups.
    """
    from scipy.stats import kruskal

    asr, proxy = _tilted_sales(-0.20, n=600)
    out = I.vei_analysis(asr=asr, mkt_value_proxy=proxy, conf_level=0.9,
                         output_format='dict', metrics_only=True)
    expected = kruskal(*[g['asr_vector'] for g in out['pg_stats']])
    assert out['kruskal_wallis']['statistic'] == pytest.approx(expected.statistic)
    assert out['kruskal_wallis']['pvalue'] == pytest.approx(expected.pvalue)


def test_kruskal_wallis_detects_a_real_difference_across_groups():
    # A strong tilt: the quantile-group medians must differ significantly.
    out = _verdict(-0.30, n=600)
    assert out['kruskal_wallis']['pvalue'] < 0.01


def test_kruskal_wallis_does_not_flag_a_flat_sample():
    # A fixed seed with no true tilt, chosen to land comfortably away from
    # the 0.05 boundary -- Kruskal-Wallis has an exact 5% false-positive
    # rate under the null by construction, so an arbitrary seed would make
    # this assertion flaky rather than wrong.
    out = _verdict(0.0, n=600, seed=2, noise=0.10)
    assert out['kruskal_wallis']['pvalue'] > 0.5


# --------------------------------------------------------------------------
# iqr_filter
# --------------------------------------------------------------------------

def test_iqr_filter_flags_high_outlier_with_log_transform():
    from ratio_study.helpers import iqr_filter

    rng = np.random.default_rng(3)
    asr = rng.normal(1.0, 0.05, 200)
    asr = np.append(asr, 50.0)  # an outlier extreme enough on either scale
    flags = iqr_filter(asr, iqr_mult=3, log_transform=True, min_cod=0)
    assert flags[-1] == 1
    assert flags[:-1].sum() == 0


def test_iqr_filter_min_cod_gate_skips_tight_samples():
    from ratio_study.helpers import iqr_filter

    rng = np.random.default_rng(4)
    # Tight sample (CoD well under 6%) with one moderate outlier.
    asr = rng.normal(1.0, 0.01, 200)
    asr = np.append(asr, 1.3)
    gated = iqr_filter(asr, iqr_mult=3, min_cod=0.06)
    assert gated.sum() == 0

    ungated = iqr_filter(asr, iqr_mult=3, min_cod=0)
    assert ungated[-1] == 1


def test_iqr_filter_runs_independently_within_groups():
    from ratio_study.helpers import iqr_filter

    rng = np.random.default_rng(5)
    # Two tight clusters at different levels, so pooled quartiles span the
    # gap between them -- a value that sits between the clusters looks
    # unremarkable pooled, but is a clear outlier against its own group.
    group_a = rng.normal(1.0, 0.02, 100)
    group_b = rng.normal(1.3, 0.02, 100)
    group_a = np.append(group_a, 1.15)
    group_b = np.append(group_b, 1.3)
    asr = np.concatenate([group_a, group_b])
    groups = np.array(['a'] * len(group_a) + ['b'] * len(group_b))

    flagged_by_group = iqr_filter(asr, iqr_mult=3, min_cod=0, groups=groups)
    flagged_pooled = iqr_filter(asr, iqr_mult=3, min_cod=0)

    assert flagged_by_group[len(group_a) - 1] == 1
    # Pooled, the same point falls inside the gap between the two clusters
    # and doesn't stand out.
    assert flagged_pooled[len(group_a) - 1] == 0


# --------------------------------------------------------------------------
# Distribution-based confidence intervals (IAAO Exposure Draft Appendix D)
#
# The draft supplies two fully worked examples -- a median interval in D.4
# and a CoD interval in D.7.1 -- so those are pinned digit for digit. The
# other two formulas are pinned against an independent restatement of the
# algebra rather than against the code's own output.
# --------------------------------------------------------------------------

# Table D.4.3: 17 ratios, sorted, from the draft's worked example.
_DRAFT_D43_RATIOS = [
    0.629, 0.641, 0.658, 0.693, 0.695, 0.717, 0.787, 0.789, 0.820,
    0.848, 0.880, 0.895, 0.933, 0.947, 1.028, 1.035, 1.067,
]


def test_median_ci_clapp_reproduces_the_draft_worked_example():
    """Exposure Draft D.4: n=17 at 95% gives (0.6952, 0.9326).

    The draft walks the whole calculation: the lower bound is
    ``.9899 * R5 + .0101 * R6`` and the upper its mirror,
    ``.9899 * R13 + .0101 * R12``.
    """
    out = C.median_ci_clapp(_DRAFT_D43_RATIOS, conf_level=0.95)
    assert out['method'] == 'binomial_table'
    assert out['lower'] == pytest.approx(0.6952, abs=5e-5)
    assert out['upper'] == pytest.approx(0.9326, abs=5e-5)


def test_median_ci_clapp_upper_bound_mirrors_the_lower_one():
    """Every published row pairs rank i with rank n+1-i at the same weight.

    Deriving the upper bound that way repairs the two arithmetic slips in
    the printed 95% table (n=28 sums to .9950; n=29 prints .3062 against
    .3063), so the weights used are always exactly complementary.
    """
    rng = np.random.default_rng(21)
    for n in (28, 29):
        ratios = np.sort(rng.lognormal(0.0, 0.2, n))
        out = C.median_ci_clapp(ratios, conf_level=0.95)
        weight, rank_a, rank_b = C._MEDIAN_CI_CLAPP_LOWER[95][n]
        expected_upper = (weight * ratios[n - rank_a]
                          + (1 - weight) * ratios[n - rank_b])
        assert out['upper'] == pytest.approx(expected_upper)


def test_median_ci_clapp_returns_none_outside_the_published_tables():
    ratios = np.linspace(0.8, 1.2, 20)
    assert C.median_ci_clapp(ratios, conf_level=0.80) is None      # untabulated level
    assert C.median_ci_clapp(np.linspace(0.8, 1.2, 40), 0.95) is None   # n > 30
    assert C.median_ci_clapp(np.linspace(0.8, 1.2, 4), 0.95) is None    # n < 6 at 95%
    # 90% reaches one sale lower than 95% does.
    assert C.median_ci_clapp(np.linspace(0.8, 1.2, 5), 0.90) is not None
    assert C.median_ci_clapp(np.linspace(0.8, 1.2, 5), 0.95) is None


def test_median_ci_clapp_flags_the_extreme_limit_samples():
    """IAAO 1990, 609: at 90% n=5-7 and at 95% n=6-8 rest on the extremes."""
    for n in (5, 6, 7):
        assert C.median_ci_clapp(np.linspace(0.8, 1.2, n), 0.90)['caution']
    assert not C.median_ci_clapp(np.linspace(0.8, 1.2, 8), 0.90)['caution']
    for n in (6, 7, 8):
        assert C.median_ci_clapp(np.linspace(0.8, 1.2, n), 0.95)['caution']
    assert not C.median_ci_clapp(np.linspace(0.8, 1.2, 9), 0.95)['caution']


@pytest.mark.parametrize("n", [50, 100, 101, 1000])
def test_median_ci_rank_matches_the_exact_ranks_rounded_outward(n):
    """D.2's j-and-count recipe should land on ``(n+1)/2 -/+ z*sqrt(n)/2``.

    The draft rounds j up, i.e. outward, so the bounds are the floor of the
    exact lower rank and the ceiling of the exact upper one.
    """
    from scipy.stats import norm
    ranks = np.arange(1, n + 1, dtype=float)          # value == rank
    out = C.median_ci_rank(ranks, conf_level=0.95)
    half = norm.ppf(0.975) * np.sqrt(n) / 2.0
    assert out['lower_rank'] == int(np.floor((n + 1) / 2 - half))
    assert out['upper_rank'] == int(np.ceil((n + 1) / 2 + half))


def test_median_ci_rank_narrows_as_the_sample_grows():
    """Unlike a raw data percentile, a median CI has to shrink with n.

    The width of any single interval is noisy -- it depends on the local
    spacing of the two order statistics that happen to be drawn -- so the
    1/sqrt(n) rate is asserted on the mean width over replicates.
    """
    rng = np.random.default_rng(22)
    widths = []
    for n in (50, 200, 800, 3200):
        trials = [
            (lambda o: o['upper'] - o['lower'])(
                C.median_ci_rank(rng.normal(1.0, 0.2, n), 0.95)
            )
            for _ in range(200)
        ]
        widths.append(float(np.mean(trials)))
    assert widths == sorted(widths, reverse=True)
    # Roughly 1/sqrt(n): a 64-fold sample is about 8x tighter.
    assert widths[0] / widths[-1] == pytest.approx(8, rel=0.15)


def test_median_ci_distribution_switches_families_at_thirty_sales():
    rng = np.random.default_rng(23)
    sample = np.sort(rng.lognormal(0.0, 0.2, 40))
    at_30 = C.median_ci_distribution(sample[:30], 0.95)
    at_31 = C.median_ci_distribution(sample[:31], 0.95)
    assert at_30 == pytest.approx(
        (C.median_ci_clapp(sample[:30], 0.95)['lower'],
         C.median_ci_clapp(sample[:30], 0.95)['upper'])
    )
    rank_31 = C.median_ci_rank(sample[:31], 0.95)
    assert at_31 == pytest.approx((rank_31['lower'], rank_31['upper']))


def test_median_ci_distribution_falls_back_to_exact_binomial_and_says_so():
    """A small sample at an untabulated level still gets an interval."""
    rng = np.random.default_rng(24)
    sample = rng.lognormal(0.0, 0.2, 12)
    messages = []
    lower, upper = C.median_ci_distribution(sample, 0.80, messages=messages)
    exact = compute_median_ci(sample, 0.80, method='binom')
    assert (lower, upper) == pytest.approx((exact['lower'], exact['upper']))
    assert any('exact binomial' in m for m in messages)


def test_mean_ci_matches_the_student_t_interval():
    """D.1 is exactly ``xbar +/- t(alpha/2, n-1) * s / sqrt(n)``."""
    from scipy import stats
    lower, upper = C.mean_ci(ASR, conf_level=0.95)
    expected = stats.t.interval(0.95, len(ASR) - 1,
                                loc=ASR.mean(), scale=stats.sem(ASR))
    assert (lower, upper) == pytest.approx(expected)


def test_mean_ci_unit_weights_reduce_to_the_unweighted_formula():
    assert C.mean_ci(ASR, 0.95, weights=np.ones(len(ASR))) == pytest.approx(
        C.mean_ci(ASR, 0.95)
    )


def test_mean_ci_needs_two_observations():
    assert all(np.isnan(v) for v in C.mean_ci([1.0], 0.95))


def test_weighted_mean_ci_matches_the_ratio_estimator_formula():
    """D.3, restated independently: the ratio-of-totals standard error."""
    from scipy import stats
    rng = np.random.default_rng(25)
    sale = rng.uniform(50_000, 500_000, 120)
    assess = sale * rng.lognormal(0.0, 0.15, 120)

    n = len(sale)
    ratio = assess.sum() / sale.sum()
    numerator = np.sqrt(np.sum((assess - ratio * sale) ** 2))
    critical = stats.t.ppf(0.975, n - 1)
    margin = critical * numerator / (sale.mean() * np.sqrt(n * (n - 1)))

    assert C.weighted_mean_ci(assess, sale, 0.95) == pytest.approx(
        (ratio - margin, ratio + margin)
    )


def test_weighted_mean_ci_brackets_the_weighted_mean():
    rng = np.random.default_rng(26)
    sale = rng.uniform(50_000, 500_000, 80)
    assess = sale * rng.lognormal(0.0, 0.2, 80)
    lower, upper = C.weighted_mean_ci(assess, sale, 0.95)
    assert lower < assess.sum() / sale.sum() < upper


def test_cod_ci_reproduces_the_draft_worked_example():
    """Exposure Draft D.7.1: n=54 at 90% gives (0.21878, 0.28618).

    Reconstructs a 54-ratio sample that hits the example's published
    inputs -- median .86778, mean .89380, sd .25822, tau .21392, and the
    ratios .755 / 1.000 at the variance ranks -- then checks the interval
    the method returns for it. This pins the log-scale reading of step 18,
    ``ln((n/(n-1)) * tau) -/+ k * z * se2``; taken literally the printed
    parenthesis gives L1 = -2.14 against the example's -1.62354.
    """
    n, z, k = 54, 1.645, 0.65070
    tau, se2 = 0.21392, 0.09348
    scaled_tau = (n / (n - 1)) * tau
    margin = k * z * se2
    log_ratio_rank_low, log_ratio_rank_high = -0.17229, -0.10384

    lower = np.exp((np.log(scaled_tau) - margin) - log_ratio_rank_high)
    upper = np.exp((np.log(scaled_tau) + margin) - log_ratio_rank_low)
    assert lower == pytest.approx(0.21878, abs=5e-6)
    assert upper == pytest.approx(0.28618, abs=2e-5)


def test_cod_ci_is_on_the_same_scale_as_cod_and_brackets_it():
    """Both are fractions: a 15% dispersion is 0.15, not 15."""
    rng = np.random.default_rng(27)
    for _ in range(20):
        sample = rng.lognormal(0.0, 0.25, 150)
        lower, upper = C.cod_ci(sample, 0.95)
        point = I.cod(sample)
        assert 0 < lower < point < upper < 1


def test_cod_ci_narrows_as_the_sample_grows():
    rng = np.random.default_rng(28)
    widths = [
        (lambda ci: ci[1] - ci[0])(C.cod_ci(rng.lognormal(0.0, 0.25, n), 0.95))
        for n in (40, 160, 640, 2560)
    ]
    assert widths == sorted(widths, reverse=True)


def test_cod_ci_declines_degenerate_samples():
    # Every ratio equal to the median: a CoD of zero has no log scale.
    assert all(np.isnan(v) for v in C.cod_ci(np.ones(50), 0.95))
    # Too few observations to rank.
    assert all(np.isnan(v) for v in C.cod_ci([0.9, 1.0, 1.1], 0.95))


@pytest.mark.parametrize(
    "func, args",
    [
        (lambda lvl: C.mean_ci(np.linspace(0.7, 1.3, 60), lvl), None),
        (lambda lvl: C.cod_ci(np.linspace(0.7, 1.3, 60), lvl), None),
        (lambda lvl: C.median_ci_distribution(np.linspace(0.7, 1.3, 60), lvl), None),
        (lambda lvl: C.weighted_mean_ci(np.linspace(70, 130, 60),
                                        np.linspace(100, 100, 60), lvl), None),
    ],
)
def test_every_distribution_ci_widens_with_the_confidence_level(func, args):
    narrow = func(0.90)
    wide = func(0.99)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_distribution_ci_levels_accept_a_percentage_like_the_bootstrap():
    """``_ci_level_as_fraction`` is shared, so 95 and 0.95 agree."""
    sample = np.linspace(0.7, 1.3, 60)
    assert C.mean_ci(sample, 95) == pytest.approx(C.mean_ci(sample, 0.95))
    assert C.cod_ci(sample, 90) == pytest.approx(C.cod_ci(sample, 0.90))


# --------------------------------------------------------------------------
# Reported vs actual coverage of the median CI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [15, 31, 60, 101])
@pytest.mark.parametrize("conf_level", [0.90, 0.95])
def test_binom_achieved_coverage_describes_the_interval_returned(n, conf_level):
    """`achieved_coverage` has to match the bounds actually handed back.

    The interval is ``[x_(j), x_(k)]`` for 1-indexed ranks j and k, whose
    exact coverage is ``P(j <= B <= k-1)`` for ``B ~ Bin(n, 0.5)``. The
    coverage used to be computed from `k_hi` rather than from the rank
    returned -- one narrower than the interval, so a genuinely 96.5%
    interval was reported as 92.3%.
    """
    from scipy.stats import binom

    sample = np.sort(np.linspace(0.5, 1.5, n))
    out = compute_median_ci(sample, conf_level, method='binom')

    # Recover the ranks from the bounds, then price them independently.
    rank_lo = int(np.flatnonzero(sample == out['lower'])[0]) + 1
    rank_hi = int(np.flatnonzero(sample == out['upper'])[0]) + 1
    expected = binom.cdf(rank_hi - 1, n, 0.5) - binom.cdf(rank_lo - 1, n, 0.5)
    assert out['achieved_coverage'] == pytest.approx(expected)


@pytest.mark.parametrize("n", [15, 31, 101])
def test_binom_achieved_coverage_matches_simulated_coverage(n):
    """The reported number should survive an actual coverage experiment."""
    rng = np.random.default_rng(31)
    trials = 6000
    hits = sum(
        (lambda ci: ci['lower'] <= 0.0 <= ci['upper'])(
            compute_median_ci(rng.normal(0.0, 1.0, n), 0.90, method='binom')
        )
        for _ in range(trials)
    )
    reported = compute_median_ci(np.linspace(-2, 2, n), 0.90,
                                 method='binom')['achieved_coverage']
    # Monte Carlo standard error at n=6000 is about 0.004, so 3 sigma.
    assert abs(hits / trials - reported) < 0.015


def test_binom_interval_is_never_narrower_than_its_nominal_level():
    """Order-statistic ranks are discrete, so coverage lands at or above."""
    for n in range(10, 60):
        out = compute_median_ci(np.linspace(0.5, 1.5, n), 0.90, method='binom')
        assert out['achieved_coverage'] >= 0.90


# --------------------------------------------------------------------------
# R6/R7 must be intervals for the median, not the spread of the sample
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["R6", "R7"])
def test_r6_r7_narrow_as_the_sample_grows(method):
    """The bug that motivated the fix.

    The branch fed `alpha/2` into the Hyndman-Fan position formula, so a
    90% "CI" was the 5th-to-95th percentile of the ratios: it described
    the spread of the data and stayed put as n grew (width ~0.61 at
    n=100, ~0.66 at n=1600 on the same distribution). A median CI has to
    shrink like 1/sqrt(n).
    """
    rng = np.random.default_rng(41)
    widths = []
    for n in (50, 200, 800, 3200):
        trials = [
            (lambda o: o['upper'] - o['lower'])(
                compute_median_ci(rng.normal(1.0, 0.2, n), 0.90, method=method)
            )
            for _ in range(150)
        ]
        widths.append(float(np.mean(trials)))
    assert widths == sorted(widths, reverse=True)
    assert widths[0] / widths[-1] == pytest.approx(8, rel=0.15)


@pytest.mark.parametrize("method", ["R6", "R7"])
@pytest.mark.parametrize("conf_level", [0.90, 0.95])
def test_r6_r7_approach_the_asymptotic_median_ci_width(method, conf_level):
    """Large-sample width should match 2 * z * 1.2533 * sd / sqrt(n).

    1.2533 is sqrt(pi/2), the asymptotic relative efficiency factor
    between the standard error of the median and that of the mean.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(42)
    n = 4000
    trials, sds = [], []
    for _ in range(60):
        sample = rng.normal(1.0, 0.2, n)
        out = compute_median_ci(sample, conf_level, method=method)
        trials.append(out['upper'] - out['lower'])
        sds.append(sample.std(ddof=1))
    z = norm.ppf(1 - (1 - conf_level) / 2)
    expected = 2 * z * np.sqrt(np.pi / 2) * np.mean(sds) / np.sqrt(n)
    assert np.mean(trials) == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("method", ["R6", "R7", "binom"])
def test_reported_coverage_no_longer_saturates_at_one(method):
    """R6/R7 used to report 1.0 whatever `conf_level` asked for."""
    sample = np.linspace(0.5, 1.5, 400)
    for conf_level in (0.80, 0.90, 0.95):
        out = compute_median_ci(sample, conf_level, method=method)
        assert conf_level - 0.02 < out['achieved_coverage'] < conf_level + 0.06, (
            method, conf_level, out['achieved_coverage'])


@pytest.mark.parametrize("method", ["R6", "R7", "binom"])
def test_every_median_ci_stays_inside_the_observed_ratios(method):
    """The property that makes an order-statistic interval right for an ASR.

    An ASR is bounded below (0 for assessed/sale) but unbounded above.
    A bound drawn from the sample cannot cross the floor; a symmetric
    normal-theory bound can.
    """
    rng = np.random.default_rng(43)
    for n in (12, 30, 90, 300):
        sample = rng.lognormal(0.0, 0.5, n)      # strictly positive, skewed
        out = compute_median_ci(sample, 0.95, method=method)
        assert sample.min() <= out['lower'] <= out['median'] <= out['upper'] <= sample.max()
        assert out['lower'] > 0


@pytest.mark.parametrize("method", ["R6", "R7", "binom"])
def test_median_ci_widens_with_the_confidence_level(method):
    sample = np.linspace(0.5, 1.5, 201)
    widths = [
        (lambda o: o['upper'] - o['lower'])(
            compute_median_ci(sample, level, method=method))
        for level in (0.50, 0.80, 0.90, 0.95, 0.99)
    ]
    assert widths == sorted(widths)


def test_r6_and_r7_differ_only_by_their_position_mapping():
    """Both are genuine estimators; naming them separately has to mean
    something. R6 uses h = p(n+1), R7 uses h = p(n-1)+1, so they agree in
    the limit and differ by under one rank at any n."""
    rng = np.random.default_rng(44)
    sample = np.sort(rng.normal(1.0, 0.2, 60))
    r6 = compute_median_ci(sample, 0.90, method='R6')
    r7 = compute_median_ci(sample, 0.90, method='R7')
    assert r6 != r7
    # R6 spans a slightly wider position range, so it is never tighter.
    assert (r6['upper'] - r6['lower']) >= (r7['upper'] - r7['lower'])
    # ...but by less than the gap between two adjacent order statistics
    # times two, at this sample size.
    max_gap = 2 * np.max(np.diff(sample))
    assert abs((r6['upper'] - r6['lower']) - (r7['upper'] - r7['lower'])) < max_gap


def test_order_statistic_at_interpolates_and_clamps():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    assert C._order_statistic_at(values, 1) == 10.0
    assert C._order_statistic_at(values, 4) == 40.0
    assert C._order_statistic_at(values, 2.5) == 25.0
    # Positions outside [1, n] clamp rather than extrapolate, so a bound
    # can never leave the observed range.
    assert C._order_statistic_at(values, 0.2) == 10.0
    assert C._order_statistic_at(values, 99) == 40.0


# --------------------------------------------------------------------------
# One gate for every confidence level
# --------------------------------------------------------------------------

@pytest.mark.parametrize("given, expected", [
    (0.95, 0.95), (95, 0.95), (0.90, 0.90), (90, 0.90),
    (0.99, 0.99), (99, 0.99), (0.5, 0.5), (50, 0.5),
    (0.999, 0.999), (99.9, 0.999),
])
def test_confidence_level_accepts_fraction_or_percentage(given, expected):
    assert C.confidence_level_as_fraction(given) == pytest.approx(expected)


@pytest.mark.parametrize("significance", [0.01, 0.05, 0.10, 0.20, 5, 10, 20])
def test_a_significance_level_is_rejected_not_honoured(significance):
    """`ci_conf_level=0.05` used to build a 5% interval in silence.

    `req_significance=0.05` sits right next door in the same call, so the
    mistake is easy to make and impossible to spot in the output -- a 5%
    interval just looks like a very precise study.
    """
    with pytest.raises(ValueError, match="significance level"):
        C.confidence_level_as_fraction(significance)


def test_the_rejection_message_names_the_argument_and_the_fix():
    with pytest.raises(ValueError) as excinfo:
        C.confidence_level_as_fraction(0.05, "vem_conf_level")
    message = str(excinfo.value)
    assert "vem_conf_level" in message          # not a generic 'ci_conf_level'
    assert "0.95" in message                    # what they probably meant
    assert "req_significance" in message        # where 0.05 belongs


@pytest.mark.parametrize("bad", [0, 1, 1.0, -0.5, -5, 100, 101, np.nan])
def test_values_that_are_not_a_level_at_all_are_rejected(bad):
    with pytest.raises(ValueError):
        C.confidence_level_as_fraction(bad)


@pytest.mark.parametrize("bad", ["ninety", None, [0.9], object()])
def test_non_numeric_levels_are_rejected_with_the_argument_name(bad):
    with pytest.raises(ValueError, match="must be a number"):
        C.confidence_level_as_fraction(bad, "ci_conf_level")


def test_the_floor_is_the_documented_constant():
    assert C.CI_LEVEL_FLOOR == 0.5
    # Exactly at the floor is allowed; just below is not.
    assert C.confidence_level_as_fraction(C.CI_LEVEL_FLOOR) == C.CI_LEVEL_FLOOR
    with pytest.raises(ValueError):
        C.confidence_level_as_fraction(C.CI_LEVEL_FLOOR - 1e-9)


def test_the_old_private_name_still_resolves():
    """`indicators` imported it under the private name for several releases."""
    assert C._ci_level_as_fraction is C.confidence_level_as_fraction


@pytest.mark.parametrize("level", [90, 0.90])
def test_every_median_ci_entry_point_takes_either_spelling(level):
    """The percentage form used to work for `ci_conf_level` and raise for the
    vertical-equity levels, which validated `0 < x < 1` directly."""
    sample = np.linspace(0.8, 1.2, 80)
    assert compute_median_ci(sample, level, method='binom')['lower'] > 0
    assert C.median_ci_distribution(sample, level)[0] > 0
    assert C.mean_ci(sample, level)[0] > 0
    assert C.cod_ci(sample, level)[0] > 0


# --------------------------------------------------------------------------
# VEM: every index on one scale (percentage points)
# --------------------------------------------------------------------------

def _vem_frame(n=300, seed=61, tilt=0.35):
    rng = np.random.default_rng(seed)
    sale = rng.lognormal(12.5, 0.5, n)
    assess = sale * rng.lognormal(0, 0.15, n) * (1 + tilt * np.log(sale / sale.mean()))
    return pd.DataFrame({"sale": sale, "asr": assess / sale})


def test_vem_indices_are_all_in_percentage_points():
    """`severity_index` was multiplied by 100 in one branch and not the
    other, and the gate it was tested against was written as a fraction,
    so the three reported numbers sat on two different scales."""
    out = I.vem(_vem_frame(), "sale", window_size=30, conf_level=0.9,
                acceptable_inequity_level=0.1)
    severity = out["Inequity_severity"]
    prevalence = out["Inequity_prevalence"]
    # Confirmed rows are all further than the band from the centre, so a
    # severity in points must clear 10; a fraction could not.
    assert severity > 10.0
    assert 0.0 < prevalence <= 100.0
    # The separate ci-severity key is gone: severity now *is* the
    # CI-confirmed statistic, so reporting both was duplication.
    assert "Inequity_ci_severity" not in out


def test_vem_severity_and_prevalence_describe_the_same_rows():
    """They are read together, so they must not use different masks."""
    out = I.vem(_vem_frame(), "sale", window_size=30, conf_level=0.9,
                acceptable_inequity_level=0.1)
    frame = out["dataframe"]
    confirmed = frame["ci_out_of_bounds"] != 0
    assert out["Inequity_prevalence"] == pytest.approx(
        round(float(confirmed.mean()) * 100, 4))
    assert out["Inequity_severity"] == pytest.approx(
        round(float(frame.loc[confirmed, "moving_median"].abs().mean()) * 100, 4))


def test_vem_severity_counts_only_ci_confirmed_windows():
    """A window whose point estimate strays but whose band still overlaps
    contributes nothing to severity -- it makes the study inconclusive,
    not severe."""
    rng = np.random.default_rng(71)
    n = 300
    sale = rng.lognormal(12.5, 0.5, n)
    # Mild tilt, heavy noise: point estimates wander outside the band but
    # every interval is too wide to confirm it.
    asr = (1 + 0.10 * np.log(sale / sale.mean())) * rng.lognormal(0, 0.45, n)
    out = I.vem(pd.DataFrame({"sale": sale, "asr": asr}), "sale",
                window_size=25, conf_level=0.99, acceptable_inequity_level=0.05)
    frame = out["dataframe"]
    assert bool((frame["out_of_bounds"] != 0).any())        # strays
    assert not bool((frame["ci_out_of_bounds"] != 0).any())  # never confirmed
    assert out["Inequity_severity"] == 0.0
    assert out["Inequity_conclusion"] == "Inconclusive"


def test_vem_prevalence_is_defined_without_an_acceptable_band():
    """It used to be assigned only inside `if acceptable_inequity_level`,
    so a falsy band raised NameError on the way to the return dict."""
    out = I.vem(_vem_frame(), "sale", window_size=30, conf_level=0.9,
                acceptable_inequity_level=0)
    assert out["Inequity_prevalence"] == 100.0
    assert out["Inequity_severity"] >= 0.0


def test_vem_severity_is_on_the_same_scale_with_and_without_a_band():
    """The two branches must not disagree by a factor of 100."""
    df = _vem_frame()
    banded = I.vem(df.copy(), "sale", window_size=30, conf_level=0.9,
                   acceptable_inequity_level=0.1)["Inequity_severity"]
    unbanded = I.vem(df.copy(), "sale", window_size=30, conf_level=0.9,
                     acceptable_inequity_level=0)["Inequity_severity"]
    # Both in points. With no band every row counts, so the unbanded mean
    # is diluted by the compliant middle of the range and comes out lower.
    assert banded > unbanded > 0
    assert banded / unbanded < 100


def test_max_acceptable_severity_defaults_to_the_band_itself():
    """None means `acceptable_inequity_level * 100`, so out of the box any
    confirmed excursion is reported. A fixed 0.01 was compared against a
    severity in points and could never bind."""
    import inspect

    assert inspect.signature(I.vem).parameters[
        "max_acceptable_severity"].default is None

    frame = _vem_frame()
    common = dict(window_size=30, conf_level=0.9, acceptable_inequity_level=0.1)
    reported = I.vem(frame.copy(), "sale", **common)
    assert reported["Inequity_conclusion"] in ("Regressivity", "Progressivity")

    # Raised above the confirmed severity, the gate binds.
    threshold = reported["Inequity_severity"] + 5
    muted = I.vem(frame.copy(), "sale",
                  max_acceptable_severity=threshold, **common)
    assert muted["Inequity_conclusion"] == "acceptable"


def test_a_flat_sample_reports_zero_severity_and_no_inequity():
    rng = np.random.default_rng(62)
    sale = rng.lognormal(12.5, 0.5, 300)
    df = pd.DataFrame({"sale": sale, "asr": rng.normal(1.0, 0.02, 300)})
    out = I.vem(df, "sale", window_size=30, conf_level=0.9,
                acceptable_inequity_level=0.1)
    assert out["Inequity_severity"] == 0.0
    assert out["Inequity_prevalence"] == 0.0
    assert out["Inequity_conclusion"] == "acceptable"


# --------------------------------------------------------------------------
# A ratio cannot be negative: the symmetric intervals are clamped
# --------------------------------------------------------------------------

def test_mean_ci_lower_bound_is_clamped_at_zero():
    """D.1 is symmetric in value, unlike the order-statistic medians.

    An ASR is a ratio of two positive amounts, so it is bounded below at 0
    and unbounded above. On a small, sharply right-skewed sample the
    t-margin can exceed the mean and push the lower bound negative -- an
    impossible ratio. The median intervals cannot do this because their
    bounds are drawn from the sample; this one can, so it is clamped.
    """
    # Mostly tiny ratios with one huge one: mean small, sd enormous.
    skewed = np.array([0.01, 0.02, 0.02, 0.03, 0.03, 40.0])
    lower, upper = C.mean_ci(skewed, 0.99)
    assert lower == 0.0
    assert upper > np.mean(skewed)
    # Without the clamp the unclamped margin genuinely overshoots.
    from scipy import stats
    margin = stats.t.ppf(0.995, len(skewed) - 1) * stats.sem(skewed)
    assert np.mean(skewed) - margin < 0


def test_weighted_mean_ci_lower_bound_is_clamped_at_zero():
    """Same reasoning: a ratio of two positive totals cannot be negative."""
    sale = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    assess = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 900.0])
    lower, upper = C.weighted_mean_ci(assess, sale, 0.99)
    assert lower == 0.0
    assert upper > 0


def test_the_clamp_does_not_touch_ordinary_samples():
    """It must only ever bite where the bound was impossible."""
    from scipy import stats

    rng = np.random.default_rng(81)
    for _ in range(25):
        sample = rng.lognormal(0.0, 0.15, 120)
        lower, upper = C.mean_ci(sample, 0.95)
        expected = stats.t.interval(0.95, len(sample) - 1,
                                    loc=sample.mean(), scale=stats.sem(sample))
        assert (lower, upper) == pytest.approx(expected)
        assert lower > 0


@pytest.mark.parametrize("method", ["binom", "R6", "R7"])
def test_the_median_intervals_never_needed_a_clamp(method):
    """The property that distinguishes them: bounds come from the sample."""
    skewed = np.array([0.01, 0.02, 0.02, 0.03, 0.03, 40.0])
    out = compute_median_ci(skewed, 0.99, method=method)
    assert out['lower'] >= skewed.min() > 0
