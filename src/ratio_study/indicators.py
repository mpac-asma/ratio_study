import numpy as np
import pandas as pd
import narwhals as nw
from typing import Optional, Union
from scipy.stats import shapiro, kruskal
from statsmodels.stats.weightstats import DescrStatsW
import math
from ratio_study.helpers import _simple_regression, _ols_slope, _get_group_sizes, _determine_num_groups
from ratio_study.graphs import vei_boxplot, ve_scat_median_plot
from ratio_study.confint import (compute_median_ci, log_median_diff_ci,
                                 confidence_level_as_fraction)
from . import nw_utils, _optional
import warnings

VEI_ACCEPTABLE = 'Acceptable'
VEI_SMALL_SAMPLE = 'Sample size too small'

VEI_INEQUITY_CONCLUSIONS = ('Regressivity', 'Progressivity')
VEI_CONCLUSIONS = (VEI_ACCEPTABLE,) + VEI_INEQUITY_CONCLUSIONS + (VEI_SMALL_SAMPLE,)

def _as_center(asr_md, caller):
    """Coerce a caller-supplied median to a float, rejecting arrays.

    ``cod`` and ``coc`` take ``asr_md`` in the second positional slot, so
    ``cod(asr, weights)`` binds the weight *array* to the centre and used to
    return a plausible wrong number in silence. ``weights`` is keyword-only
    now, which stops it being passed in the slot after ``asr_md``, but not
    this case -- an explicit check is the only way to catch it.
    """
    if asr_md is None:
        return None
    if np.ndim(asr_md) != 0:
        raise TypeError(
            f"{caller}(): asr_md must be a scalar median, got an array of shape "
            f"{np.shape(asr_md)}. Did you mean {caller}(..., weights=...)? "
            "`weights` is keyword-only."
        )
    return float(asr_md)


def asr_median(asr, weights=None):
    """
    Compute the (optionally weighted) median of an array of Assessment-to-Sales Ratios.

    Parameters
    ----------
    asr : array-like
        ASR - Assessment to Sales Ratio (numpy, pandas, polars, or
        narwhals Series).
    weights : array-like, optional
        Weights, assumed to have mean = 1. If provided, the weighted
        median of ``asr`` is returned. Only relative magnitudes matter --
        the result is invariant to rescaling the weights.

    Returns
    -------
    float
        The median (or weighted median) of ``asr``.
    """
    try:
        asr_arr = nw_utils.to_numpy_1d(asr).astype(float)
        if weights is None:
            return float(np.median(asr_arr))

        w = nw_utils.to_numpy_1d(weights).astype(float)
        order = np.argsort(asr_arr)
        asr_sorted = asr_arr[order]
        cum = np.cumsum(w[order])
        half = 0.5 * cum[-1]
        # An exact tie -- cumulative weight landing precisely on half the total
        # -- means every value between asr_sorted[k] and asr_sorted[k + 1]
        # splits the weight evenly, so both are equally valid medians and
        # floating-point noise alone would otherwise decide which one is
        # returned. Averaging the pair resolves it deterministically and makes
        # uniform weights reproduce ``np.median`` on even-length samples.
        #
        # The tolerance is applied on both sides of the search: ``cum`` is a
        # running sum, so a tie can land a few ULPs either side of ``half``
        # and a bare ``searchsorted(cum, half)`` would step over it.
        tol = 1e-9 * cum[-1]
        k = int(np.searchsorted(cum, half - tol))
        if k + 1 < asr_sorted.size and cum[k] <= half + tol:
            return float(0.5 * (asr_sorted[k] + asr_sorted[k + 1]))
        return float(asr_sorted[k])
    except Exception as ex:
        raise RuntimeError('error in ratio study: asr_median:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def asr_mean(asr, weights=None):
    """
    Compute the (optionally weighted) mean of an array of Assessment-to-Sales Ratios.

    Parameters
    ----------
    asr : array-like
        ASR - Assessment to Sales Ratio (numpy, pandas, polars, or
        narwhals Series).
    weights : array-like, optional
        Weights, assumed to have mean = 1. If provided, the weighted
        mean of ``asr`` is returned.

    Returns
    -------
    float
        The mean (or weighted mean) of ``asr``.
    """
    try:
        asr_arr = nw_utils.to_numpy_1d(asr).astype(float)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(float)
            # Use DescrStatsW with a numpy array (it accepts ndarray).
            return float(DescrStatsW(asr_arr, weights=weights).mean)
        else:
            return float(np.mean(asr_arr))
    except Exception as ex:
        raise RuntimeError('error in ratio study: asr_mean:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def cod(asr, asr_md=None, *, weights=None):
    """
    Coefficient of Dispersion.

    For performance purposes, uses ``asr_md`` as input, since it is
    already calculated in other steps, so the result is re-used instead
    of re-calculated in the COD formula.

    Parameters
    ----------
    asr : array-like
        ASR - Assessment to Sales Ratio (numpy/pandas/polars/narwhals
        Series).
    asr_md : float or Decimal, optional
        Median of ``asr``. If ``weights`` are used, this is assumed to
        be the weighted median of ``asr``. If None, it is computed
        internally via ``asr_median``.
    weights : array-like, optional
        Weights, assumed to have mean = 1.

    Returns
    -------
    float
        The (optionally weighted) coefficient of dispersion.
    """
    asr_md = _as_center(asr_md, 'cod')
    try:
        asr_arr = nw_utils.to_numpy_1d(asr).astype(float)
        if asr_md is None:
            asr_md = asr_median(asr, weights=weights)

        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(float)
            return float(np.average(np.abs(asr_arr - asr_md) / asr_md, weights=weights))
        else:
            return float(np.mean(np.abs(asr_arr - asr_md) / asr_md))
    except Exception as ex:
        raise RuntimeError('error in ratio study: cod:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def cod_residual(market_value, assessment, weights=None):
    """
    Coefficient of Dispersion computed directly from residuals.

    Mirrors ``cod``, but works from the raw market value and assessment
    arrays rather than a precomputed ASR array.

    Parameters
    ----------
    market_value : array-like
        Time-adjusted sale amount.
    assessment : array-like
        Assessed value.
    weights : array-like, optional
        Weights, assumed to have mean = 1.

    Returns
    -------
    float
        The (optionally weighted) coefficient of dispersion computed
        from residuals.
    """
    try:
        if weights is not None:
            return np.average(np.abs(assessment - market_value), weights=weights)/np.average(market_value, weights=weights)
        else:
            return np.mean(np.abs(assessment - market_value))/np.mean(market_value)
    except Exception as ex:
        raise RuntimeError('error in ratio study: cod:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def prd(market_value, asr, weights=None):
    """
    Price-Related Differential (PRD).

    Parameters
    ----------
    market_value : array-like
        Time-adjusted sale amount (numpy/pandas/polars/narwhals Series).
    asr : array-like
        ASR - Assessment to Sales Ratio (numpy/pandas/polars/narwhals
        Series).
    weights : array-like, optional
        Weights, assumed to have mean = 1.

    Returns
    -------
    float
        The (optionally weighted) price-related differential.
    """
    try:
        asr_arr = nw_utils.to_numpy_1d(asr).astype(float)
        mv_arr = nw_utils.to_numpy_1d(market_value).astype(float)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(float)
            mean_ratio = np.average(asr_arr, weights=weights)
            dollar_weighted_ratio = np.average(asr_arr, weights=mv_arr * weights)
            return float(mean_ratio / dollar_weighted_ratio)
        else:
            return float(np.mean(asr_arr) / np.average(a=asr_arr, weights=mv_arr))
    except Exception as ex:
        raise RuntimeError('error in ratio study: prd:', ex)

# ---------------------------------------------------------------------------------------------------------------------

# Default share of the sale price in the market value proxy. 50/50 is what
# the Standard's PRB and the VEI both assume, so it is the package-wide
# default everywhere `mvp_sale_price_share` appears.
DEFAULT_MVP_SALE_PRICE_SHARE = 0.5


def mvp_sale_price_share_as_fraction(mvp_sale_price_share):
    """
    Validate a market value proxy sale-price share and return it as a fraction.

    Accepts the share either as a fraction (``0.3``) or as a percentage
    (``30``), mirroring how `confidence_level_as_fraction` treats confidence
    levels, so ``mvp_sale_price_share=30`` means a 30/70 proxy rather than a
    nonsensical weight of 30.

    Parameters
    ----------
    mvp_sale_price_share : float
        Share of the sale price in the proxy, in ``[0, 1]`` or ``(1, 100]``.

    Returns
    -------
    float
        The share as a fraction in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the share is not a number in ``[0, 1]`` or ``(1, 100]``.
    """
    try:
        share = float(mvp_sale_price_share)
    except (TypeError, ValueError):
        raise ValueError(
            f'mvp_sale_price_share must be a number, got {mvp_sale_price_share!r}.')
    if not np.isfinite(share):
        raise ValueError(
            f'mvp_sale_price_share must be finite, got {mvp_sale_price_share!r}.')
    if 1 < share <= 100:
        share = share / 100.0
    if not 0 <= share <= 1:
        raise ValueError(
            'mvp_sale_price_share must be a fraction in [0, 1] (or a percentage '
            f'in (1, 100]), got {mvp_sale_price_share!r}.')
    return share


def market_value_proxy(market_value, assessment, asr=None, asr_md=None,
                       mvp_sale_price_share=DEFAULT_MVP_SALE_PRICE_SHARE,
                       *, weights=None):
    """
    Blend the sale price and the level-adjusted assessment into a value proxy.

    The proxy is the single definition used everywhere in the package that
    needs a measure of value free of the sale price's own noise -- the PRB
    regression's independent variable, the VEI's grouping variable, and the
    optional VEM sort order::

        share * sale price + (1 - share) * (assessment / median ASR)

    Dividing the assessment by the median ASR restates it at the overall
    level of assessment, so both halves are on the same scale as the sale
    price. The Standard's 50/50 blend is the default; `mvp_sale_price_share`
    exists to lean the proxy further onto the assessment (``0.3`` for a 30/70
    proxy) when sale prices are the noisier of the two, or further onto the
    sale price when they are not.

    Parameters
    ----------
    market_value : array-like
        Time-adjusted sale amount (numpy/pandas/polars/narwhals Series).
        Returned as-is when ``mvp_sale_price_share`` is 1.
    assessment : array-like
        Predicted (assessed) values, same length as `market_value`.
    asr : array-like, optional
        Assessment to sales ratios, used only to derive `asr_md` when that
        is not supplied. Computed as ``assessment / market_value`` if both
        are omitted.
    asr_md : float, optional
        Median ASR (the level of assessment). Computed from `asr` via
        :func:`asr_median` when None. Pass the study's own median so the
        proxy cannot drift from the ratios table.
    mvp_sale_price_share : float, default 0.5
        Share of the sale price in the proxy, as a fraction in ``[0, 1]``
        or a percentage in ``(1, 100]``. The assessment half gets the
        complement, so ``0.5`` is the Standard's 50/50 proxy and ``0.3``
        a 30/70 one.
    weights : array-like, optional
        Weights, assumed to have mean = 1. Only used when `asr_md` has to
        be computed.

    Returns
    -------
    array-like
        The proxy, in the same container arithmetic on the inputs produces
        -- a pandas/polars Series in, a Series out; numpy arrays in, a
        numpy array out.

    Raises
    ------
    ValueError
        If `mvp_sale_price_share` is outside the accepted range, or if
        neither `asr` nor `asr_md` can be resolved.

    Examples
    --------
    >>> import numpy as np
    >>> sale = np.array([100.0, 200.0, 300.0])
    >>> assessed = np.array([95.0, 210.0, 280.0])
    >>> market_value_proxy(sale, assessed, asr_md=1.0)
    array([ 97.5, 205. , 290. ])
    >>> market_value_proxy(sale, assessed, asr_md=1.0, mvp_sale_price_share=0.3)
    array([ 96.5, 207. , 286. ])
    """
    share = mvp_sale_price_share_as_fraction(mvp_sale_price_share)

    if share == 1:
        return market_value

    if asr_md is None:
        if asr is None:
            if market_value is None or assessment is None:
                raise ValueError(
                    'market_value_proxy() needs `asr` or `asr_md`, or both '
                    '`market_value` and `assessment` to derive the ratios from.')
            asr = nw_utils.to_numpy_1d(assessment).astype(float) / nw_utils.to_numpy_1d(market_value).astype(float)
        asr_md = asr_median(asr, weights=weights)
    asr_md = _as_center(asr_md, 'market_value_proxy')

    if not asr_md:
        raise ValueError(
            'market_value_proxy() cannot restate the assessment at a median '
            f'ASR of {asr_md!r}.')

    level_adjusted = assessment / asr_md
    if share == 0:
        return level_adjusted
    return market_value * share + level_adjusted * (1.0 - share)


# `prb`'s `market_value_proxy` parameter shadows the function above inside its
# body, so internal callers reach it through this alias instead.
_market_value_proxy = market_value_proxy

# ---------------------------------------------------------------------------------------------------------------------

def prb(market_value, assessment, asr, market_value_proxy=None, asr_md=None, req_significance=0.05, non_significant_marker='*', *, weights=None, ci_conf_level=None, mvp_sale_price_share=DEFAULT_MVP_SALE_PRICE_SHARE):
    """
    Compute the Price-Related Bias (PRB).

    The significance test and the confidence interval are two separate
    questions and are controlled separately. `req_significance` decides
    whether the slope is called significant -- it is the p-value threshold
    -- while `ci_conf_level` sets the width of the interval reported alongside
    it. Left at its default the interval follows `req_significance`, which
    reproduces the historical behaviour (a 5 % threshold giving a 95 %
    interval); pass `ci_conf_level` to have the interval match the rest of a
    study's intervals without moving the significance threshold.

    Parameters
    ----------
    market_value : array-like
        Time-adjusted sale amount.
    assessment : array-like
        Predicted (assessed) values.
    asr : array-like
        ASR - Assessment to Sales Ratio.
    market_value_proxy: array-like
        Optional proxy for ``market_value`` to use in regression.
        If provided, it overrides the proxy
        :func:`market_value_proxy` would otherwise build, and
        ``mvp_sale_price_share`` is then unused.
    asr_md : float or Decimal, optional
        Median of ``asr``. If None, it is computed internally via
        ``asr_median``.
    req_significance : float, default 0.05
        Required significance level (p-value) for the slope test, e.g.
        0.05 for 5%. If falsy (e.g. None or 0), the significance test is
        skipped and only ``prb_value`` is computed -- and, since there is
        then no regression to read an interval off, no ``prb_ci`` either.
    non_significant_marker : str, default '*'
        Marker appended to ``prb_value`` when the slope's p-value exceeds
        ``req_significance`` (i.e. the bias is not statistically
        significant). Set to a falsy value to disable the marker.
    weights : array-like, optional
        Weights, assumed to have mean = 1.
    ci_conf_level : float, optional
        Confidence level for ``prb_ci``, as a fraction (``0.90``) or a
        percentage (``90``). Defaults to ``1 - req_significance``, so a
        study that leaves both alone still gets a 95 % interval. Setting
        it does not change the significance test or the marker.
    mvp_sale_price_share : float, default 0.5
        Share of the sale price in the market value proxy the regression
        runs on; see :func:`market_value_proxy`. Ignored when
        `market_value_proxy` is supplied.

    Returns
    -------
    tuple
        * prb_value : float or str -- the PRB coefficient, optionally
          suffixed with ``non_significant_marker``; None if the
          regression could not be computed.
        * prb_sig_flag : bool -- True when the slope's p-value exceeds
          ``req_significance`` (i.e. the PRB is not statistically
          significant); False otherwise, including when
          ``req_significance`` is falsy.
        * prb_ci : tuple or None -- confidence interval bounds for the
          PRB slope at `ci_conf_level`, or None if not computed.
    """
    try:
        prb_value = None
        prb_ci = None
        prb_sig_flag = False
        if asr_md is None:
            asr_md = asr_median(asr, weights=weights)

        asr_md = float(asr_md)
        pct_diff = (asr - asr_md) / asr_md
        if market_value_proxy is not None:
            value = market_value_proxy
        else:
            value = _market_value_proxy(market_value, assessment, asr_md=asr_md,
                                        mvp_sale_price_share=mvp_sale_price_share)
        # log base 2, as the PRB is read per doubling of value. Kept as the
        # 0.693 divisor rather than np.log2 so the published coefficients do
        # not move by the last decimal. A zero or negative `value` logs to
        # NaN by design -- `_simple_regression` / `_ols_slope` mask those
        # rows out below -- so the warning numpy raises for it is expected
        # and suppressed rather than left to alarm the caller.
        with np.errstate(invalid='ignore', divide='ignore'):
            ln_value = np.log(value) / 0.693
        if pct_diff.size != np.sum(np.isnan(pct_diff)) and ln_value.size != np.sum(np.isnan(ln_value)):
            if req_significance:
                # The interval's alpha and the test's alpha are decoupled:
                # `ci_alpha` only ever reaches ``model.conf_int``, while the
                # p-value below is still compared against
                # `req_significance`.
                ci_alpha = (req_significance if ci_conf_level is None
                            else 1.0 - confidence_level_as_fraction(ci_conf_level, 'ci_conf_level'))
                results = _simple_regression(pct_diff, ln_value, weights=weights, ci_alpha=ci_alpha)
                prb_ci = tuple(results['slope']['beta_ci'][1])
                prb_value = round(results['slope']['beta'], 5)
                if results['slope']['pvalue'] > req_significance:
                    prb_sig_flag = True
                    if non_significant_marker:
                        prb_value = str(prb_value) + non_significant_marker
                    else:
                        prb_value = round(results['slope']['beta'], 5)
            else:
                # No significance test asked for, so nothing but the slope is
                # needed and the closed form gets it for a fraction of the
                # cost. This is the path the bootstrap resampler takes.
                prb_value = round(_ols_slope(pct_diff, ln_value, weights=weights), 5)

            return prb_value, prb_sig_flag, prb_ci
        else:
            return prb_value, prb_sig_flag, prb_ci
    except Exception as ex:
        raise RuntimeError('error in ratio study: prb:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def rmse(market_value, assessment, metric=np.mean, log=False, *, weights=None):
    """
    Root Mean Squared Error (RMSE), an accuracy metric for evaluating
    relative model quality; the standard deviation of the residuals
    (prediction errors).

    Parameters
    ----------
    market_value : array-like
        Time-adjusted sale amount (numpy/pandas/polars/narwhals Series).
    assessment : array-like
        Predicted values (numpy/pandas/polars/narwhals Series).
    metric : callable, default np.mean
        Aggregation function applied to the squared residuals before
        taking the square root. ``np.median`` can be used instead to
        compute the root-median-square error. Ignored when ``weights``
        is provided (a weighted average is used in that case).
    log : bool, default False
        If True, take the natural log of ``market_value`` and
        ``assessment`` (each plus 1) before computing the error.
    weights : array-like, optional
        Weights, assumed to have mean = 1.

    Returns
    -------
    float
        The (optionally weighted) root mean squared error.
    """
    if not callable(metric):
        raise TypeError(
            "rmse(): metric must be a callable aggregator such as np.mean or "
            f"np.median, got {type(metric).__name__}. Did you mean "
            "rmse(..., weights=...)? `weights` is keyword-only."
        )
    try:
        assessment_arr = nw_utils.to_numpy_1d(assessment)
        market_value_arr = nw_utils.to_numpy_1d(market_value)
        if log:
            assessment_arr = np.log(assessment_arr + 1)
            market_value_arr = np.log(market_value_arr + 1)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights)
            return float(np.sqrt(np.average(np.square(market_value_arr - assessment_arr), weights=weights)))
        else:
            # RMSE = sqrt(metric((market_value - assessment)^2)).  ``metric``
            # is typically ``np.mean`` but can be ``np.median`` to compute
            # the root-median-square error instead.
            return float(math.sqrt(metric(np.square(np.subtract(market_value_arr, assessment_arr)))))
    except Exception as ex:
        raise RuntimeError('error in ratio study: rmse:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def mae(market_value, assessment, weights=None):
    """
    Mean Absolute Error (MAE).

    Parameters
    ----------
    market_value : array-like
        Time-adjusted sale amount (numpy/pandas/polars/narwhals Series).
    assessment : array-like
        Predicted values (numpy/pandas/polars/narwhals Series).
    weights : array-like, optional
        Weights, assumed to have mean = 1. Only relative magnitudes
        matter -- the result is invariant to rescaling the weights.

    Returns
    -------
    float
        The (optionally weighted) mean absolute error.
    """
    try:
        assessment_arr = nw_utils.to_numpy_1d(assessment)
        market_value_arr = nw_utils.to_numpy_1d(market_value)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(float)
            return float(np.average(np.abs(np.subtract(assessment_arr, market_value_arr)), weights=weights))
        else:
            return float(np.mean(np.abs(np.subtract(assessment_arr, market_value_arr))))
    except Exception as ex:
        raise RuntimeError('error in ratio study: mae:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def mape(sale_error, weights=None):
    """
    Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    sale_error : array-like
        Sale error, computed as ``(assessment - market_value) / market_value``.
    weights : array-like, optional
        Weights, assumed to have mean = 1. Only relative magnitudes
        matter -- the result is invariant to rescaling the weights.

    Returns
    -------
    float
        The (optionally weighted) mean absolute percentage error.
    """
    try:
        sale_error_arr = nw_utils.to_numpy_1d(sale_error)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(float)
            return float(np.average(np.abs(sale_error_arr), weights=weights))
        else:
            return float(np.mean(np.abs(sale_error_arr)))
    except Exception as ex:
        raise RuntimeError('error in ratio study: mape:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def mdmape(sale_error, weights=None):  # change to MdMAPE
    """
    Median Absolute Percentage Error (MdMAPE).

    Parameters
    ----------
    sale_error : array-like
        Sale error, computed as ``(predict - market_value) / market_value``.
    weights : array-like, optional
        Weights, assumed to have mean = 1.

    Returns
    -------
    float
        The (optionally weighted) median absolute percentage error.
    """

    try:
        mape = asr_median(np.abs(sale_error), weights=weights)
        return mape
    except Exception as ex:
        raise RuntimeError('error in ratio study: mdmape:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def fsd(sale_error, weights=None):
    """
    Forecast Standard Deviation (FSD).

    The FSD is an estimate of the amount of variation that can occur
    between the actual sales price and the forecast (the most probable
    market value) made by the AVM.

    Note: because this interpretation of an FSD assumes that the sales
    errors are normally distributed, AVM providers should verify this
    normality assumption using a hypothesis test, such as the
    Shapiro-Wilk test.

    Parameters
    ----------
    sale_error : array-like
        Sale error, computed as ``(predict - market_value) / market_value``.
    weights : array-like, optional
        Weights, assumed to have mean = 1.

    Returns
    -------
    str
        The FSD formatted as a percentage, followed by the Shapiro-Wilk
        normality test p-value in parentheses, e.g. ``'12.34% (0.045)'``.

    Notes
    -----
    The FSD itself honours ``weights``, but the Shapiro-Wilk p-value is
    always computed on the unweighted ``sale_error``; scipy's ``shapiro``
    has no weighted form. Read the p-value as a statement about the
    sample, not about the weighted population.
    """
    # normality test (Shapiro-Wilk Test)
    try:
        sale_error = nw_utils.to_numpy_1d(sale_error)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights)
            FSD = DescrStatsW(sale_error, weights=weights).std
        else:
            FSD = round(np.std(sale_error), 4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            stat, p = shapiro(sale_error)

        return '{:.2%}'.format(FSD) + ' (' + str("%.3f" % p) + ')'
    except Exception as ex:
        raise RuntimeError('error in ratio study: fsd:', ex) from ex

# ---------------------------------------------------------------------------------------------------------------------

def cov_adj(asr, weights=None):
    """
    Coefficient of variation, adjusted by Fisher's small-sample
    correction factor (1 + 1/(4n)).

    Uses sample standard deviation (n-1 divisor) per IAAO Standard on
    Ratio Studies.

    Parameters
    ----------
    asr : array-like
        ASR - Assessment to Sales Ratio (numpy/pandas/polars/narwhals
        Series).
    weights : array-like, optional
        Weights, assumed to have mean = 1. Treated as sampling weights:
        Kish's effective sample size drives both the variance divisor and
        Fisher's correction, so the result is invariant to rescaling the
        weights. With uniform weights ``n_eff == n`` and the weighted path
        reduces exactly to the unweighted one.

    Returns
    -------
    float
        The (optionally weighted) coefficient of variation, adjusted by
        the Fisher small-sample correction factor. NaN when the effective
        sample size is too small (<= 1) to support an n-1 divisor.
    """
    try:
        asr = nw_utils.to_numpy_1d(asr).astype(float)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(float)
            n_eff = float((weights.sum() ** 2) / (weights ** 2).sum())
            if n_eff <= 1:
                return float('nan')
            mean = float(np.average(asr, weights=weights))
            # Sample variance under sampling weights: scale the weighted
            # population variance by n_eff/(n_eff - 1). Using ``Sum(w) - 1``
            # here instead -- as DescrStatsW.std_ddof(1) does -- would mix
            # frequency-weight semantics into a Kish n_eff and make the
            # result depend on the absolute scale of the weights.
            var = float(np.average((asr - mean) ** 2, weights=weights)) * n_eff / (n_eff - 1)
            std = math.sqrt(var)
        else:
            std = np.std(asr, ddof=1)
            mean = np.mean(asr)
            n_eff = float(len(asr))

        cov = std / abs(mean)
        adj_factor = 1 + (1 / (4 * n_eff))
        return float(cov * adj_factor)
    except Exception as ex:
        raise RuntimeError('error in ratio study: cov_adj:', ex) from ex

# ---------------------------------------------------------------------------------------------------------------------

def coc(asr, asr_md=None, within_prcnt=0.15, *, weights=None):
    """
    Coefficient of Concentration: percentage of ratios within
    ``within_prcnt`` of the median (multiplicatively).

    Per IAAO: "the percentage of observations within a specified
    percentage of a measure of central tendency."

    Parameters
    ----------
    asr : array-like
        ASR - Assessment to Sales Ratio (numpy/pandas/polars/narwhals
        Series).
    asr_md : float, optional
        Pre-computed median (weighted, if ``weights`` given). If None,
        it is computed internally via ``asr_median``.
    within_prcnt : float, default 0.15
        Fractional band around the median, e.g. 0.15 means +/-15% of
        the median.
    weights : array-like, optional
        Weights, assumed to have mean = 1.

    Returns
    -------
    float
        Fraction, in [0, 1], of observations falling within the band.
    """
    asr_md = _as_center(asr_md, 'coc')
    try:
        asr = nw_utils.to_numpy_1d(asr).astype(float)
        if asr_md is None:
            asr_md = asr_median(asr, weights=weights)

        lower = asr_md * (1 - within_prcnt)
        upper = asr_md * (1 + within_prcnt)
        in_band = (asr >= lower) & (asr <= upper)

        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(float)
            return float(weights[in_band].sum() / weights.sum())
        else:
            return float(in_band.sum() / len(asr))
    except Exception as ex:
        raise RuntimeError('error in ratio study: coc:', ex) from ex
# ---------------------------------------------------------------------------------------------------------------------

def moran_i(df, ratio, *, lat_col='latitude', long_col='longitude', k=None, permutations=999):
    """
    Global Moran's I: whether a ratio clusters spatially (nearby sales
    have similar ratios) rather than scattering at random across sale
    locations.

    Optional indicator -- requires the ``spatial`` extra::

        pip install "ratio_study[spatial]"

    Parameters
    ----------
    df : DataFrame
        Sales frame (pandas, polars, or PyArrow) holding ``ratio``,
        ``lat_col``, and ``long_col``.
    ratio : str
        Name of the ratio column to test for spatial autocorrelation
        (e.g. an ASR column).
    lat_col : str, default 'latitude'
        Name of the latitude column. Must exist in ``df``.
    long_col : str, default 'longitude'
        Name of the longitude column. Must exist in ``df``.
    k : int, optional
        Number of nearest neighbours for the KNN spatial weights
        matrix. Defaults to the cube root of the sample size (rounded),
        a common rule of thumb for KNN weights.
    permutations : int, default 999
        Number of conditional permutations used for the simulated
        p-value (``p_sim``/``z_sim``).

    Returns
    -------
    dict
        ``I`` (the statistic), ``EI`` (expected value under no spatial
        autocorrelation), ``p_norm``/``z_norm`` (normal approximation),
        ``p_rand``/``z_rand`` (randomization approximation),
        ``p_sim``/``z_sim`` (conditional permutation), and ``n`` (sample
        size used).
    """
    if not nw_utils.has_columns(df, [ratio, lat_col, long_col]):
        raise ValueError(
            f"moran_i(): df must contain '{ratio}', '{lat_col}', and '{long_col}' columns."
        )

    weights_mod = _optional.require('libpysal.weights', 'spatial', 'moran_i')
    esda = _optional.require('esda', 'spatial', 'moran_i')

    try:
        pdf = nw_utils.require_pandas(df, reason='moran_i uses libpysal/esda')
        coords = pdf[[long_col, lat_col]].to_numpy(dtype=float)
        values = pdf[ratio].to_numpy(dtype=float)

        n = len(pdf)
        if k is None:
            k = max(1, round(n ** (1 / 3)))

        kw = weights_mod.KNN.from_array(coords, k=k)
        kw.transform = 'r'

        moran = esda.Moran(values, kw, transformation='r', permutations=permutations)

        return {
            'I': float(moran.I),
            'EI': float(moran.EI),
            'p_norm': float(moran.p_norm),
            'z_norm': float(moran.z_norm),
            'p_rand': float(moran.p_rand),
            'z_rand': float(moran.z_rand),
            'p_sim': float(moran.p_sim),
            'z_sim': float(moran.z_sim),
            'n': n,
        }
    except Exception as ex:
        raise RuntimeError('error in ratio study: moran_i:', ex) from ex

# ---------------------------------------------------------------------------------------------------------------------


# Main VEI Analysis
def vei_analysis(asr,
                 mkt_value_proxy,
                 conf_level=0.9,
                 acceptable_range=10,
                 percentile_method='R7',
                 ci_method=None,  # if None, defaults to percentile_method
                 output_format='dict',  # or 'df'
                 metrics_only=True,
                 gen_plot=False,
                 plot_ref='',
                 plot_filepath='',
                 plot_return_object=False,
                 vei_use_log_normal_delta=True,
                 compare_all_buckets=False,
                 print_settings=True,
                 warnings=False):
    """
    Vertical Equity Indicator (VEI) -- Appendix E of the Standard on Ratio Studies.

    Splits the sales into quantile groups of the market value proxy and
    compares the median ratio of the highest-value group against the lowest,
    scaled by the overall median::

        VEI = 100 * (median(last group) - median(first group)) / median(all)

    **Negative is regressive, positive is progressive**, and +/-10% is the
    acceptable range. A point estimate outside that range only fails the
    standard once the confidence interval confirms it (Appendix E.7).

    Group counts follow Table E.3.1: halves for 20-50 sales, quartiles for
    51-500, deciles for 501 or more. Below 20 sales the VEI is undefined and
    the result carries the small-sample conclusion instead.

    Parameters
    ----------
    asr : array-like
        Assessment-to-sale ratios.
    mkt_value_proxy : array-like
        Market value proxy the groups are formed on. Appendix E.1 recommends
        an equal weighting of sale price and assessed value, to keep the
        errors-in-variables bias from tilting the result toward regressivity.
    conf_level : float, default 0.9
        Confidence level for the group median intervals and the VEI interval.
    acceptable_range : float, default 10
        Acceptance threshold in percentage points, per section 11.2.6.
    percentile_method : {'R7', 'R6'}, default 'R7'
        Quantile definition used to form the groups. Appendix E.5 notes the
        two can assign boundary observations differently and produce
        different VEIs when the sample does not divide evenly.
    ci_method : str, optional
        Median-CI family. Follows `percentile_method` when omitted.
    output_format : {'dict', 'df'}, default 'dict'
        Shape of the return value.
    metrics_only : bool, default True
        Return just the metrics. False also returns the per-group table.
    gen_plot : bool, default False
        Also build the VEI boxplot, under ``'vei_plot'``.
    plot_ref, plot_filepath, plot_return_object
        Passed to :func:`ratio_study.vei_boxplot`.
    vei_use_log_normal_delta : bool, default True
        Build the VEI interval with the log-normal delta method.
    compare_all_buckets : bool, default False
        Compare the two quantile groups with the widest gap in median ratio,
        instead of the first and last. Appendix E defines the VEI on the
        lowest- and highest-value groups, so this is a **diagnostic, not the
        acceptance test** -- it answers "where is the worst inequity in this
        range?", which Appendix E.4 asks the analyst to check by eye. Value
        order is preserved, so the sign still reads the same way. Ignored
        when there are only two groups, where it would change nothing.
    print_settings : bool, default True
        Print the group count and ranking method in use. Appendix E.5 makes
        which one ran part of the result. Switched off internally when the
        VEI is recomputed per bootstrap resample.
    warnings : bool, default False
        Warn when a group holds fewer than 10 sales.

    Returns
    -------
    dict or DataFrame
        ``vei``, ``vei_sig``, ``direction``, ``conclusion``, ``Median``,
        ``pg_stats`` (per-group detail), ``percentile_method``,
        ``ci_method``, ``acceptable_range``, ``conf_level``,
        ``kruskal_wallis``, ``compare_all_buckets``, ``n_groups``, and
        ``group_compared_low`` / ``group_compared_high`` (1-based).

    See Also
    --------
    ratio_study.vei_boxplot : the Appendix E.4 visualisation of these groups.
    ratio_study.vem : the moving-median measure, which covers the whole range.
    """
    try:
        vei_sig, conclusion = None, None
        # --- Force everything to NumPy arrays ---
        asr = nw_utils.to_numpy_1d(asr).astype(float)
        mkt_value_proxy = nw_utils.to_numpy_1d(mkt_value_proxy).astype(float)

        # The Standard on Ratio Studies (final, 2026) Appendix E fixes the
        # acceptable VEI point-estimate threshold at +/-10% regardless of
        # the median-CI confidence level. ``acceptable_range`` is therefore
        # decoupled from ``conf_level`` and defaults to 10 (percent units).
        # Per the standard's own software-defaults table (E.5.3), Python's
        # numpy defaults to R7; the median-CI family and the percentile
        # rank family match by default -- ``ci_method`` follows
        # ``percentile_method`` (both default to 'R7').
        if ci_method is None:
            ci_method = percentile_method
        percentile_method = str(percentile_method).upper()
        ci_method = str(ci_method).upper()

        sample_median = np.median(asr)

        # --- Sort both arrays by market value ---
        sorted_indices = np.argsort(mkt_value_proxy)
        sorted_asr = asr[sorted_indices]
        sorted_mkt_value_proxy = mkt_value_proxy[sorted_indices]

        n = len(asr)

        # --- Minimum sample size check ---
        if n < 20:
            vei_result = {
                "vei": 0,
                "vei_sig": 0,
                "direction": '',
                "conclusion": VEI_SMALL_SAMPLE,
                "Median": sample_median,
                "pg_stats": [],
                "kruskal_wallis": None,
                "percentile_method": percentile_method,
                "compare_all_buckets": bool(compare_all_buckets),
                "n_groups": 0,
                "group_compared_low": None,
                "group_compared_high": None,
            }
            return pd.DataFrame([vei_result]) if output_format == 'df' else vei_result

        # --- Determine groups ---
        m = _determine_num_groups(n)
        group_sizes = _get_group_sizes(n, m, percentile_method=percentile_method)

        if any(s < 10 for s in group_sizes) and warnings:
            print("⚠️ Warning: Some groups have fewer than 10 observations; results may be unreliable.")

        cum_ends = np.cumsum(group_sizes)
        cum_starts = np.concatenate(([0], cum_ends[:-1]))
        
        pg_stats = []
        for i in range(m):
            group_r = sorted_asr[cum_starts[i]:cum_ends[i]]
            ci = compute_median_ci(group_r, conf_level, method=ci_method)
            med_proxy = np.median(sorted_mkt_value_proxy[cum_starts[i]:cum_ends[i]])
            pg_stats.append({
                "sales": group_sizes[i],
                "med_proxy": med_proxy,
                "median_ratio": ci["median"],
                "lower_ci": ci["lower"],
                "upper_ci": ci["upper"],
                "achieved_coverage": ci["achieved_coverage"],
                # Within-group dispersion, as a fraction (0.15 = 15%). The
                # VEI boxplot labels each group with it, so the reader sees
                # the spread each group median is drawn from.
                "cod": cod(group_r, ci["median"]),
                "asr_vector" : group_r
            })

        # --- Kruskal-Wallis across all quantile groups ---
        # Appendix E.4 ("Further Analysis") recommends this as a supplementary
        # check of whether the level of assessment differs across *all* the
        # quantile groups, not just the first and last -- the VEI itself only
        # ever compares those two. It is not a pass/fail criterion for the
        # VEI test (the standard says it "may be run"), so it rides along on
        # every VEI result rather than gating anything.
        if m >= 2:
            kw_statistic, kw_pvalue = kruskal(*[g["asr_vector"] for g in pg_stats])
            kruskal_wallis = {"statistic": float(kw_statistic), "pvalue": float(kw_pvalue)}
        else:
            kruskal_wallis = None

        # --- Which two quantile groups to compare ---
        # Appendix E compares the first and last groups: the VEI is defined as
        # the gap between the lowest- and highest-value bands. `compare_all_buckets`
        # instead takes the widest gap anywhere in the range, which Appendix E.4
        # asks the analyst to look for by eye -- "if another quantile group has
        # the highest or lowest median ratio, it should be noted and reviewed".
        #
        # This is a *diagnostic*, not the acceptance test: it reports the worst
        # pair rather than the pair the Standard defines, so the number it
        # produces is not the VEI a acceptance report should quote.
        #
        # Value order is preserved (the lower-value group stays "first") so the
        # sign keeps its meaning: negative is still regressive.
        medians = [g["median_ratio"] for g in pg_stats]
        if compare_all_buckets and m > 2:
            # The widest pairwise gap is always between the min and the max.
            i_min, i_max = int(np.argmin(medians)), int(np.argmax(medians))
            first_idx, last_idx = sorted((i_min, i_max))
        else:
            first_idx, last_idx = 0, m - 1
        first_pg, last_pg = pg_stats[first_idx], pg_stats[last_idx]

        if compare_all_buckets and print_settings:
            lowest, highest = int(np.argmin(medians)) + 1, int(np.argmax(medians)) + 1
            print(
                f'VEI: compare_all_buckets is ON - comparing the widest gap between any '
                f'two of the {m} quantile groups, not just the first and last.\n'
                f'     Lowest median ratio : group {lowest} of {m} '
                f'({medians[lowest - 1]:.4f})\n'
                f'     Highest median ratio: group {highest} of {m} '
                f'({medians[highest - 1]:.4f})\n'
                f'     Comparing groups {first_idx + 1} and {last_idx + 1} in value order. '
                f'This is a diagnostic, not the IAAO standard acceptance test.'
            )
        elif print_settings:
            print(f'VEI: {m} quantile groups, {percentile_method} ranking '
                  f'(Appendix E.5); comparing groups 1 and {m}.')

        if vei_use_log_normal_delta:
            log_vei_result = log_median_diff_ci( low_ratios=first_pg["asr_vector"]
                                            , high_ratios=last_pg["asr_vector"]
                                            , overall_median=sample_median
                                            , conf_level=conf_level
                                            , threshold=acceptable_range)
            vei = log_vei_result['vei_pct']
            # The smallest inequity the interval still supports: the CI bound
            # nearest zero, or zero when the interval straddles it. Comparing
            # *that* against `acceptable_range` asks whether the whole
            # interval clears the threshold, which is the min-effect test
            # `log_median_diff_ci` reports as 'flag_min_effect'.
            #
            # This used to hold `vei_ci_upper - vei_ci_lower`, the interval's
            # *width*, so the test read "wide interval means inequity": a
            # large, tightly-estimated VEI was concluded Acceptable while a
            # noisy one was flagged. A -45.8% VEI with a CI of (-49.7, -41.8)
            # -- unambiguous regressivity -- came out 'Acceptable' because its
            # width, 7.9, fell under the 10-point threshold.
            lower_ci, upper_ci = log_vei_result['vei_ci_lower'], log_vei_result['vei_ci_upper']
            if lower_ci > 0:
                vei_sig = lower_ci
            elif upper_ci < 0:
                vei_sig = -upper_ci
            else:
                vei_sig = 0.0

        else:
            vei = 100 * (last_pg["median_ratio"] - first_pg["median_ratio"]) / sample_median

        direction = "Progressivity" if vei > 0 else "Regressivity" if vei < 0 else "Neutral"

        abs_vei = abs(vei)
        

        if abs_vei <= acceptable_range:
            conclusion = VEI_ACCEPTABLE
        else:
            if vei_use_log_normal_delta:
                conclusion = direction if vei_sig > acceptable_range else VEI_ACCEPTABLE
            else:
                # Same reading of `vei_sig` as the log-normal branch above:
                # the smallest inequity the intervals still support. Here
                # that is the gap between the two group intervals, which is
                # zero by definition once they overlap.
                ci_overlap = max(first_pg["lower_ci"], last_pg["lower_ci"]) <= min(first_pg["upper_ci"], last_pg["upper_ci"])
                if ci_overlap:
                    vei_sig = 0.0
                    conclusion = VEI_ACCEPTABLE
                else:
                    high_pg = last_pg if last_pg["median_ratio"] > first_pg["median_ratio"] else first_pg
                    low_pg = first_pg if high_pg is last_pg else last_pg
                    vei_sig = 100 * (high_pg["lower_ci"] - low_pg["upper_ci"]) / sample_median
                    conclusion = direction if vei_sig > acceptable_range else VEI_ACCEPTABLE

        vei_result = {
            "vei": vei,
            "vei_sig": vei_sig,
            "direction": direction,
            "conclusion": conclusion,
            "Median": sample_median,
            "pg_stats": pg_stats,
            "percentile_method": percentile_method,
            "ci_method": ci_method,
            "acceptable_range": acceptable_range,
            "conf_level": conf_level,
            "kruskal_wallis": kruskal_wallis,
            "compare_all_buckets": bool(compare_all_buckets),
            "n_groups": m,
            "group_compared_low": first_idx + 1,
            "group_compared_high": last_idx + 1,
        }

        # --- Optional plot ---

        if gen_plot:
            vei_result['vei_plot'] = vei_boxplot(asr, mkt_value_proxy, vei_result,
                                                 graph_filepath=plot_filepath,
                                                 title=plot_ref,
                                                 return_object=plot_return_object)

        # --- Output formatting ---
        if metrics_only:
            return pd.DataFrame([vei_result]) if output_format == 'df' else vei_result

        if output_format == 'df':
            group_stats_df = pd.DataFrame([{
                "Group": i + 1,
                "Sales": g["sales"],
                "Median_Proxy": g["med_proxy"],
                "Median_Ratio": g["median_ratio"],
                "Lower_CI": g["lower_ci"],
                "Upper_CI": g["upper_ci"],
                "COD": g["cod"]
            } for i, g in enumerate(pg_stats)])
            vei_result = pd.DataFrame([vei_result])
            result = {
                "vei_metrics": vei_result,
                "group_stats": group_stats_df
            }
        else:
            result = {
                "vei_metrics": vei_result,
                "group_stats": pg_stats
            }

        return result
 
    except Exception as ex:
        raise RuntimeError('error in ratio study: vei analysis:', ex)
    

def _validate_inputs(
    df,  # pandas/polars/pyarrow/narwhals DataFrame
    market_value_proxy: str,
    asr: str,
    conf_level: Optional[float],
    var_measure: Optional[str],
    quantiles: Optional[int],
    window_size: Optional[int],
    band_method: Optional[str] = "quantile",
    x_scale: str = "log",
    y_scale: str = "linear",
    asr_col='asr'
) -> None:
    """
    Validate the shared inputs of the vertical-equity indicators.

    ``conf_level``, ``var_measure`` and ``band_method`` are only
    meaningful for band-based measures; pass None for any of them (as
    ``vem`` does) to skip the corresponding check.
    """
    if not nw_utils.is_supported_frame(df):
        raise TypeError("`df` must be a pandas, polars, or narwhals DataFrame.")

    cols = nw_utils.frame_columns(df)
    for col, label in [
        (market_value_proxy, "market_value_proxy"),
        (asr, asr_col),
    ]:
        if col not in cols:
            raise ValueError(f"Column '{col}' not found in DataFrame ({label}).")
        # Check that the column is numeric.
        col_vals = nw_utils.to_numpy_1d(df[col] if not isinstance(df, nw.Series) else df)
        if not np.issubdtype(col_vals.dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric.")

    # One gate for every confidence level in the package, so the percentage
    # spelling works here exactly as it does for `ci_conf_level`, and a
    # significance level passed by mistake is rejected rather than honoured.
    if conf_level is not None:
        confidence_level_as_fraction(conf_level, 'conf_level')

    if var_measure is not None and var_measure not in ("sd", "mae", "cod"):
        raise ValueError("`var_measure` must be one of 'sd', 'mae', 'cod'.")

    if band_method is not None and band_method not in ("median_ci", "quantile"):
        raise ValueError("`band_method` must be 'median_ci' or 'quantile'.")

    if x_scale not in ("linear", "log"):
        raise ValueError("`x_scale` must be 'linear' or 'log'.")
    if y_scale not in ("linear", "log"):
        raise ValueError("`y_scale` must be 'linear' or 'log'.")

    if quantiles is None and window_size is None:
        raise ValueError(
            "At least one of `quantiles` or `window_size` must be provided."
        )

    if quantiles is not None and window_size is not None:
        warnings.warn(
            "Both `quantiles` and `window_size` provided; `quantiles` will be used.",
            UserWarning,
            stacklevel=3,
        )

    # Backend-agnostic uniqueness check.
    for col, label in [
        (asr, asr_col),
        (market_value_proxy, "market_value_proxy"),
    ]:
        col_vals = nw_utils.to_numpy_1d(df[col])
        unique_count = len(np.unique(col_vals[~np.isnan(col_vals)]))
        if unique_count < 2:
            if col == asr:
                raise ValueError(f"ASR column '{col}' is constant; cannot compute equity.")
            else:
                raise ValueError(
                    f"Market value proxy column '{col}' is constant."
                )

def _var_func(series, var_measure: str) -> float:
    """
    Scalar variability metric for a 1-D series.

    Parameters
    ----------
    series : array-like
        Values to summarize (numpy/pandas/polars/narwhals Series).
    var_measure : str
        Variability measure to compute; one of 'sd' (sample standard
        deviation), 'mae' (mean absolute error about the mean), or
        'cod' (coefficient of dispersion about the median).

    Returns
    -------
    float
        The requested variability metric, or NaN if ``series`` has no
        non-NaN values (or, for 'cod', if the median is zero).
    """
    s = nw_utils.to_numpy_1d(series).astype(float)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return np.nan
    if var_measure == "sd":
        return float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if var_measure == "mae":
        return float(np.mean(np.abs(s - s.mean())))
    # cod
    asr_md = float(np.median(s))
    if asr_md == 0:
        return np.nan
    return float(np.mean(np.abs(s - asr_md) / abs(asr_md)))


def _compute_moving_median(
    data,  # pandas DataFrame
    use_quantiles: bool,
    window_size: Optional[int],
    asr_col='asr',
    conf_level: Optional[float] = None,
    ci_method: str = "binom",
):
    """
    Compute the local (moving) median of the 'asr' column and add it to
    ``data`` as 'moving_median'.

    ``min_periods=1`` (window case) / every group (quantile case) is used
    so that every record keeps a value: the windows at either end of the
    market value range are simply computed from fewer observations rather
    than being dropped to NaN.

    Parameters
    ----------
    data : pandas DataFrame
        Working frame containing at least an 'asr' column (and a
        'quantile_group' column when ``use_quantiles`` is True). Rows are
        expected to be sorted by the market value proxy.
    use_quantiles : bool
        If True, the median is computed per quantile group (using the
        'quantile_group' column). If False, it is computed over a
        centred rolling window of size ``window_size``.
    window_size : int, optional
        Rolling-window size used when ``use_quantiles`` is False.
    conf_level : float, optional
        If provided, also compute a nonparametric confidence interval for
        the local median (via ``compute_median_ci``) at each group/window
        and add it as 'median_ci_lower' / 'median_ci_upper'. These bounds
        are order statistics of the local ASR sample, so -- unlike a
        normal-theory (SE-based) CI -- they cannot fall outside the range
        actually observed in the data. That matters here because ASR
        (e.g. assessed/sale - 1) is bounded below at -1 but unbounded
        above: a symmetric Wald-type CI could push a lower bound past -1
        in small, skewed, or high-variance windows, while an
        order-statistic CI never can. If None (default), only
        'moving_median' is computed, using the same fast
        rolling/groupby path as before.
    ci_method : {'binom', 'R6', 'R7'}, default 'binom'
        CI family passed through to ``compute_median_ci``. Only used when
        ``conf_level`` is given. 'binom' is the exact sign-test CI for the
        median rank; 'R6' / 'R7' place the bounds at the same rank
        distance and interpolate between adjacent order statistics. All
        three are intervals for the median and narrow as the window grows.

    Returns
    -------
    pandas DataFrame
        ``data`` with a new 'moving_median' column added, plus
        'median_ci_lower' / 'median_ci_upper' when ``conf_level`` is
        given.
    """
    if conf_level is None:
        if use_quantiles:
            data["moving_median"] = (
                data.groupby("quantile_group")[asr_col]
                    .transform("median")
            )
        else:
            data["moving_median"] = (
                data[asr_col]
                .rolling(window=window_size, center=True, min_periods=1)
                .median()
            )
        return data

    medians, lowers, uppers = [], [], []

    if use_quantiles:
        qg_arr = nw_utils.to_numpy_1d(data["quantile_group"])
        asr_vals = nw_utils.to_numpy_1d(data[asr_col])
        unique_qgs = np.unique(qg_arr)
        for _g in unique_qgs:
            mask = qg_arr == _g
            ci = compute_median_ci(asr_vals[mask], conf_level, method=ci_method)
            n = int(mask.sum())
            medians.extend([ci["median"]] * n)
            lowers.extend([ci["lower"]] * n)
            uppers.extend([ci["upper"]] * n)
    else:
        hw = window_size // 2
        asr_vals = nw_utils.to_numpy_1d(data[asr_col])
        total = len(asr_vals)
        for i in range(total):
            lo_idx = max(0, i - hw)
            hi_idx = min(total, i + hw + 1)
            ci = compute_median_ci(asr_vals[lo_idx:hi_idx], conf_level, method=ci_method)
            medians.append(ci["median"])
            lowers.append(ci["lower"])
            uppers.append(ci["upper"])

    data["moving_median"] = medians
    data["median_ci_lower"] = lowers
    data["median_ci_upper"] = uppers
    return data


def _select_masked(data, column, mask):
    """
    Return the values of ``column`` for rows where ``mask`` is True.

    Works on pandas / polars / narwhals frames.

    Parameters
    ----------
    data : pandas / polars / narwhals DataFrame
        Frame to select from.
    column : str
        Name of the column to extract.
    mask : array-like of bool
        Boolean mask selecting the rows to keep.

    Returns
    -------
    numpy.ndarray
        The values of ``column`` for the selected rows.
    """
    mask_arr = np.asarray(mask, dtype=bool)
    if hasattr(data, "loc"):
        # pandas
        return data.loc[mask_arr, column].to_numpy()
    if hasattr(data, "get_column"):
        # polars or pyarrow (via narwhals)
        col_vals = nw_utils.to_numpy_1d(data[column])
        return col_vals[mask_arr]
    # narwhals DataFrame
    if hasattr(data, "to_native"):
        native = data.to_native()
        return _select_masked(native, column, mask_arr)
    raise TypeError(f"Cannot select masked rows from {type(data).__name__}")

def vem(
    df,  # pandas/polars/pyarrow/narwhals DataFrame
    market_value_proxy: str,
    asr_col = 'asr',
    quantiles: Optional[int] = None,
    window_size: Optional[int] = None,
    conf_level: Optional[float] = 0.90,
    ci_method: str = "binom",
    max_acceptable_severity: Optional[float] = None,
    generate_plot = False,
    x_scale: str = "log",
    y_scale: str = "linear",
    plot_title: str = "Vertical Equity Moving Median",
    plot_return_object: bool = True,
    plot_ref: Optional[str] = None,
    plot_filepath: Optional[str] = None,
    acceptable_inequity_level: float = 0.1,
    ve_md_zoom_y=True,
    smart_zoom: bool = True,
    warnings=False) -> dict:
    """
    Compute a measure of vertical equity from the moving median of the ASR.

    This function works directly on the local (moving) median of the
    median-centred ASR: any stretch of the market
    value range whose local median sits further than
    ``acceptable_inequity_level`` from the jurisdiction-wide median is
    treated as vertical inequity.

    The centring is a calculation device only. ``severity_index``,
    ``prevalence_index`` and ``Inequity_conclusion`` are all computed on
    the zero-centred scale, but the plot is drawn in ASR units: the
    moving median, its CI band and the acceptable-inequity bands are all
    shifted back by the overall median ASR (returned as ``asr_center``),
    so the chart shows the ASR around its median -- typically, but not
    necessarily, around 1 -- rather than around 0.

    Parameters
    ----------
    df : pandas / polars / pyarrow / narwhals DataFrame
        Input data. Must contain ``market_value_proxy`` and a column
        literally named ``'asr'``.
    market_value_proxy : str
        Column name for the market value proxy (e.g. sale price).
    quantiles : int, optional
        Number of quantile groups. Takes precedence over ``window_size``.
    window_size : int, optional
        Rolling-window size used when ``quantiles`` is None.
    conf_level : float, optional, default 0.90
        Confidence level for a band drawn around the local (moving)
        median. The band comes from ``compute_median_ci()`` applied per
        quantile group / rolling window, with the raw lower/upper bounds
        kept, and it is what `ci_out_of_bounds` and the
        ``Inequity_conclusion`` are tested against -- not decoration.

        The bounds are order statistics of the local ASR sample (or an
        interpolation between two of them), so they can never fall outside
        the range actually observed. That matters because the ASR is
        asymmetric: it is bounded below -- at 0 for assessed/sale, or -1
        on the median-centred scale used here -- averages about 1, and has
        no upper bound at all. Equal rank distance either side of the
        median therefore maps to unequal *value* distance, automatically
        wider on the long right tail, which is the behaviour wanted. A
        symmetric normal-theory interval has no such guarantee and can put
        its lower bound past the floor on a small or skewed window.

        Pass None to skip the band and compute the moving median alone.
    ci_method : {'binom', 'R6', 'R7'}, default 'binom'
        CI family passed to ``compute_median_ci()`` when ``conf_level``
        is given. 'binom' is the exact sign-test interval for the median:
        it selects order-statistic ranks close to the middle of the sample
        and tightens as the group/window size grows. 'R6' and 'R7' place
        the bounds at the same rank distance but read the value off by
        interpolating between adjacent order statistics, following the
        percentile-rank presentation of Appendix D.2; all three narrow
        with the window size.
    max_acceptable_severity : float, optional
        How large a confirmed inequity has to be before it is reported, in
        **percentage points** of the ASR -- the same scale as
        ``severity_index``. Defaults to the acceptable band itself
        (``acceptable_inequity_level * 100``), which reports any confirmed
        excursion at all: every confirmed window is by construction further
        than the band from the centre, so the mean over them is too.

        Raising it above the band is what makes this gate bind. With a 5 %
        band and ``max_acceptable_severity=7``, a stretch of the value
        range has to be *confirmed* outside the band **and** average more
        than 7 points away from the overall median before
        ``Inequity_conclusion`` names a direction.

        Requires ``conf_level``, since confirming an excursion needs the CI
        band; ``ValueError`` if the local median leaves the band anywhere
        and no band is available.
    generate_plot : bool, default False
        If True, generate the vertical-equity moving-median plot and
        include it in the result under ``'plot'``.
    x_scale : str, default "log"
        X-axis scale - 'linear' or 'log'.
    y_scale : str, default "linear"
        Y-axis scale - 'linear' or 'log'.
    plot_title : str, default "Vertical Equity Moving Median"
        Title for the output chart.
    plot_return_object : bool, default True
        If True, return the Plotly Figure; otherwise save to
        ``plot_filepath`` and return the path string.
    plot_ref : str, optional
        Optional annotation / source label for the plot.
    plot_filepath : str, optional
        File path when ``plot_return_object`` is False. Defaults to
        'vertical_equity_median_plot.png'.
    acceptable_inequity_level : float, default 0.1
        Threshold applied to the moving median: rows whose moving median
        is within +/- this level are excluded before averaging to compute
        the vertical equity index. If falsy, all rows are averaged.
    ve_md_zoom_y : bool, list, or tuple, default True
        Y-axis range for the plot, in ASR units. Only used when
        ``smart_zoom`` is False, unless it is given as an explicit
        ``(low, high)`` pair, which always wins.
    smart_zoom : bool, default True
        Frame the plot on the bulk of the scatter: the x-axis on the
        1st-99th percentile of the market value proxy, the y-axis on 95%
        of the ASR values inside that window. Keeps the moving median,
        its CI band and the acceptable-inequity lines in view.

    Returns
    -------
    dict
        On success:

        * 'Inequity_index' : float -- the vertical equity index, i.e.
          the mean absolute moving median over the rows whose moving
          median falls outside ``acceptable_inequity_level``. 0.0 when no
          row breaches the threshold.
        * 'asr_center' : float -- the overall median ASR that was
          subtracted to centre the working ASR. Add it back to
          'moving_median' / the CI columns to read them in ASR units;
          this is what the plots do.
        * 'window_size' / 'quantiles' : int or None -- whichever of the
          two produced the local medians (the other is None), so callers
          can label the chart with the smoothing actually used.
        * 'dataframe' : pandas / polars / pyarrow DataFrame -- the
          working data with added columns, restored to the same backend
          as ``df``. Holds the raw ASR in 'asr_raw' and the centred ASR
          in ``asr_col``; 'moving_median' and (when ``conf_level`` is not
          None) 'median_ci_lower' / 'median_ci_upper' are on the centred
          scale.
        * 'plot' : plotly.graph_objects.Figure, str, or None -- the
          figure object or saved file path when ``generate_plot`` is
          True; otherwise None.
        * 'Inequity_conclusion' : str -- one of 'Progressivity',
          'Regressivity', 'Inconclusive', or 'acceptable' (nothing
          confirmed outside the acceptable band).

        When ``len(df) < 30``, a warning is printed and a dict with the
        same keys is returned, with the three indices NaN and
        ``'dataframe'``, ``'plot'`` and ``'Inequity_conclusion'`` all
        None.
    """

    # 1. Validate

    plot_out = None
    data = None
    severity_index = 0.0
    prevalence_index = 0.0
    input_backend = nw_utils.native_backend(df)

    if len(df) < 30:
        if warnings:
            print("⚠️ Warning: Sample size is less than 30; VE Median cannot be calculated")
        # Same keys as the success path below: callers (``rts_calculator``,
        # the vertical equity panel) read these directly, so a differently
        # shaped small-sample result raises KeyError and aborts the study.
        return {
            "Inequity_severity": np.nan,
            "Inequity_prevalence": np.nan,
            "asr_center": np.nan,
            "window_size": None,
            "quantiles": None,
            "dataframe": None,
            "plot": None,
            "Inequity_conclusion": None,
        }

    _validate_inputs(
    df, market_value_proxy, asr_col, conf_level,
    None, quantiles, window_size, None, x_scale, y_scale,
    )
    use_quantiles = quantiles is not None

    # 2. Build a working pandas DataFrame internally; restore the original
    # backend on return. The ASR is centred on the overall median so that
    # the moving median is zero wherever the local level matches the
    # jurisdiction-wide level. Every index below is computed on that
    # centred scale; the plots shift it back by ``asr_center`` so the
    # chart reads in ASR units (around the median, typically ~1) instead
    # of around zero.
    df_pd = nw_utils.require_pandas(df, reason="vem()")
    mv_arr = df_pd[market_value_proxy].to_numpy()
    asr_arr = df_pd[asr_col].to_numpy()
    valid = ~(np.isnan(mv_arr) | np.isnan(asr_arr))
    # The caller's index rides along, so the per-sale flags this returns can be
    # joined back to the source frame -- which is what `vem_map` needs to put a
    # colour on each sale, and what any per-sale follow-up needs generally.
    data = pd.DataFrame({"mv": mv_arr[valid], "asr_raw": asr_arr[valid]},
                        index=df_pd.index[valid])
    asr_center = float(data["asr_raw"].median())
    data[asr_col] = data["asr_raw"] - asr_center

    # 3. Sort by market value proxy
    data = data.sort_values("mv")

    # 4. Quantile groups (pandas qcut is fine here because we are working
    # on an internal pandas DataFrame).
    if use_quantiles:
        data["quantile_group"] = pd.qcut(
            data["mv"], q=quantiles, labels=False, duplicates="drop"
        )

    # 5. Moving median of the centred ASR. ``min_periods=1`` keeps every
    # record, so the ends of the value range are not lost to NaN. When
    # `conf_level` is given, this also adds 'median_ci_lower' /
    # 'median_ci_upper' -- a nonparametric CI band that tracks the moving
    # median itself.
    data = _compute_moving_median(
        data, use_quantiles, window_size, asr_col=asr_col,
        conf_level=conf_level, ci_method=ci_method,
    )

    # 6. out_of_bounds flag  (-1 / 0 / 1) -- driven by the moving median
    # rather than by band breaches.
    if acceptable_inequity_level:
        data["out_of_bounds"] = np.where(
            data["moving_median"] < -acceptable_inequity_level, -1,
            np.where(data["moving_median"] > acceptable_inequity_level, 1, 0),
        )
    else:
        data["out_of_bounds"] = 0

    # 6b. ci_out_of_bounds flag (-1 / 0 / 1) -- stricter than
    # `out_of_bounds`: a row only qualifies if its entire local median CI
    # clears `acceptable_inequity_level`, not just the point estimate.
    has_ci_band = "median_ci_lower" in data.columns and "median_ci_upper" in data.columns
    if acceptable_inequity_level and has_ci_band:
        data["ci_out_of_bounds"] = np.where(
            data["median_ci_upper"] < -acceptable_inequity_level, -1,
            np.where(data["median_ci_lower"] > acceptable_inequity_level, 1, 0),
        )
    else:
        data["ci_out_of_bounds"] = 0

    # 7. Vertical equity indices, both in *percentage points* of the ASR:
    # a local median sitting 0.15 from the jurisdiction-wide one reads as
    # 15. `max_acceptable_severity` is on the same scale.
    #
    # Both indices are measured over the *confirmed* rows -- those where the
    # moving median and its whole CI band sit outside the acceptable band --
    # so the two numbers describe the same set of windows and can be read
    # as one statement ("44% of the value range, averaging 15.8 points").
    # `strays_out_of_band` is kept separately because a row that strays on
    # the point estimate but whose band still overlaps is the difference
    # between "acceptable" and "inconclusive", not between the two indices.
    if acceptable_inequity_level:
        strays_out_of_band = (
            (data['moving_median'] > acceptable_inequity_level)
            | (data['moving_median'] < -acceptable_inequity_level)
        )
        confirmed_out_of_band = data["ci_out_of_bounds"] != 0
    else:
        # With no acceptable band there is nothing to be outside of, so
        # every row counts and the indices summarise the whole sample.
        strays_out_of_band = pd.Series(True, index=data.index)
        confirmed_out_of_band = strays_out_of_band

    def _mean_deviation_points(mask):
        """Mean |local median - overall median| over `mask`, in points."""
        if not bool(mask.any()):
            return 0.0
        value = float(data.loc[mask, "moving_median"].abs().mean()) * 100
        return 0.0 if math.isnan(value) else round(value, 4)

    severity_index = _mean_deviation_points(confirmed_out_of_band)
    prevalence_index = round(
        float(int(confirmed_out_of_band.sum()) / len(data)) * 100, 4
    )
    if math.isnan(prevalence_index):
        prevalence_index = 0.0

    # The threshold defaults to the band itself, so out of the box any
    # confirmed excursion counts: every confirmed row is further than
    # `acceptable_inequity_level` from the centre, so the mean over them is
    # too. Raising it above the band is what makes this gate bind -- e.g. a
    # 5% band with `max_acceptable_severity=7` only reports inequity once
    # the confirmed excursions average more than 7 points.
    severity_threshold = (
        max_acceptable_severity if max_acceptable_severity is not None
        else (acceptable_inequity_level or 0.0) * 100
    )

    # 7b. VE Median Direction
    if not bool(strays_out_of_band.any()):
        # Nothing leaves the acceptable band at all.
        vem_direction = "acceptable"
    else:
        if not has_ci_band and acceptable_inequity_level:
            raise ValueError(
                "`conf_level` must be set (so a median CI band is available) "
                "to determine `Inequity_conclusion`: the local median leaves "
                f"the +/-{acceptable_inequity_level * 100:g} point acceptable "
                "band somewhere, and confirming that needs the band."
            )

        if severity_index == 0.0:
            # The point estimate strays outside the acceptable window, but
            # every CI band still overlaps it -- not statistically
            # confirmed.
            vem_direction = "Inconclusive"
        elif severity_index <= severity_threshold:
            vem_direction = "acceptable"
        else:
            # Divide data into two halves (already sorted by mv)
            mid = len(data) // 2
            first_half = data.iloc[:mid]
            second_half = data.iloc[mid:]

            def calc_half_index(half_df):
                if acceptable_inequity_level:
                    mask = (half_df['moving_median'] > acceptable_inequity_level) | (half_df['moving_median'] < -acceptable_inequity_level)
                    return float(half_df.loc[mask, "moving_median"].mean()) if not half_df.loc[mask].empty else 0.0
                return float(half_df["moving_median"].mean())

            idx1 = calc_half_index(first_half)
            idx2 = calc_half_index(second_half)

            if idx1 - idx2 > 0:
                vem_direction = "Regressivity"
            elif idx1 - idx2 < 0:
                vem_direction = "Progressivity"
            elif (idx1 == 0 and idx2 != 0) or (idx2 == 0 and idx1 != 0):
                vem_direction = "Inconclusive"
            else:
                vem_direction = "Inconclusive"

    # 8. Plot
    if generate_plot:
        fig = ve_scat_median_plot(data, plot_title, plot_ref, x_scale, y_scale, acceptable_inequity_level, asr_col=asr_col, zoom_y=ve_md_zoom_y, conf_level=conf_level, center=asr_center, smart_zoom=smart_zoom)

        # 9. Return
        if plot_return_object:
            plot_out = fig
        else:
            # Suppress Kaleido/Chromium browser registration – prevents the
            # macOS "change default browser?" prompt triggered by write_image().
            # ``pio.defaults`` only exists on plotly >= 6; older versions have
            # nothing to suppress.
            import plotly.io as pio
            try:
                pio.defaults.mathjax = None
            except AttributeError:
                pass
            fp = plot_filepath or "vertical_equity_median_plot.png"
            fig.write_image(fp)
            plot_out = fp

    # Restore the original backend for the returned data frame.
    if input_backend == "polars":
        try:
            import polars as pl
            data_out = pl.from_pandas(data)
        except Exception:
            data_out = data
    elif input_backend == "pyarrow":
        try:
            import pyarrow as pa
            data_out = pa.Table.from_pandas(data)
        except Exception:
            data_out = data
    else:
        data_out = data

    return {
        "Inequity_severity": severity_index,
        "Inequity_prevalence": prevalence_index,
        "asr_center": asr_center,
        "window_size": None if use_quantiles else window_size,
        "quantiles": quantiles if use_quantiles else None,
        "dataframe": data_out,
        "plot": plot_out,
        "Inequity_conclusion": vem_direction,
    }
