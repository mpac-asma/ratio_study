# Indicators

Individual IAAO indicators, callable on their own.

*Module:* `ratio_study.indicators`

## Contents

- [`asr_mean`](#asr-mean) — Compute the (optionally weighted) mean of an array of Assessment-to-Sales Ratios.
- [`asr_median`](#asr-median) — Compute the (optionally weighted) median of an array of Assessment-to-Sales Ratios.
- [`coc`](#coc) — Coefficient of Concentration: percentage of ratios within
- [`cod`](#cod) — Coefficient of Dispersion.
- [`cod_residual`](#cod-residual) — Coefficient of Dispersion computed directly from residuals.
- [`cov_adj`](#cov-adj) — Coefficient of variation, adjusted by Fisher's small-sample
- [`fsd`](#fsd) — Forecast Standard Deviation (FSD).
- [`mae`](#mae) — Mean Absolute Error (MAE).
- [`mape`](#mape) — Mean Absolute Percentage Error (MAPE).
- [`market_value_proxy`](#market-value-proxy) — Blend the sale price and the level-adjusted assessment into a value proxy.
- [`mdmape`](#mdmape) — Median Absolute Percentage Error (MdMAPE).
- [`moran_i`](#moran-i) — Global Moran's I: whether a ratio clusters spatially (nearby sales
- [`prb`](#prb) — Compute the Price-Related Bias (PRB).
- [`prd`](#prd) — Price-Related Differential (PRD).
- [`rmse`](#rmse) — Root Mean Squared Error (RMSE), an accuracy metric for evaluating
- [`vei_analysis`](#vei-analysis) — Vertical Equity Indicator (VEI) -- Appendix E of the Standard on Ratio Studies.
- [`vem`](#vem) — Compute a measure of vertical equity from the moving median of the ASR.

---

## `asr_mean`

```python
asr_mean(asr, weights=None)
```

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

---

## `asr_median`

```python
asr_median(asr, weights=None)
```

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

---

## `coc`

```python
coc(asr, asr_md=None, within_prcnt=0.15, *, weights=None)
```

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

---

## `cod`

```python
cod(asr, asr_md=None, *, weights=None)
```

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

---

## `cod_residual`

```python
cod_residual(market_value, assessment, weights=None)
```

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

---

## `cov_adj`

```python
cov_adj(asr, weights=None)
```

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

---

## `fsd`

```python
fsd(sale_error, weights=None)
```

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

---

## `mae`

```python
mae(market_value, assessment, weights=None)
```

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

---

## `mape`

```python
mape(sale_error, weights=None)
```

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

---

## `market_value_proxy`

```python
market_value_proxy(market_value, assessment, asr=None, asr_md=None,
                   mvp_sale_price_share=0.5, *, weights=None)
```

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

---

## `mdmape`

```python
mdmape(sale_error, weights=None)
```

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

---

## `moran_i`

```python
moran_i(df, ratio, *, lat_col='latitude', long_col='longitude', k=None, permutations=999)
```

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

---

## `prb`

```python
prb(market_value, assessment, asr, market_value_proxy=None, asr_md=None,
    req_significance=0.05, non_significant_marker='*', *, weights=None,
    ci_conf_level=None, mvp_sale_price_share=0.5)
```

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

---

## `prd`

```python
prd(market_value, asr, weights=None)
```

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

---

## `rmse`

```python
rmse(market_value, assessment, metric=<function mean at 0x108272cb0>, log=False, *,
     weights=None)
```

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

---

## `vei_analysis`

```python
vei_analysis(asr, mkt_value_proxy, conf_level=0.9, acceptable_range=10,
             percentile_method='R7', ci_method=None, output_format='dict',
             metrics_only=True, gen_plot=False, plot_ref='', plot_filepath='',
             plot_return_object=False, vei_use_log_normal_delta=True,
             compare_all_buckets=False, print_settings=True, warnings=False)
```

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

---

## `vem`

```python
vem(df, market_value_proxy: str, asr_col='asr', quantiles: Optional[int] = None,
    window_size: Optional[int] = None, conf_level: Optional[float] = 0.9,
    ci_method: str = 'binom', max_acceptable_severity: Optional[float] = None,
    generate_plot=False, x_scale: str = 'log', y_scale: str = 'linear',
    plot_title: str = 'Vertical Equity Moving Median', plot_return_object: bool = True,
    plot_ref: Optional[str] = None, plot_filepath: Optional[str] = None,
    acceptable_inequity_level: float = 0.1, ve_md_zoom_y=True, smart_zoom: bool = True,
    warnings=False) -> dict
```

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
