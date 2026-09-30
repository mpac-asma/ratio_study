# Confidence intervals

Bootstrap and Appendix D closed-form interval estimators.

*Module:* `ratio_study.confint`

## Contents

- [`cod_ci`](#cod-ci) — Confidence interval for the Coefficient of Dispersion (Exposure Draft
- [`compute_median_ci`](#compute-median-ci) — Confidence interval for the sample median of a ratio-study sample.
- [`distribution_analysis`](#distribution-analysis) — Build every requested distribution-based confidence interval for one
- [`mean_ci`](#mean-ci) — Confidence interval for the mean ratio (Exposure Draft D.1).
- [`median_ci_clapp`](#median-ci-clapp) — Small-sample binomial confidence interval for the median (Exposure
- [`median_ci_distribution`](#median-ci-distribution) — Median confidence interval, picking the formula the sample size calls
- [`median_ci_rank`](#median-ci-rank) — Large-sample rank confidence interval for the median (Exposure
- [`normalize_ci_args`](#normalize-ci-args) — Resolve the confidence-interval options of `ratios` / `ratios_report`
- [`prd_ci`](#prd-ci) — Confidence interval for PRD, the price-related differential.
- [`weighted_mean_ci`](#weighted-mean-ci) — Confidence interval for the weighted mean ratio (Exposure Draft D.3;

---

## `cod_ci`

```python
cod_ci(asr, conf_level=0.95)
```

Confidence interval for the Coefficient of Dispersion (Exposure Draft
D.6; Bonett and Seier 2006).

Measures the typical absolute distance from the median, stabilises it
with a log transformation, estimates the uncertainty analytically, and
exponentiates back -- which keeps the interval usable on the skewed
ratio distributions a CoD is normally computed on.

Parameters
----------
asr : array-like
    Assessment-to-Sales Ratios. Must be strictly positive: the method
    takes logs of the ratios at two rank positions.
conf_level : float, default 0.95
    Confidence level, as a fraction or a percentage. The critical
    value is the normal deviate once the sample reaches 30 and
    Student's t below that, per the note in D.6.

Returns
-------
tuple
    ``(lower, upper)`` on the same scale as
    :func:`ratio_study.indicators.cod` -- a fraction, so 0.219 means
    21.9 %. ``(nan, nan)`` when the sample is too small (fewer than
    four ratios), when every ratio equals the median (a CoD of zero
    has no log scale), or when a rank position lands on a
    non-positive ratio.

Notes
-----
D.6 step 18 is printed as ``ln((n/(n-1)) * tau - (k * z * se2))``. The
closing parenthesis is misplaced: the interval is symmetric on the log
scale, as the section's own preamble describes and as the worked
example in D.7.1 confirms, so it is implemented as
``ln((n/(n-1)) * tau) -/+ (k * z * se2)``. Read literally, the printed
form reproduces neither the example's L1 of -1.62354 nor its U1 of
-1.42342.

---

## `compute_median_ci`

```python
compute_median_ci(asr, conf_level=0.9, method='binom')
```

Confidence interval for the sample median of a ratio-study sample.

Every method here is an order-statistic interval: both bounds are
values drawn from the sample itself, so neither can fall outside the
observed range. That matters for an ASR, which is bounded below by
zero, centres near 1, and has no upper bound -- a symmetric
normal-theory interval on a small, right-skewed sample can put its
lower bound below zero, and none of these can.

All three place the bounds at a rank distance of about
``z * sqrt(n) / 2`` either side of the middle of the sample, and
differ only in how that distance is turned into a value:

1. **Hyndman-Fan percentile rank** (``method='R6'`` or
   ``method='R7'``) -- the form prescribed by the IAAO Exposure Draft
   (May 2026) Section E, Appendix D.2, and Gloudemans & Almy (2011)
   p. 365. The interval is read off as the percentile ranks
   ``p = 0.5 -/+ z / (2 * sqrt(n))`` of the data, with ``p`` mapped to
   a position by ``h = p * (n + 1)`` (R6) or ``h = p * (n - 1) + 1``
   (R7) and the value at a fractional position interpolated between
   the two order statistics bracketing it.
2. **Exact binomial / sign test** (``method='binom'``, the default) --
   the ranks come from the exact ``Bin(n, 0.5)`` quantiles rather than
   a normal approximation to them, and no interpolation is used. The
   most conservative of the three, and the only one whose coverage is
   exact rather than approximate.

For the conservative integer-rank form Appendix D.2 spells out
step by step -- positions rounded outward rather than interpolated --
see :func:`median_ci_rank`, which is what ``ci_mode='distribution'``
uses above 30 sales. Below 30 sales that mode switches to the
tabulated binomial interpolation of Appendix D.4
(:func:`median_ci_clapp`).

Parameters
----------
asr : array-like (pd.Series, np.ndarray, or list)
    Assessment-to-Sales Ratio values.
conf_level : float, default 0.9
    Confidence level as a fraction, e.g. 0.95.
method : {'binom', 'R6', 'R7'}, default 'binom'
    CI family, as described above. ``'binom'`` is the default and is
    what ``vem`` and the other internal callers use; ``'R6'`` /
    ``'R7'`` follow the draft's percentile-rank presentation and are
    what ``vei_analysis`` uses, matching its ``percentile_method``.

Returns
-------
dict
    Dictionary with the following keys:

    * median : float -- sample median.
    * lower : float -- lower bound of the confidence interval.
    * upper : float -- upper bound of the confidence interval. For
      ``'binom'`` both bounds are order statistics of `asr`; for
      ``'R6'`` / ``'R7'`` they may be interpolated between two
      adjacent ones, but still lie within the observed range.
    * achieved_coverage : float -- coverage probability of the
      interval actually returned, which for a discrete rank-based
      interval is generally a little above `conf_level` rather than
      equal to it. Exact for ``'binom'``; for ``'R6'`` / ``'R7'`` a
      normal approximation to the same binomial quantity, accurate to
      well under a percentage point by n=200.

Notes
-----
The R6/R7 branch used to pass ``alpha/2`` in as ``p`` -- the 5th and
95th percentiles of the ratios for a 90 % interval -- which returns
the spread of the sample rather than the uncertainty of its median.
The symptoms were that the interval did not narrow as ``n`` grew
(width ~0.61 at n=100 and ~0.66 at n=1600 on the same distribution)
and that ``achieved_coverage`` came back as 1.0 whatever `conf_level`
asked for.

Examples
--------
>>> import numpy as np
>>> asr = np.linspace(0.80, 1.20, 101)
>>> out = compute_median_ci(asr, 0.95, method='binom')
>>> round(out['lower'], 3), round(out['upper'], 3)
(0.96, 1.04)

---

## `distribution_analysis`

```python
distribution_analysis(input_df, assessment_header, market_value_header,
                      weights_header=None, confint_level=0.95, indicators=None,
                      messages=None)
```

Build every requested distribution-based confidence interval for one
sample.

The distribution-mode counterpart of ``_bootstrap_analysis``: same
``{'<indicator>_ci': (lower, upper)}`` return shape, so the results
table is assembled identically whichever mode produced it. Unlike the
bootstrap there is nothing to resample, so this is a single pass over
the sample and carries no minimum-count gate of its own -- each
formula returns ``(nan, nan)`` when the sample is too small for it,
which leaves that column blank.

Parameters
----------
input_df : pandas DataFrame
    Sales for this scope (the whole study, or one group).
assessment_header : str
    Column holding the assessed / predicted value.
market_value_header : str
    Column holding the sale amount.
weights_header : str, optional
    Column holding sample weights, assumed to have mean 1.
confint_level : float, default 0.95
    Confidence level, as a fraction or a percentage.
indicators : sequence of str, optional
    Which of `DISTRIBUTION_CI_SUPPORTED_INDICATORS` to build an
    interval for. Defaults to all of them. Anything else is ignored,
    having already been reported by `normalize_ci_args`.
messages : list, optional
    Appended to with any note a formula raises about the sample --
    currently the small-sample median cautions.

Returns
-------
dict
    ``{'<indicator>_ci': (lower, upper)}`` for each requested
    indicator that this mode supports.

---

## `mean_ci`

```python
mean_ci(asr, conf_level=0.95, weights=None)
```

Confidence interval for the mean ratio (Exposure Draft D.1).

``xbar +/- t(alpha/2, n-1) * s / sqrt(n)``: the sample standard
deviation over the root of the sample size, scaled by the two-tailed
Student's t critical value for ``n - 1`` degrees of freedom.

Parameters
----------
asr : array-like
    Assessment-to-Sales Ratios (numpy/pandas/polars/narwhals Series).
conf_level : float, default 0.95
    Confidence level, as a fraction (``0.95``) or a percentage
    (``95``).
weights : array-like, optional
    Weights, assumed to have mean 1 as elsewhere in the package. The
    weighted mean and the weighted sample variance are used, which
    reduces exactly to the unweighted formula when every weight is 1.

Returns
-------
tuple
    ``(lower, upper)``, or ``(nan, nan)`` when fewer than two usable
    ratios remain (a single observation has no standard error).

Examples
--------
>>> lo, hi = mean_ci([0.9, 0.95, 1.0, 1.05, 1.1], conf_level=0.95)
>>> round(lo, 4), round(hi, 4)
(0.9018, 1.0982)

---

## `median_ci_clapp`

```python
median_ci_clapp(asr, conf_level=0.95)
```

Small-sample binomial confidence interval for the median (Exposure
Draft D.4, Tables D.4.1 and D.4.2; Clapp 1989).

Each bound is a weighted average of two adjacent order statistics,
with the weights read from the published table for the sample size.
Covers ``n`` of 5 to 30 at 90 % and 6 to 30 at 95 %; no other
confidence level is tabulated.

Parameters
----------
asr : array-like
    Assessment-to-Sales Ratios.
conf_level : float, default 0.95
    Confidence level, as a fraction or a percentage. Must resolve to
    90 % or 95 %, the two levels the tables cover.

Returns
-------
dict or None
    ``median``, ``lower``, ``upper``, ``method`` (``'binomial_table'``),
    and ``caution`` -- True when the bounds land on the lowest and
    highest ratios in the sample, which Property Appraisal and
    Assessment Administration (IAAO 1990, 609) warns against reading
    too closely. Returns None when the level or the sample size is
    outside the tables, so the caller can fall back rather than treat
    a missing table as a missing interval.

Examples
--------
The worked example from Exposure Draft D.4 (n = 17, 95 %):

>>> ratios = [0.629, 0.641, 0.658, 0.693, 0.695, 0.717, 0.787, 0.789,
...           0.820, 0.848, 0.880, 0.895, 0.933, 0.947, 1.028, 1.035,
...           1.067]
>>> out = median_ci_clapp(ratios, conf_level=0.95)
>>> round(out['lower'], 4), round(out['upper'], 4)
(0.6952, 0.9326)

---

## `median_ci_distribution`

```python
median_ci_distribution(asr, conf_level=0.95, messages=None)
```

Median confidence interval, picking the formula the sample size calls
for (Exposure Draft D.2 and D.4).

``n <= 30`` uses the exact binomial tables of D.4; a larger sample
uses the rank formula of D.2. A small sample at a confidence level
the tables do not cover falls back to the exact binomial sign-test
interval of `compute_median_ci`, which is the same family computed
rather than tabulated.

Parameters
----------
asr : array-like
    Assessment-to-Sales Ratios.
conf_level : float, default 0.95
    Confidence level, as a fraction or a percentage.
messages : list, optional
    Appended to when the sample size is below the smallest tabulated
    one, or when the bounds sit on the extreme ratios of the sample.

Returns
-------
tuple
    ``(lower, upper)``, or ``(nan, nan)`` when the sample is too small
    for any of the three formulas.

---

## `median_ci_rank`

```python
median_ci_rank(asr, conf_level=0.95)
```

Large-sample rank confidence interval for the median (Exposure
Draft D.2).

Counts ``j`` ranks out from the middle of the sorted ratios, where
``j = z * sqrt(n) / 2`` for an odd sample and ``j = z * sqrt(n) / 2 +
0.5`` for an even one, rounded up. The bounds are the ratios at those
ranks, so they are always observed values and cannot fall outside the
sample. Intended for ``n > 30``; below that the exact binomial tables
of D.4 are tighter (see `median_ci_clapp`).

Parameters
----------
asr : array-like
    Assessment-to-Sales Ratios.
conf_level : float, default 0.95
    Confidence level, as a fraction or a percentage. ``z`` is the
    normal deviate for it -- 1.960 at 95 %, 1.645 at 90 %, the values
    the draft quotes.

Returns
-------
dict
    ``median``, ``lower``, ``upper``, ``lower_rank``, ``upper_rank``
    (1-indexed), and ``method`` (``'rank'``). The bounds are
    ``(nan, nan)`` for an empty sample.

Notes
-----
Rounding ``j`` up widens the interval, so the achieved coverage is at
or above the nominal level -- the conservative direction, and the one
the draft prescribes ("if the result from step 2 is not an integer,
round up").

---

## `normalize_ci_args`

```python
normalize_ci_args(bootstrap_args=None, confidence_intervals_aggs=None,
                  ci_conf_level=None, ci_indicators=None, ci_seed=None, ci_mode=None,
                  study_indicators=None, has_groups=False, defaults=None)
```

Resolve the confidence-interval options of `ratios` / `ratios_report`
into a fully populated ``bootstrap_args`` dictionary.

Three call styles are supported and can be mixed:

* the high-level switches -- ``confidence_intervals_aggs``, `ci_conf_level`,
  `ci_indicators`;
* short-form `bootstrap_args` keys, e.g.
  ``{'groups': True, 'level': 0.9, 'resamplings': 500}``;
* the fully spelled-out `bootstrap_args` keys, e.g.
  ``{'bootstrap_cis_for_groups': True}``.

`confidence_intervals_aggs` / `ci_conf_level` / `ci_indicators` are applied last,
so ``confidence_intervals_aggs=False`` reliably switches every interval off
regardless of what `bootstrap_args` says.

Parameters
----------
bootstrap_args : dict, optional
    Bootstrap options in canonical or short form. See
    `_CI_ARG_ALIASES` for the accepted short forms.
confidence_intervals_aggs : bool, str, sequence, or dict, optional
    High-level switch:

    * None -- leave `bootstrap_args` / the defaults alone.
    * False / ``'none'`` / ``'off'`` -- no confidence intervals.
    * True / ``'overall'`` / ``'population'`` -- overall CIs only.
    * ``'groups'`` -- per-group CIs only.
    * ``'both'`` / ``'all'`` -- overall and per-group CIs.
    * a sequence of indicator names -- overall CIs for those
      indicators, e.g. ``['Median', 'CoD']``.
    * a dict -- treated as `bootstrap_args`, short forms included.
ci_conf_level : float, optional
    Confidence level, as a fraction (``0.9``) or a percentage
    (``90``).
ci_indicators : str or sequence of str, optional
    Indicators to bootstrap. A bare string is accepted for one
    indicator.
ci_seed : int, optional
    Seed for the resampler, so the intervals are reproducible. Only
    the bootstrap uses it; the distribution formulas are exact.
ci_mode : {'bootstrap', 'distribution'}, optional
    Which family of interval to build. ``'distribution'`` (the default)
    uses the closed-form formulas of the IAAO Exposure Draft (May 2026)
    Appendix D; ``'bootstrap'`` resamples. The mode decides
    which indicators can carry an interval at all --
    `CI_SUPPORTED_INDICATORS` for the bootstrap,
    `DISTRIBUTION_CI_SUPPORTED_INDICATORS` for the distribution
    formulas -- so it is resolved before the indicator list is
    validated against it.
study_indicators : sequence of str, optional
    Indicators the study itself computes. The CI indicator list is
    intersected with these, since a ``<name>_ci`` column with no
    ``<name>`` column beside it would be unreadable. When the caller
    names the CI indicators themselves and the mode is
    ``'distribution'``, the study takes on the missing indicator
    instead of losing the interval -- each closed form is one pass
    over the ratios -- and the names to add come back as
    ``'ci_study_indicator_additions'``. Otherwise the interval is
    dropped, with a message when it was explicitly requested.
has_groups : bool, default False
    Whether the study has a `group_by`. Per-group CIs are switched
    off when it does not.
defaults : dict, optional
    Starting values for any option the caller did not set. Defaults to
    the package defaults: overall CIs on, distribution mode, BCa, 1000
    resamplings, 90 %, the mode's default indicators,
    ``min_count_threshold=20``.

Returns
-------
tuple
    ``(bootstrap_args, messages)`` -- the resolved options keyed by
    canonical name, and a list of input-problem messages for the
    caller to report. The resolved options carry one output-only key,
    ``'ci_study_indicator_additions'``: indicators the study must add
    to `indicators` so the requested intervals have a point estimate
    to sit beside (distribution mode only, see `study_indicators`).

Examples
--------
>>> args, msgs = normalize_ci_args(confidence_intervals_aggs='both',
...                                ci_conf_level=90, has_groups=True)
>>> args['bootstrap_cis_for_groups'], args['bootstrap_confint_level']
(True, 0.9)
>>> args, msgs = normalize_ci_args(ci_mode='distribution',
...                                study_indicators=['Median', 'CoD', 'RMSE'])
>>> args['ci_mode'], args['bootstrap_indicators']
('distribution', ['Median', 'CoD'])

---

## `prd_ci`

```python
prd_ci(assessment, market_value, conf_level=0.95, weights=None)
```

Confidence interval for PRD, the price-related differential.

Not one of the Exposure Draft's Appendix D formulas -- PRD has none --
so this is a delta-method (first-order Taylor) interval built from two
formulas the module already has, since ``PRD = Mean / Weighted_Mean``
(see `prd` in ``indicators.py``): `mean_ci`'s D.1 formula for the
numerator and `weighted_mean_ci`'s D.3 ratio-estimator formula for the
denominator.

Both statistics are computed from the same sample, so they are
correlated, and that covariance has to enter the ratio's variance or
the interval comes out too narrow. Writing ``M`` for the mean ratio and
``W`` for the weighted mean ratio, with per-sale weight ``w_i``
(normalised to mean 1) and ``mean_s`` the weighted mean sale amount:

``Var(M)   = sum(w_i * (asr_i - M)^2)       / ((sum(w) - 1) * n)``
``Var(W)   = sum(w_i * (A_i - W*S_i)^2)      / (mean_s^2 * n*(n-1))``
``Cov(M,W) = sum(w_i * (asr_i-M)*(A_i-W*S_i)) / (mean_s * n*(n-1))``

-- the same weighted-cross-product structure as each variance above,
which is what makes `Cov(M,W)` the consistent covariance estimator for
this pair rather than an ad hoc one. The delta method then gives

``Var(PRD) = Var(M)/W^2 + M^2*Var(W)/W^4 - 2*M*Cov(M,W)/W^3``

and the interval is ``PRD +/- t(alpha/2, n-1) * sqrt(Var(PRD))``.

Parameters
----------
assessment : array-like
    Assessed / predicted values (``A``).
market_value : array-like
    Sale amounts, time-adjusted where applicable (``S``).
conf_level : float, default 0.95
    Confidence level, as a fraction or a percentage.
weights : array-like, optional
    Weights, assumed to have mean 1. Applied throughout, which reduces
    to the unweighted formula when every weight is 1.

Returns
-------
tuple
    ``(lower, upper)``, or ``(nan, nan)`` when fewer than two usable
    sales remain, the sale amounts sum to zero, or the weighted mean
    ratio is zero (PRD is undefined).

---

## `weighted_mean_ci`

```python
weighted_mean_ci(assessment, market_value, conf_level=0.95, weights=None)
```

Confidence interval for the weighted mean ratio (Exposure Draft D.3;
Fundamentals of Mass Appraisal 2011, 367, Table 10.2).

The weighted mean is the ratio of the two totals,
``R = sum(A) / sum(S)``, so its standard error is the ratio-estimator
one:

``SE(R) = sqrt(sum(A^2) - 2*R*sum(A*S) + R^2*sum(S^2))
          / (Sbar * sqrt(n * (n - 1)))``

and the interval is ``R +/- t(alpha/2, n-1) * SE(R)``. The numerator
is ``sqrt(sum((A - R*S)^2))`` written out in the sums the draft's
steps 2, 10 and 11 accumulate.

Parameters
----------
assessment : array-like
    Assessed / predicted values (``A``).
market_value : array-like
    Sale amounts, time-adjusted where applicable (``S``).
conf_level : float, default 0.95
    Confidence level, as a fraction or a percentage.
weights : array-like, optional
    Weights, assumed to have mean 1. Applied to every sum, which
    reduces to the unweighted formula when every weight is 1.

Returns
-------
tuple
    ``(lower, upper)``, or ``(nan, nan)`` when fewer than two usable
    sales remain or the sale amounts sum to zero.

Notes
-----
The draft renders step 11 as ``(Abar / Sbar^2)(sum(S^2))``. Read that
way, step 12 would not be the square root of ``sum((A - R*S)^2)`` that
steps 10 and 12 imply, and the worked reference in Fundamentals of
Mass Appraisal would not reproduce; the exponent belongs to the whole
ratio, so it is implemented as ``R^2 * sum(S^2)``.
