# Sales chasing

The Appendix B detection techniques.

*Module:* `ratio_study.sales_chasing`

## Contents

- [`critical_concentration`](#critical-concentration) — The critical proportion of ratios from Appendix B.4, Table B.4.1.
- [`detect_sales_chasing`](#detect-sales-chasing) — Run every Appendix B technique the supplied data supports and report
- [`sc_cdf_gap`](#sc-cdf-gap) — Detect a discontinuity ("flat spot") in the empirical CDF of the ratios.
- [`sc_concentration`](#sc-concentration) — Appendix B.4 -- observed versus expected distribution of ratios.
- [`sc_independent_values`](#sc-independent-values) — Appendix B.6 -- independent value chasing detection.
- [`sc_mass_appraisal`](#sc-mass-appraisal) — Appendix B.5 -- mass appraisal technique, on caller-supplied independent
- [`sc_split_sample`](#sc-split-sample) — Appendix B.3 -- split-sample technique.
- [`sc_unit_values`](#sc-unit-values) — Appendix B.2 -- comparison of unit values between sold and unsold parcels.
- [`sc_value_change`](#sc-value-change) — Appendix B.1 -- comparison of value changes between sold and unsold parcels.

---

## `critical_concentration`

```python
critical_concentration(min_achievable_cod=0.04, band=0.02)
```

The critical proportion of ratios from Appendix B.4, Table B.4.1.

Given the lowest CoD credibly achievable for the property type, the
table asks: under a normal distribution with that CoD, what share of
ratios would land within ``±band`` of the centre? Anything above that
share is more concentration than honest valuation can produce, and so
points at selective revaluation.

The published table is reproduced exactly rather than looked up:
``sigma = 1.25 * CoD`` (Table B.4.1's own rounding of the normal
identity ``sigma = CoD * sqrt(pi/2)``), then the expected proportion is
``2 * Phi(band / sigma) - 1``, and the critical proportion is "the
integer immediately exceeding" it in percentage points.

Parameters
----------
min_achievable_cod : float, default 0.04
    Minimum achievable CoD for the stratum, as a fraction. The
    default 0.04 is the standard's own conservative reference point
    (the row that yields the well-known 32% critical value).
band : float, default 0.02
    Half-width of the window around the centre, as a fraction.

Returns
-------
dict
    ``sd`` (implied standard deviation), ``expected_proportion``
    (exact, under the normal assumption), and
    ``critical_proportion`` (the table's integer-percent threshold,
    as a fraction).

Examples
--------
>>> round(critical_concentration(0.04)['critical_proportion'], 2)
0.32
>>> round(critical_concentration(0.016)['critical_proportion'], 2)
0.69

---

## `detect_sales_chasing`

```python
detect_sales_chasing(df=None, assessment_col=None, sale_price_col=None, asr=None,
                     group_by=None, sold_col=None, prior_value_col=None,
                     current_value_col=None, size_col=None, sale_date_col=None,
                     cutoff_date=None, independent_value_col=None,
                     independent_technique='auto', exclude_col=None,
                     min_achievable_cod=0.04, band=0.02, alpha=0.05, tolerance=0.05,
                     conf_level=0.95, standards=None, min_n=30)
```

Run every Appendix B technique the supplied data supports and report
the results in one frame.

Which tests run is decided by which columns were passed, so a caller
with only a sales file gets B.4 plus the CDF screen, and a caller with
the full roll gets all six. Nothing is silently skipped: a technique
that lacks its inputs appears in the output as ``not run`` with the
reason, because "we did not test for it" and "we tested and found
nothing" are different findings and an oversight agency reading the
report needs to be able to tell them apart.

Every applicable technique runs on every scope. The appendix
stratifies throughout -- B.1 works "within a specified group", B.2's
visualisations are "repeated across multiple data strata", B.6 requires
"the same strata" -- so ``group_by`` is not a ratio-screen-only
convenience.

Parameters
----------
df : DataFrame, optional
    Any supported frame. Required for every test except the
    ratio-only ones, which can take ``asr`` directly.
assessment_col, sale_price_col : str, optional
    Columns from which the ratio is built. Ignored if ``asr`` is given.
asr : array-like, optional
    Pre-computed ratios, as an alternative to the two columns above.
    Cannot be combined with ``group_by``.
group_by : str or list of str, optional
    Also run every applicable technique within each stratum. Strata
    smaller than ``min_n`` are reported as too small rather than
    dropped.
sold_col : str, optional
    Enables B.1 and B.2, and splits sold from unsold in B.5. Boolean /
    0-1: did the parcel sell.
prior_value_col, current_value_col : str, optional
    Enable B.1 (both required, with ``sold_col``).
size_col : str, optional
    Enables B.2 (with ``sold_col`` and ``assessment_col``).
sale_date_col, cutoff_date : optional
    Enable B.3 (both required).
independent_value_col : str, optional
    Caller-supplied independent estimate of market value. Enables B.5
    and/or B.6; nothing is modelled here.
independent_technique : {'auto', 'B.5', 'B.6', 'both'}, default 'auto'
    Which of the two independent-value comparisons to run. ``'auto'``
    runs B.5 when ``sold_col`` is available -- it needs the sold/unsold
    split -- and B.6 otherwise.
exclude_col : str, optional
    Parcels to drop from B.1, B.2, B.5 and B.6.
min_achievable_cod, band, alpha, tolerance, conf_level, standards, min_n
    Passed through to the individual tests.

Returns
-------
pandas.DataFrame
    One row per (technique, scope) with columns ``technique``,
    ``reference`` (the Appendix B section), ``scope``, ``n``,
    ``statistic``, ``threshold``, ``p_value``, ``flag``,
    ``conclusion``, plus ``detail`` holding the full result dict for
    the technique.

---

## `sc_cdf_gap`

```python
sc_cdf_gap(asr, center=('median', 'unity'), band=0.05, gap=0.03, top_k=3, min_n=30)
```

Detect a discontinuity ("flat spot") in the empirical CDF of the ratios.

Not one of Appendix B's numbered techniques, but the sharpest available
signal for the crudest form of chasing: setting the assessment equal to
the sale price. That produces a pile of *identical* ratios, which shows
up as a vertical jump in the ECDF -- a flat spot when the CDF is drawn
with ratio on the vertical axis. Concentration tests (B.4) see this too,
but only as part of a general bunching; the ECDF jump identifies it as
one repeated value, which is far more specific and is directly
actionable: the flagged ratio value names the parcels to pull.

Differences from ``assesspy.is_sales_chased(method='cdf')``, which
inspired this function:

* The largest jump is taken *within* the band around the centre, rather
  than taking the global largest jump and then testing whether it
  happens to lie in the band. A single far outlier creates the largest
  jump in the whole sample and, under the assesspy logic, silently
  suppresses a genuine spike at 1.00.
* The band is the union of one around the sample's own central tendency
  and one around 1.00, so a stratum assessed at 0.85 is testable *and*
  a pile at 1.00 is still found when the honest ratios sit elsewhere.
  A single centre fails the bimodal case in both directions: chasing
  drags the median toward the pile in a partly-chased sample, and a
  band pinned to 1.00 misses a stratum that was chased to its own
  level rather than to the sale price.
* The gaps are returned, not just a boolean, so the flagged ratio value
  can be looked up in the sales file.

Parameters
----------
asr : array-like
    Assessment-to-sales ratios. Non-finite values are dropped.
center : str, float, or sequence, default ``('median', 'unity')``
    Centre(s) of the search band; each is ``'median'``, ``'mean'``,
    ``'unity'``, or a number. The searched region is the union of the
    bands around them.
band : float, default 0.05
    Half-width of the search band, as a fraction of the centre. Wider
    than B.4's ±2% on purpose: a spike need only be *near* the centre
    to be suspicious, and a wider band costs nothing here because the
    test is about a single discontinuity rather than an aggregate.
gap : float, default 0.03
    Jump size that constitutes a finding, as a fraction of the sample.
    3% of sales sharing one ratio value to the last decimal is already
    hard to explain.
top_k : int, default 3
    How many of the largest in-band jumps to return.
min_n : int, default 30
    Below this the test is not run.

Returns
-------
dict
    ``n``, ``center_value`` (list), ``bounds`` (list of pairs), ``max_gap``,
    ``max_gap_at`` (the ratio value carrying the jump),
    ``max_gap_n`` (how many sales share it), ``top_gaps``
    (list of ``(ratio, gap, count)``), ``flag``, ``conclusion``.

---

## `sc_concentration`

```python
sc_concentration(asr, center='median', band=0.02, min_achievable_cod=0.04, alpha=0.05,
                 scan_modes=True, min_n=30, weights=None)
```

Appendix B.4 -- observed versus expected distribution of ratios.

Counts the ratios inside a narrow window around the centre of the
distribution and compares that share against the most a normal
distribution with the *lowest credible* dispersion could produce. Its
logic is a floor argument, not a model: whatever the true shape of
honest ratios, a stratum cannot beat its own minimum achievable CoD,
so a concentration above the Table B.4.1 threshold has to come from
somewhere other than valuation skill. Chased sales are set at (or very
near) their sale price, which is exactly a spike at ratio 1.

Two refinements over the bare table:

* The window is centred on the sample's own measure of central
  tendency, per B.4 ("within ±2% of a measure of central tendency,
  such as the median or a mode, or 100%"). A stratum assessed at 0.85
  is still testable.
* ``scan_modes`` additionally reports the fullest window anywhere in
  the sample, which is what catches the bimodal case B.4 warns about
  -- e.g. chasing to 1.00 in a stratum whose honest ratios sit at 0.90.

The test is one-sided by construction: only *excess* concentration is
evidence. A sample flatter than the reference normal is a dispersion
problem, not a chasing problem.

Parameters
----------
asr : array-like
    Assessment-to-sales ratios (numpy/pandas/polars/narwhals).
    Non-finite values are dropped.
center : {'median', 'mean', 'unity'} or float, default 'median'
    Centre of the window. ``'unity'`` fixes it at 1.00, which is the
    right choice when chasing is expected to target the sale price
    itself rather than the stratum's level.
band : float, default 0.02
    Half-width of the window, as a fraction of the centre. The
    standard's ±2%.
min_achievable_cod : float, default 0.04
    Lowest CoD credibly achievable for this property type. Drives the
    critical proportion; see :func:`critical_concentration`. Raise it
    for strata where 4% was never plausible -- doing so *lowers* the
    threshold and makes the test more sensitive, which is the point
    of Table B.4.1.
alpha : float, default 0.05
    Significance level for the exact binomial test of the observed
    count against the reference proportion.
scan_modes : bool, default True
    Also report the fullest ``±band`` window anywhere in the sample.
min_n : int, default 30
    Below this the test is not run (B.4).
weights : array-like, optional
    Sample weights, used only for the weighted median/mean when
    ``center`` is ``'median'``/``'mean'``. The concentration count
    itself is unweighted -- the test is about how many parcels were
    moved, not their aggregate value.

Returns
-------
dict
    ``n``, ``center``, ``center_value``, ``band``, ``bounds``,
    ``n_in_band``, ``observed_proportion``, ``expected_proportion``,
    ``critical_proportion``, ``excess_proportion``, ``p_value``
    (one-sided exact binomial), ``observed_cod``,
    ``implied_reference_cod``, ``max_window_center``,
    ``max_window_proportion``, ``flag`` (bool or None), and
    ``conclusion``.

Notes
-----
The reference proportion is deliberately *not* derived from the
sample's own standard deviation, which is what
``assesspy.is_sales_chased(method='dist')`` does. Chasing shrinks the
sample's standard deviation, so a self-referential benchmark moves
toward the very concentration it is supposed to detect, and the test
loses power exactly where it is needed. The sample's own CoD is
reported as ``observed_cod`` for context instead.

---

## `sc_independent_values`

```python
sc_independent_values(df, assessment_col, sale_price_col, independent_value_col,
                      exclude_col=None, conf_level=0.95, alpha=0.05, tolerance=0.05,
                      standards=None, min_n=30)
```

Appendix B.6 -- independent value chasing detection.

Mechanically the same comparison as :func:`sc_mass_appraisal`, with the
suspicion pointed the other way. When a ratio study leans in part on
proxy values determined independently -- typically an oversight
agency's own appraisals -- those values can be chased as readily as
assessments can, and the appendix asks for the ratios of independently
valued properties and of selling properties **in the same strata** to
be compared for statistically significant differences.

So the test here is two-sided by design. A material gap says the two
groups were not valued alike; which side moved is a question the
numbers cannot answer, and the conclusion string says so rather than
assuming it. Read alongside B.5 on the same data: B.5 asks whether the
sales look too good, B.6 whether the two sources disagree at all.

"In the same strata" is a requirement, not a suggestion -- run this
within a stratum, or pass ``group_by`` to
:func:`detect_sales_chasing`. Across a whole jurisdiction the
comparison mostly measures which strata each source happened to cover.

Parameters
----------
df : DataFrame
    Any supported frame, covering both the sales and the
    independently valued parcels. Unlike B.5 the two groups need not
    be disjoint by sale status -- a parcel with both a sale price and
    an independent value contributes to both studies -- so the
    Mann-Whitney p-value is reported for information and the
    interval-overlap tests carry the finding.
assessment_col : str
    Assessed value -- the numerator of both ratio studies.
sale_price_col : str
    Time-adjusted sale price. Denominator of the sales study.
independent_value_col : str
    Independently determined value. Denominator of the second study.
exclude_col : str, optional
    Boolean / 0-1 column of parcels to drop from both studies.
conf_level : float, default 0.95
    Confidence level for the median and CoD intervals.
alpha : float, default 0.05
    Significance level.
tolerance : float, default 0.05
    Practical significance floor: the median gap as a fraction of the
    independent study's median, the CoD gap in percentage points.
standards : dict, optional
    ``{'Median': (lower, upper), 'CoD': (lower, upper)}``; when given,
    each study's acceptance is reported.
min_n : int, default 30
    Minimum size for each study.

Returns
-------
dict
    Same shape as :func:`sc_mass_appraisal`, plus ``n_overlap`` --
    parcels contributing to both studies, which is the caveat on the
    p-value.

---

## `sc_mass_appraisal`

```python
sc_mass_appraisal(df, assessment_col, sale_price_col, independent_value_col,
                  sold_col=None, exclude_col=None, conf_level=0.95, alpha=0.05,
                  tolerance=0.05, standards=None, min_n=30)
```

Appendix B.5 -- mass appraisal technique, on caller-supplied independent
values.

Runs the ratio study the appendix describes: on the unsold parcels,
substitute the independent estimate of market value for the sale price
in the denominator, then compare the resulting level and uniformity
against the ordinary sales ratio study. If the sold parcels were chased,
the sales study is the flattering one -- its median sits nearer 1.00 and
its CoD is lower -- while the unsold study, which no sale price could
have influenced, reports what the valuation actually achieved.

Its advantage over B.1/B.2 is that it speaks in ratio-study units: the
output is a level and a uniformity measure that can be tested against
the same standards as the sales study, rather than a difference in
value changes that has to be argued about separately. Its cost is that
it is only as good as the independent values.

``independent_value_col`` is supplied by the caller and is not built
here; see the module notes on what "independent" has to mean for the
result to carry weight.

Sample split
------------
With ``sold_col``: the sales study runs on parcels where it is true,
the independent study on parcels where it is false. Without it, a
usable sale price is taken to mean sold, and the independent study runs
on the parcels that have an independent value but no sale price. Either
way the two samples are disjoint, which the Mann-Whitney test requires.

Parameters
----------
df : DataFrame
    Any supported frame, covering sold and unsold parcels.
assessment_col : str
    Assessed value -- the numerator of both ratio studies.
sale_price_col : str
    Time-adjusted sale price. Denominator of the sales study.
independent_value_col : str
    Independently determined market value. Denominator of the second
    study. Rows where it is missing or non-positive are dropped from
    that study only.
sold_col : str, optional
    Boolean / 0-1 column: did the parcel sell in the period.
exclude_col : str, optional
    Boolean / 0-1 column of parcels to drop from both studies.
conf_level : float, default 0.95
    Confidence level for the median and CoD intervals.
alpha : float, default 0.05
    Significance level for the Mann-Whitney test on the two sets of
    ratios.
tolerance : float, default 0.05
    Practical significance floor (B.1's tolerance, carried over as the
    appendix directs), read in the units each measure is argued in: the
    median gap as a *fraction* of the independent study's median, the
    CoD gap in *percentage points*. See
    :func:`_compare_ratio_studies` for why the two differ.
standards : dict, optional
    ``{'Median': (lower, upper), 'CoD': (lower, upper)}``, e.g. from
    ``config.assessment_standards``. When given, each study's
    acceptance is reported -- the appendix's point that this technique
    "yields measures of central tendency, which can be ... tested for
    acceptance with standards for the level of valuation".
min_n : int, default 30
    Minimum size for each study.

Returns
-------
dict
    ``n_sales``, ``n_independent``, ``median_*``, ``cod_*``, ``prd_*``
    and ``*_ci_*`` for both studies, ``median_difference``,
    ``median_relative_difference``, ``cod_difference``,
    ``cod_relative_gap``, ``mannwhitney_p`` (location, on the ratios),
    ``dispersion_p`` (uniformity, on the relative absolute deviations),
    ``median_ci_disjoint``, ``cod_ci_disjoint``, ``findings``, the
    ``*_acceptable`` flags when ``standards`` is given, ``flag``, and
    ``conclusion``.

Notes
-----
The uniformity leg of this comparison is the weaker one. Independent
values carry estimation error of their own, and that error lands in the
denominator, inflating the independent study's CoD even when nothing
was chased -- so a CoD gap is only evidence when the independent values
are of a quality comparable to sale prices. The level (median)
comparison is far more robust to it, because denominator noise is
roughly symmetric and largely cancels in a median. Read a CoD-only
finding with that in mind; a level finding is the serious one.

---

## `sc_split_sample`

```python
sc_split_sample(df, assessment_col, sale_price_col, sale_date_col, cutoff_date,
                conf_level=0.95, alpha=0.05, tolerance=0.05, min_n=30)
```

Appendix B.3 -- split-sample technique.

The only technique here that an oversight agency can run on a sales
file alone, with no access to the unsold population, which is why the
appendix singles it out. Sales that closed before the agency's values
were locked *could* have been chased; sales that closed after could
not. Time-adjusted, the two subsamples estimate the same thing, so if
the early one looks materially better -- tighter CoD, median closer to
1.00 -- the difference is an artefact of the valuation process rather
than of the market.

The asymmetry in the appendix's conclusion is worth keeping in mind:
when the split flags, "the given statistic in the second study is
still valid but the statistic in the first study should be rejected."
The late sample is the usable one.

This function does **not** time-adjust. The sale prices passed in are
assumed already adjusted for market conditions, exactly as elsewhere in
this package; an unadjusted split will show a trending market as
chasing.

Parameters
----------
df : DataFrame
    Sales frame (any supported backend).
assessment_col, sale_price_col : str
    Assessed value and time-adjusted sale price.
sale_date_col : str
    Sale date, parseable by ``pandas.to_datetime``.
cutoff_date : str or datetime
    Last date on which the agency could still change values other than
    through appeal. Sales on or before it fall in the "known" sample.
conf_level : float, default 0.95
    Confidence level for the median and CoD intervals.
alpha : float, default 0.05
    Significance level for the Mann-Whitney test on the ratios.
tolerance : float, default 0.05
    Practical significance floor for the CoD gap, as a fraction of the
    late sample's CoD (B.1's tolerance, carried over as the appendix
    directs).
min_n : int, default 30
    Minimum size for each subsample.

Returns
-------
dict
    ``cutoff_date``, ``n_known``, ``n_unknown``, per-sample
    ``median_*`` / ``cod_*`` point estimates and intervals,
    ``cod_ratio`` (known / unknown), ``mannwhitney_p``,
    ``median_ci_disjoint``, ``cod_ci_disjoint``, ``flag``,
    ``conclusion``.

---

## `sc_unit_values`

```python
sc_unit_values(df, sold_col, value_col, size_col, exclude_col=None, statistic='median',
               alpha=0.05, min_n=30)
```

Appendix B.2 -- comparison of unit values between sold and unsold parcels.

Where B.1 needs two years of assessed values, this needs only one, which
is often the difference between a test that can be run and one that
cannot. If sold parcels were pushed toward their sale prices, their
value per unit of size (per square foot, per acre) drifts away from that
of comparable unsold parcels. It is a weaker test than B.1 because sold
parcels genuinely differ from unsold ones -- turnover is not random, and
what sells skews newer, better-located, more marketable -- so a
difference here is only evidence within a stratum tight enough that the
two groups are otherwise alike. Run it inside strata, not across a
jurisdiction.

Parameters
----------
df : DataFrame
    Any supported frame, covering sold and unsold parcels.
sold_col : str
    Boolean / 0-1 column: did the parcel sell.
value_col : str
    Assessed value.
size_col : str
    Size denominator (living area, lot area, acres). Non-positive
    values are dropped.
exclude_col : str, optional
    Boolean / 0-1 column of parcels to drop first.
statistic : {'median', 'mean'}, default 'median'
    Measure of central tendency. Unit values are right-skewed, so the
    median is the default here (B.2 allows either).
alpha : float, default 0.05
    Significance level.
min_n : int, default 30
    Minimum size for each group.

Returns
-------
dict
    Same shape as :func:`sc_value_change`, with ``difference``
    additionally expressed as ``relative_difference`` (a fraction of
    the unsold group's central value), which is what the tolerance is
    applied to.

---

## `sc_value_change`

```python
sc_value_change(df, sold_col, prior_value_col, current_value_col, exclude_col=None,
                statistic='auto', no_change_threshold=0.5, alpha=0.05, min_n=30)
```

Appendix B.1 -- comparison of value changes between sold and unsold parcels.

The most direct test in the appendix, and the one that speaks to the
mechanism rather than to a symptom: if the sold parcels were revalued in
a way the unsold ones were not, the year-over-year percentage change in
assessed value differs between the two groups. No ratio is involved, so
the test is immune to whatever the sale prices happen to be, and it
covers chasing on recoding as well as on value.

Follows the appendix's rules on all three of its judgement calls:

* **Median over mean**, except that when a large share of the unsold
  group had *no* value change the median of that group is
  uninformative (it is just 0), and the mean is used instead.
  ``statistic='auto'`` applies that switch at ``no_change_threshold``.
* **Two gates, not one.** Large samples make trivial differences
  significant, so a difference must clear both statistical significance
  and the practical tolerance -- 5 percentage points, tightened to 3
  when the sold group's own change is under 5%.
* **Exclusions matter more than the statistics.** Parcels with a
  physical change, a change in use, an appeal or stipulation, or values
  the agency does not set (e.g. farmland use values) bias the test
  *toward* finding chasing and must come out first. ``exclude_col``
  is the hook; getting its definition right is the real work of this
  test.

Parameters
----------
df : DataFrame
    Any supported frame (pandas/polars/pyarrow/narwhals). Must cover
    both sold and unsold parcels -- this is not a sales-file test.
sold_col : str
    Boolean / 0-1 column: did the parcel sell in the period under study.
prior_value_col, current_value_col : str
    Assessed value before and after the revaluation.
exclude_col : str, optional
    Boolean / 0-1 column marking parcels to drop before comparison
    (physical change, use change, appeal, non-agency value).
statistic : {'auto', 'median', 'mean'}, default 'auto'
    Measure of central tendency. ``'auto'`` uses the median unless the
    unsold group's no-change share exceeds ``no_change_threshold``.
no_change_threshold : float, default 0.5
    Share of unsold parcels with zero value change above which
    ``'auto'`` switches to the mean.
alpha : float, default 0.05
    Significance level.
min_n : int, default 30
    Minimum size for *each* group.

Returns
-------
dict
    ``n_sold``, ``n_unsold``, ``n_excluded``, ``statistic``,
    ``sold_statistic``, ``unsold_statistic``, ``difference``
    (percentage points, as a fraction), ``tolerance``,
    ``sold_pct_no_change``, ``unsold_pct_no_change``,
    ``mannwhitney_p``, ``ttest_p``, ``statistically_significant``,
    ``practically_significant``, ``flag``, ``conclusion``.
