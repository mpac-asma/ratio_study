"""Sales-chasing (selective reappraisal) detection, per the IAAO Standard on
Ratio Studies Appendix B.

Sales chasing is the selective revaluation or recoding of parcels that sold,
using the sale price (or a proxy such as a listing price) as the trigger. It
does not make a ratio study look bad -- it makes it look *good*: the ratios of
the chased sales collapse onto the centre, so CoD falls, the median moves to
1.00, and every uniformity statistic computed from that sample stops describing
the unsold population it is meant to represent. Detection therefore cannot rely
on the ordinary indicators; it needs tests that look at the *shape* of the ratio
distribution and at whether sold and unsold parcels were treated alike.

Appendix B lists six techniques. All six are implemented here, and each returns
a dict carrying the raw statistics, a boolean ``flag``, and a ``conclusion``
string:

===== ============================================ =============================
B.1   Comparison of value changes                  :func:`sc_value_change`
B.2   Comparison of unit values                    :func:`sc_unit_values`
B.3   Split-sample technique                       :func:`sc_split_sample`
B.4   Observed vs expected distribution of ratios  :func:`sc_concentration`
--    CDF-discontinuity heuristic (not in B, but
      the standard's "bunching" idea made local)   :func:`sc_cdf_gap`
B.5   Mass appraisal technique                     :func:`sc_mass_appraisal`
B.6   Independent value chasing detection          :func:`sc_independent_values`
===== ============================================ =============================

:func:`detect_sales_chasing` runs whichever of these the supplied data
supports, optionally within strata, and returns one tidy frame.

Every test here is a *screen*, not proof. Appendix B is explicit that even when
a critical proportion is exceeded, "further investigation should be conducted
before concluding that sales chasing has occurred". A flag means: pull the
parcels and look at what happened to them.

Notes on B.5 / B.6
------------------
Both take the independent values as **a column the caller supplies**. Nothing
here builds a model. That is not a shortcut: B.5's whole premise is that the
independent estimate is arrived at *independently* -- a different modelling
structure, and ideally a different analyst or team -- and a model fitted inside
this module from the same frame, by the same process, on the same features
would satisfy none of that. Whatever produced the column (an oversight agency's
appraisals, a second team's AVM, a regression built elsewhere in your own
pipeline), its independence and its reliability are established outside this
module and are the caller's to defend. What is implemented here is the
comparison the appendix asks for once those values exist.

Minimum sample size
-------------------
Appendix B.4 says the concentration method "should not be employed for sample
sizes less than 30". The same floor is applied to every test here: below it the
functions return ``flag=None`` and a "Sample size too small" conclusion rather
than a number that cannot be defended.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, norm, ttest_ind

from . import nw_utils
from .confint import cod_ci, median_ci_rank
from .indicators import asr_median, cod, prd

__all__ = [
    'SC_MIN_N',
    'sc_concentration',
    'sc_cdf_gap',
    'sc_value_change',
    'sc_unit_values',
    'sc_split_sample',
    'sc_mass_appraisal',
    'sc_independent_values',
    'detect_sales_chasing',
    'critical_concentration',
]

# Appendix B.4: "the method should not be employed for sample sizes less
# than 30."
SC_MIN_N = 30

# Appendix B.4, Table B.4.1: the standard deviation column is exactly
# 1.25 x COD. That is the normal-distribution identity COD = E|x - mu| / mu =
# sigma * sqrt(2/pi), i.e. sigma = 1.2533 * COD, rounded to 1.25 in the table.
# The rounded factor is used here so the function reproduces the published
# table to the digit.
_SD_PER_COD = 1.25

# Appendix B.1: "statistically significant differences ... greater than 5%
# likely indicate sales chasing ... However, if value changes for sold
# properties are less than 5%, the tolerable difference should be reduced to
# 3%."
_TOLERANCE_LARGE = 0.05
_TOLERANCE_SMALL = 0.03
_TOLERANCE_SWITCH = 0.05

_TOO_SMALL = 'Sample size too small'
_NO_EVIDENCE = 'No evidence of sales chasing'
_INDICATED = 'Sales chasing indicated'
_INCONCLUSIVE = 'Inconclusive'


# ---------------------------------------------------------------------------------------------------------------------

def _clean(x, name='asr'):
    """1D float array with non-finite values dropped, and the drop count."""
    arr = nw_utils.to_numpy_1d(x).astype(float)
    keep = np.isfinite(arr)
    if not keep.all():
        return arr[keep], int((~keep).sum())
    return arr, 0


# ---------------------------------------------------------------------------------------------------------------------

def critical_concentration(min_achievable_cod=0.04, band=0.02):
    """
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
    """
    if min_achievable_cod is None or min_achievable_cod <= 0:
        raise ValueError('min_achievable_cod must be > 0.')
    if band is None or band <= 0:
        raise ValueError('band must be > 0.')
    sd = float(min_achievable_cod) * _SD_PER_COD
    expected = float(2.0 * norm.cdf(band / sd) - 1.0)
    return {
        'sd': sd,
        'expected_proportion': expected,
        'critical_proportion': float(np.floor(100.0 * expected) + 1.0) / 100.0,
    }


# ---------------------------------------------------------------------------------------------------------------------

def _centre_value(arr, spec):
    """Resolve a centre specification against a sample."""
    if spec == 'median':
        return float(np.median(arr))
    if spec == 'mean':
        return float(arr.mean())
    if spec == 'unity':
        return 1.0
    return float(spec)


def _max_window(sorted_asr, band):
    """Centre and count of the fullest ``±band`` window over sorted ratios.

    Appendix B.4 asks for the concentration "around any measure of central
    tendency, such as the median or a mode" and warns that a bimodal or
    multimodal distribution can hide selective revaluation at each mode. A
    two-pointer sweep finds the densest window anywhere in the sample, so
    the modes do not have to be named in advance.
    """
    n = sorted_asr.size
    best_count, best_lo, best_hi = 0, 0, 0
    hi = 0
    for lo in range(n):
        # Window is [asr[lo], asr[lo] + 2 * band]: any interval of that
        # width containing a maximal set of points can be slid until its
        # left edge sits on a data point, so only these n windows matter.
        limit = sorted_asr[lo] + 2.0 * band
        if hi < lo:
            hi = lo
        while hi + 1 < n and sorted_asr[hi + 1] <= limit:
            hi += 1
        if hi - lo + 1 > best_count:
            best_count, best_lo, best_hi = hi - lo + 1, lo, hi
    centre = 0.5 * (sorted_asr[best_lo] + sorted_asr[best_hi])
    return float(centre), int(best_count)


# ---------------------------------------------------------------------------------------------------------------------

def sc_concentration(asr,
                     center='median',
                     band=0.02,
                     min_achievable_cod=0.04,
                     alpha=0.05,
                     scan_modes=True,
                     min_n=SC_MIN_N,
                     weights=None):
    """
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
    """
    try:
        arr, dropped = _clean(asr)
        n = arr.size
        out = {
            'n': n,
            'n_dropped': dropped,
            'center': center,
            'band': float(band),
            'min_achievable_cod': float(min_achievable_cod),
        }
        if n < min_n:
            out.update(flag=None, conclusion=_TOO_SMALL)
            return out

        w = None
        if weights is not None:
            w = nw_utils.to_numpy_1d(weights).astype(float)
            if w.size == n + dropped and dropped:
                raise ValueError('weights must align with `asr` after non-finite values are dropped.')

        if center == 'median':
            centre = asr_median(arr, weights=w)
        elif center == 'mean':
            centre = float(np.average(arr, weights=w)) if w is not None else float(arr.mean())
        elif center == 'unity':
            centre = 1.0
        else:
            centre = float(center)

        half = float(band) * centre
        lo, hi = centre - half, centre + half
        n_in = int(((arr >= lo) & (arr <= hi)).sum())
        observed = n_in / n

        ref = critical_concentration(min_achievable_cod, band)
        # Exact one-sided binomial: how surprising is n_in successes out of
        # n if the true in-band probability were the reference proportion?
        p_value = float(binomtest(n_in, n, ref['expected_proportion'], alternative='greater').pvalue)

        out.update({
            'center_value': centre,
            'bounds': (lo, hi),
            'n_in_band': n_in,
            'observed_proportion': observed,
            'expected_proportion': ref['expected_proportion'],
            'critical_proportion': ref['critical_proportion'],
            'excess_proportion': observed - ref['critical_proportion'],
            'p_value': p_value,
            'observed_cod': cod(arr),
            'implied_reference_cod': float(min_achievable_cod),
        })

        if scan_modes:
            centre_max, count_max = _max_window(np.sort(arr), half)
            out['max_window_center'] = centre_max
            out['max_window_proportion'] = count_max / n
        else:
            out['max_window_center'] = np.nan
            out['max_window_proportion'] = np.nan

        over_critical = observed >= ref['critical_proportion']
        significant = p_value < alpha
        if over_critical and significant:
            out['flag'] = True
            out['conclusion'] = _INDICATED
        elif over_critical or significant:
            # One condition met but not the other: usually a mid-sized
            # sample where the point estimate clears the table but the
            # binomial cannot rule out sampling noise.
            out['flag'] = False
            out['conclusion'] = _INCONCLUSIVE
        else:
            out['flag'] = False
            out['conclusion'] = _NO_EVIDENCE

        if scan_modes and out['max_window_proportion'] >= ref['critical_proportion'] and not out['flag']:
            out['conclusion'] = (
                f"{_INCONCLUSIVE}: concentration at the centre is acceptable, but "
                f"{out['max_window_proportion']:.1%} of ratios sit within +-{band:.0%} of "
                f"{centre_max:.3f} -- check for a secondary mode"
            )
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: sc_concentration:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def sc_cdf_gap(asr,
               center=('median', 'unity'),
               band=0.05,
               gap=0.03,
               top_k=3,
               min_n=SC_MIN_N):
    """
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
    """
    try:
        arr, dropped = _clean(asr)
        n = arr.size
        out = {'n': n, 'n_dropped': dropped, 'gap_threshold': float(gap), 'band': float(band)}
        if n < min_n:
            out.update(flag=None, conclusion=_TOO_SMALL)
            return out

        specs = [center] if isinstance(center, (str, float, int)) else list(center)
        centres = [_centre_value(arr, s) for s in specs]

        # The ECDF of a sample jumps by (count / n) at each distinct value.
        # Ties are the whole signal here, so work from value counts rather
        # than from differences of the evaluated ECDF.
        values, counts = np.unique(arr, return_counts=True)
        bounds = []
        in_band = np.zeros(values.size, dtype=bool)
        for centre in centres:
            half = float(band) * centre
            lo, hi = centre - half, centre + half
            bounds.append((lo, hi))
            in_band |= (values >= lo) & (values <= hi)
        out.update({'center_value': centres, 'bounds': bounds})

        if not in_band.any():
            out.update(max_gap=0.0, max_gap_at=np.nan, max_gap_n=0, top_gaps=[],
                       flag=False, conclusion=_NO_EVIDENCE)
            return out

        band_values, band_counts = values[in_band], counts[in_band]
        band_gaps = band_counts / n
        order = np.argsort(band_gaps)[::-1][:max(1, int(top_k))]
        out['top_gaps'] = [(float(band_values[i]), float(band_gaps[i]), int(band_counts[i])) for i in order]
        best = order[0]
        out['max_gap'] = float(band_gaps[best])
        out['max_gap_at'] = float(band_values[best])
        out['max_gap_n'] = int(band_counts[best])

        if out['max_gap'] > gap:
            out['flag'] = True
            out['conclusion'] = (
                f"{_INDICATED}: {out['max_gap_n']} of {n} sales ({out['max_gap']:.1%}) "
                f"share the ratio {out['max_gap_at']:.4f}"
            )
        else:
            out['flag'] = False
            out['conclusion'] = _NO_EVIDENCE
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: sc_cdf_gap:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def _practical_tolerance(sold_change):
    """Appendix B.1's sliding tolerance, in the units of the change itself.

    5 percentage points when the sold group moved 5% or more; 3 points when
    it moved less, because at low overall movement a smaller gap is already
    the whole story.
    """
    if not np.isfinite(sold_change):
        return _TOLERANCE_LARGE
    return _TOLERANCE_LARGE if abs(sold_change) >= _TOLERANCE_SWITCH else _TOLERANCE_SMALL


def _two_sample_verdict(sold, unsold, statistic, tolerance, alpha, label):
    """Shared B.1/B.2 machinery: central tendency, both tests, both gates."""
    if statistic == 'median':
        sold_stat, unsold_stat = float(np.median(sold)), float(np.median(unsold))
    else:
        sold_stat, unsold_stat = float(sold.mean()), float(unsold.mean())

    diff = sold_stat - unsold_stat
    # Mann-Whitney U: the standard names it first, and it is the right
    # default -- value changes and unit values are both skewed, and it
    # tests the shift in distribution rather than in the mean.
    try:
        mwu_p = float(mannwhitneyu(sold, unsold, alternative='two-sided').pvalue)
    except ValueError:
        mwu_p = np.nan
    # Welch's t-test, also named in B.2. Reported alongside rather than
    # instead of: agreement between the two is itself informative. A group
    # with no variance at all -- every unsold parcel frozen at last year's
    # value, which is the case B.1 explicitly anticipates -- makes the
    # moment calculation meaningless, so it is reported as missing rather
    # than as a number scipy warns about.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t_p = float(ttest_ind(sold, unsold, equal_var=False).pvalue)
    if not np.isfinite(t_p):
        t_p = np.nan

    # The more favourable of the two tests decides significance; both are
    # reported so a caller can see when they disagree.
    usable_p = [p for p in (mwu_p, t_p) if np.isfinite(p)]
    significant = bool(usable_p) and min(usable_p) < alpha
    practical = abs(diff) > tolerance

    if significant and practical:
        conclusion = (f"{_INDICATED}: {label} differs by {diff:+.1%} "
                      f"(tolerance {tolerance:.0%}), p={min(mwu_p, t_p):.4f}")
        flag = True
    elif significant:
        conclusion = (f"{_NO_EVIDENCE}: difference of {diff:+.1%} is statistically "
                      f"significant but within the {tolerance:.0%} tolerance")
        flag = False
    elif practical:
        conclusion = (f"{_INCONCLUSIVE}: difference of {diff:+.1%} exceeds the "
                      f"{tolerance:.0%} tolerance but is not statistically significant")
        flag = False
    else:
        conclusion = _NO_EVIDENCE
        flag = False

    return {
        'sold_statistic': sold_stat,
        'unsold_statistic': unsold_stat,
        'difference': diff,
        'tolerance': tolerance,
        'mannwhitney_p': mwu_p,
        'ttest_p': t_p,
        'statistically_significant': bool(significant),
        'practically_significant': bool(practical),
        'flag': flag,
        'conclusion': conclusion,
    }


def sc_value_change(df,
                    sold_col,
                    prior_value_col,
                    current_value_col,
                    exclude_col=None,
                    statistic='auto',
                    no_change_threshold=0.5,
                    alpha=0.05,
                    min_n=SC_MIN_N):
    """
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
    """
    try:
        frame = nw_utils.require_pandas(df, reason='sc_value_change() group comparison')
        for col in (sold_col, prior_value_col, current_value_col):
            if col not in frame.columns:
                raise ValueError(f'sc_value_change(): column not found: {col!r}')

        work = frame[[sold_col, prior_value_col, current_value_col] +
                     ([exclude_col] if exclude_col else [])].copy()
        n_excluded = 0
        if exclude_col:
            excluded = work[exclude_col].fillna(False).astype(bool)
            n_excluded = int(excluded.sum())
            work = work.loc[~excluded]

        prior = work[prior_value_col].astype(float)
        current = work[current_value_col].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            change = (current / prior) - 1.0
        is_sold = work[sold_col].fillna(False).astype(bool)
        valid = np.isfinite(change) & (prior > 0)

        sold = change[valid & is_sold].to_numpy()
        unsold = change[valid & ~is_sold].to_numpy()

        out = {
            'n_sold': int(sold.size),
            'n_unsold': int(unsold.size),
            'n_excluded': n_excluded,
        }
        if sold.size < min_n or unsold.size < min_n:
            out.update(flag=None, conclusion=_TOO_SMALL)
            return out

        sold_flat = float(np.mean(np.isclose(sold, 0.0)))
        unsold_flat = float(np.mean(np.isclose(unsold, 0.0)))
        out['sold_pct_no_change'] = sold_flat
        out['unsold_pct_no_change'] = unsold_flat

        chosen = statistic
        if statistic == 'auto':
            chosen = 'mean' if unsold_flat > no_change_threshold else 'median'
        out['statistic'] = chosen

        sold_centre = float(np.median(sold)) if chosen == 'median' else float(sold.mean())
        out.update(_two_sample_verdict(
            sold, unsold, chosen, _practical_tolerance(sold_centre), alpha,
            f'{chosen} value change',
        ))
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: sc_value_change:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def sc_unit_values(df,
                   sold_col,
                   value_col,
                   size_col,
                   exclude_col=None,
                   statistic='median',
                   alpha=0.05,
                   min_n=SC_MIN_N):
    """
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
    """
    try:
        frame = nw_utils.require_pandas(df, reason='sc_unit_values() group comparison')
        for col in (sold_col, value_col, size_col):
            if col not in frame.columns:
                raise ValueError(f'sc_unit_values(): column not found: {col!r}')

        work = frame[[sold_col, value_col, size_col] + ([exclude_col] if exclude_col else [])].copy()
        n_excluded = 0
        if exclude_col:
            excluded = work[exclude_col].fillna(False).astype(bool)
            n_excluded = int(excluded.sum())
            work = work.loc[~excluded]

        value = work[value_col].astype(float)
        size = work[size_col].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit = value / size
        is_sold = work[sold_col].fillna(False).astype(bool)
        valid = np.isfinite(unit) & (size > 0)

        sold = unit[valid & is_sold].to_numpy()
        unsold = unit[valid & ~is_sold].to_numpy()

        out = {'n_sold': int(sold.size), 'n_unsold': int(unsold.size),
               'n_excluded': n_excluded, 'statistic': statistic}
        if sold.size < min_n or unsold.size < min_n:
            out.update(flag=None, conclusion=_TOO_SMALL)
            return out

        unsold_centre = float(np.median(unsold)) if statistic == 'median' else float(unsold.mean())
        if not unsold_centre:
            out.update(flag=None, conclusion='Unsold unit values have a zero centre; cannot compare')
            return out
        # Unit values are in dollars, but the appendix's 5%/3% tolerance is a
        # percentage. Both samples are divided by the unsold centre so the
        # comparison runs in relative units: the difference is then already a
        # fraction, the tolerance applies to it directly, and neither test
        # changes -- Mann-Whitney is rank-based and Welch's t is
        # scale-equivariant.
        scale = abs(unsold_centre)
        sold_rel, unsold_rel = sold / scale, unsold / scale
        sold_centre_rel = float(np.median(sold_rel)) if statistic == 'median' else float(sold_rel.mean())
        relative = sold_centre_rel - (unsold_centre / scale)
        out.update(_two_sample_verdict(
            sold_rel, unsold_rel, statistic, _practical_tolerance(relative), alpha,
            f'{statistic} unit value',
        ))
        out['sold_unit_value'] = float(np.median(sold)) if statistic == 'median' else float(sold.mean())
        out['unsold_unit_value'] = unsold_centre
        out['relative_difference'] = out['difference']
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: sc_unit_values:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def _as_bounds(ci):
    """``(lower, upper)`` from either a pair or a confint-style dict.

    ``cod_ci`` returns a tuple; ``median_ci_rank`` returns a dict carrying
    the ranks and method alongside the bounds. Both are used here.
    """
    if isinstance(ci, dict):
        return float(ci.get('lower', np.nan)), float(ci.get('upper', np.nan))
    lo, hi = ci
    return float(lo), float(hi)


def _ci_usable(*cis):
    """True when every bound of every supplied interval is finite."""
    return all(np.isfinite(_as_bounds(ci)).all() for ci in cis)


def _ci_disjoint(ci_a, ci_b):
    """True when two intervals do not overlap (and all bounds are finite)."""
    a_lo, a_hi = _as_bounds(ci_a)
    b_lo, b_hi = _as_bounds(ci_b)
    if not all(np.isfinite([a_lo, a_hi, b_lo, b_hi])):
        return False
    return a_hi < b_lo or b_hi < a_lo


def sc_split_sample(df,
                    assessment_col,
                    sale_price_col,
                    sale_date_col,
                    cutoff_date,
                    conf_level=0.95,
                    alpha=0.05,
                    tolerance=0.05,
                    min_n=SC_MIN_N):
    """
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
    """
    try:
        frame = nw_utils.require_pandas(df, reason='sc_split_sample() date split')
        for col in (assessment_col, sale_price_col, sale_date_col):
            if col not in frame.columns:
                raise ValueError(f'sc_split_sample(): column not found: {col!r}')

        dates = pd.to_datetime(frame[sale_date_col], errors='coerce')
        cutoff = pd.to_datetime(cutoff_date)
        assessment = frame[assessment_col].astype(float)
        price = frame[sale_price_col].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            asr = (assessment / price).to_numpy()

        usable = np.isfinite(asr) & (asr > 0) & dates.notna().to_numpy()
        known_mask = usable & (dates <= cutoff).to_numpy()
        unknown_mask = usable & (dates > cutoff).to_numpy()
        known, unknown = asr[known_mask], asr[unknown_mask]

        out = {
            'cutoff_date': str(cutoff.date()),
            'n_known': int(known.size),
            'n_unknown': int(unknown.size),
        }
        if known.size < min_n or unknown.size < min_n:
            out.update(flag=None, conclusion=_TOO_SMALL)
            return out

        out['median_known'] = asr_median(known)
        out['median_unknown'] = asr_median(unknown)
        out['cod_known'] = cod(known)
        out['cod_unknown'] = cod(unknown)
        out['median_ci_known'] = _as_bounds(median_ci_rank(known, conf_level))
        out['median_ci_unknown'] = _as_bounds(median_ci_rank(unknown, conf_level))
        out['cod_ci_known'] = cod_ci(known, conf_level)
        out['cod_ci_unknown'] = cod_ci(unknown, conf_level)
        out['cod_ratio'] = out['cod_known'] / out['cod_unknown'] if out['cod_unknown'] else np.nan
        out['mannwhitney_p'] = float(mannwhitneyu(known, unknown, alternative='two-sided').pvalue)
        out['median_ci_disjoint'] = _ci_disjoint(out['median_ci_known'], out['median_ci_unknown'])
        out['cod_ci_disjoint'] = _ci_disjoint(out['cod_ci_known'], out['cod_ci_unknown'])

        # "Better" is directional: a tighter CoD, or a median nearer 1.00,
        # in the sample the agency could still have touched.
        cod_better = out['cod_known'] < out['cod_unknown']
        cod_gap = (out['cod_unknown'] - out['cod_known']) / out['cod_unknown'] if out['cod_unknown'] else 0.0
        level_better = abs(out['median_known'] - 1.0) < abs(out['median_unknown'] - 1.0)

        findings = []
        if cod_better and out['cod_ci_disjoint'] and cod_gap > tolerance:
            findings.append(f"CoD {out['cod_known']:.1%} before the cutoff vs "
                            f"{out['cod_unknown']:.1%} after ({cod_gap:.0%} tighter)")
        if level_better and out['median_ci_disjoint']:
            findings.append(f"median {out['median_known']:.3f} before the cutoff vs "
                            f"{out['median_unknown']:.3f} after")

        if findings:
            out['flag'] = True
            out['conclusion'] = (f"{_INDICATED}: " + '; '.join(findings) +
                                 '. Reject the pre-cutoff statistics; the post-cutoff sample is the valid one')
        elif out['mannwhitney_p'] < alpha and (cod_better or level_better):
            out['flag'] = False
            out['conclusion'] = (f"{_INCONCLUSIVE}: the two subsamples differ (p="
                                 f"{out['mannwhitney_p']:.4f}) but no statistic is both better and "
                                 'outside its tolerance')
        else:
            out['flag'] = False
            out['conclusion'] = _NO_EVIDENCE
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: sc_split_sample:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def _ratio_study_summary(assessment, denominator, conf_level):
    """Level and uniformity for one set of (value, denominator) pairs.

    The denominator is a sale price in one call and an independently
    determined value in the other; everything downstream treats them
    identically, which is exactly the substitution B.5 prescribes.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        asr = assessment / denominator
    keep = np.isfinite(asr) & (asr > 0)
    asr, assessment, denominator = asr[keep], assessment[keep], denominator[keep]
    if asr.size == 0:
        return {'n': 0, 'asr': asr, 'median': np.nan, 'cod': np.nan, 'prd': np.nan,
                'median_ci': (np.nan, np.nan), 'cod_ci': (np.nan, np.nan)}
    return {
        'n': int(asr.size),
        'asr': asr,
        'median': asr_median(asr),
        'cod': cod(asr),
        'prd': prd(denominator, asr),
        'median_ci': _as_bounds(median_ci_rank(asr, conf_level)),
        'cod_ci': cod_ci(asr, conf_level),
    }


def _relative_deviations(summary):
    """``|asr - median| / median`` -- the per-observation term inside CoD."""
    median = summary['median']
    if not np.isfinite(median) or median == 0:
        return np.array([], dtype=float)
    return np.abs(summary['asr'] - median) / median


def _in_standard(value, standard):
    """True when ``value`` sits inside a ``(lower, upper)`` standard."""
    if standard is None or not np.isfinite(value):
        return None
    try:
        lo, hi = standard
    except (TypeError, ValueError):
        return None
    return bool(lo <= value <= hi)


def _compare_ratio_studies(sales, independent, alpha, tolerance, standards, two_sided):
    """Shared B.5 / B.6 comparison of a sales study against an independent one.

    ``two_sided=False`` (B.5) asks only whether the *sales* study looks
    better, because that is the direction sales chasing produces. B.6 is
    two-sided: an oversight agency's proxy values can be chased too, so a
    material difference in either direction is a finding and the cause has
    to be established by investigation rather than assumed.
    """
    out = {
        'n_sales': sales['n'],
        'n_independent': independent['n'],
        'median_sales': sales['median'],
        'median_independent': independent['median'],
        'median_ci_sales': sales['median_ci'],
        'median_ci_independent': independent['median_ci'],
        'cod_sales': sales['cod'],
        'cod_independent': independent['cod'],
        'cod_ci_sales': sales['cod_ci'],
        'cod_ci_independent': independent['cod_ci'],
        'prd_sales': sales['prd'],
        'prd_independent': independent['prd'],
        'median_difference': sales['median'] - independent['median'],
        'cod_difference': sales['cod'] - independent['cod'],
    }
    out['cod_relative_gap'] = (
        (independent['cod'] - sales['cod']) / independent['cod']
        if independent['cod'] else np.nan
    )
    out['mannwhitney_p'] = float(
        mannwhitneyu(sales['asr'], independent['asr'], alternative='two-sided').pvalue
    )
    # A second test, on dispersion rather than on location. Mann-Whitney on
    # the ratios is blind to the signature of thorough chasing -- a spike at
    # the median leaves the location alone and collapses the spread, so the
    # ranks stay balanced and the p-value comes back reassuring while the
    # CoD has gone to zero. Testing the relative absolute deviations from
    # each sample's own median tests the very quantity CoD averages:
    # CoD = mean(|asr - median| / median).
    sales_dev = _relative_deviations(sales)
    indep_dev = _relative_deviations(independent)
    try:
        out['dispersion_p'] = float(
            mannwhitneyu(sales_dev, indep_dev, alternative='two-sided').pvalue
        )
    except ValueError:
        out['dispersion_p'] = np.nan
    out['median_ci_disjoint'] = _ci_disjoint(sales['median_ci'], independent['median_ci'])
    out['cod_ci_disjoint'] = _ci_disjoint(sales['cod_ci'], independent['cod_ci'])

    if standards:
        for name, key in (('median', 'Median'), ('cod', 'CoD')):
            standard = standards.get(key)
            out[f'{name}_standard'] = standard
            out[f'sales_{name}_acceptable'] = _in_standard(out[f'{name}_sales'], standard)
            out[f'independent_{name}_acceptable'] = _in_standard(out[f'{name}_independent'], standard)

    # Direction. B.5 only cares whether the sales study is the flattering
    # one; B.6 takes either direction.
    level_better = abs(sales['median'] - 1.0) < abs(independent['median'] - 1.0)
    cod_tighter = sales['cod'] < independent['cod']

    # Practical significance, in the units each measure is argued in.
    #
    # Level is relative -- the standard reads its tolerances as "5 percent,
    # not 5 percentage points" for a level of assessment -- so the median
    # gap is taken as a fraction of the independent study's median.
    #
    # Uniformity is in percentage points. A relative CoD tolerance is
    # unusable here: independent values carry estimation error that lands
    # in the denominator and inflates the independent study's CoD, so a
    # 9.3%-vs-11.6% pair (2.3 points, well inside anyone's noise) is a 20%
    # relative gap and would flag on the artefact rather than on chasing.
    out['median_relative_difference'] = (
        out['median_difference'] / independent['median'] if independent['median'] else np.nan
    )
    level_material = (np.isfinite(out['median_relative_difference'])
                      and abs(out['median_relative_difference']) > tolerance)
    cod_material = (np.isfinite(out['cod_difference'])
                    and abs(out['cod_difference']) > tolerance)

    # Statistical significance. Disjoint confidence intervals are the
    # primary gate, but a degenerate sample has no usable interval -- a
    # fully chased set of ratios is a spike, its CoD is 0 and `cod_ci`
    # returns (nan, nan), which would silently retire the CoD leg in the
    # very case it exists to catch. Each leg falls back to its own
    # two-sample test when an interval cannot be formed.
    level_separated = out['median_ci_disjoint'] or (
        not _ci_usable(sales['median_ci'], independent['median_ci'])
        and out['mannwhitney_p'] < alpha)
    cod_separated = out['cod_ci_disjoint'] or (
        not _ci_usable(sales['cod_ci'], independent['cod_ci'])
        and np.isfinite(out['dispersion_p']) and out['dispersion_p'] < alpha)

    findings = []
    if level_separated and level_material and (two_sided or level_better):
        findings.append(f"median {sales['median']:.3f} on sales vs "
                        f"{independent['median']:.3f} on independent values "
                        f"({out['median_relative_difference']:+.1%})")
    if cod_separated and cod_material and (two_sided or cod_tighter):
        findings.append(f"CoD {sales['cod']:.1%} on sales vs "
                        f"{independent['cod']:.1%} on independent values "
                        f"({abs(out['cod_difference']) * 100:.1f} points apart)")

    out['findings'] = findings
    out['flag'] = bool(findings)
    return out


# ---------------------------------------------------------------------------------------------------------------------

def sc_mass_appraisal(df,
                      assessment_col,
                      sale_price_col,
                      independent_value_col,
                      sold_col=None,
                      exclude_col=None,
                      conf_level=0.95,
                      alpha=0.05,
                      tolerance=0.05,
                      standards=None,
                      min_n=SC_MIN_N):
    """
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
    """
    try:
        frame = nw_utils.require_pandas(df, reason='sc_mass_appraisal() ratio-study comparison')
        for col in (assessment_col, sale_price_col, independent_value_col):
            if col not in frame.columns:
                raise ValueError(f'sc_mass_appraisal(): column not found: {col!r}')

        keep = np.ones(len(frame), dtype=bool)
        n_excluded = 0
        if exclude_col:
            excluded = frame[exclude_col].fillna(False).astype(bool).to_numpy()
            n_excluded = int(excluded.sum())
            keep &= ~excluded

        assessment = frame[assessment_col].astype(float).to_numpy()
        price = frame[sale_price_col].astype(float).to_numpy()
        indep = frame[independent_value_col].astype(float).to_numpy()
        has_price = np.isfinite(price) & (price > 0)
        has_indep = np.isfinite(indep) & (indep > 0)

        if sold_col:
            if sold_col not in frame.columns:
                raise ValueError(f'sc_mass_appraisal(): column not found: {sold_col!r}')
            is_sold = frame[sold_col].fillna(False).astype(bool).to_numpy()
        else:
            # Documented fallback: a usable sale price means the parcel sold.
            is_sold = has_price

        sales_mask = keep & has_price & is_sold
        indep_mask = keep & has_indep & ~is_sold

        sales = _ratio_study_summary(assessment[sales_mask], price[sales_mask], conf_level)
        independent = _ratio_study_summary(assessment[indep_mask], indep[indep_mask], conf_level)

        out = {'n_sales': sales['n'], 'n_independent': independent['n'], 'n_excluded': n_excluded}
        if sales['n'] < min_n or independent['n'] < min_n:
            out.update(flag=None, conclusion=_TOO_SMALL)
            return out

        out.update(_compare_ratio_studies(sales, independent, alpha, tolerance,
                                          standards, two_sided=False))
        out['n_excluded'] = n_excluded

        if out['flag']:
            out['conclusion'] = (
                f"{_INDICATED}: the sales ratio study is the more flattering of the two -- "
                + '; '.join(out['findings'])
                + '. The unsold parcels are the ones no sale price could have influenced'
            )
        elif out['mannwhitney_p'] < alpha:
            out['flag'] = False
            out['conclusion'] = (f"{_INCONCLUSIVE}: the two studies differ (p="
                                 f"{out['mannwhitney_p']:.4f}) but not in the direction, or by "
                                 'the margin, that sales chasing would produce')
        else:
            out['flag'] = False
            out['conclusion'] = _NO_EVIDENCE
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: sc_mass_appraisal:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def sc_independent_values(df,
                          assessment_col,
                          sale_price_col,
                          independent_value_col,
                          exclude_col=None,
                          conf_level=0.95,
                          alpha=0.05,
                          tolerance=0.05,
                          standards=None,
                          min_n=SC_MIN_N):
    """
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
    """
    try:
        frame = nw_utils.require_pandas(df, reason='sc_independent_values() ratio-study comparison')
        for col in (assessment_col, sale_price_col, independent_value_col):
            if col not in frame.columns:
                raise ValueError(f'sc_independent_values(): column not found: {col!r}')

        keep = np.ones(len(frame), dtype=bool)
        n_excluded = 0
        if exclude_col:
            excluded = frame[exclude_col].fillna(False).astype(bool).to_numpy()
            n_excluded = int(excluded.sum())
            keep &= ~excluded

        assessment = frame[assessment_col].astype(float).to_numpy()
        price = frame[sale_price_col].astype(float).to_numpy()
        indep = frame[independent_value_col].astype(float).to_numpy()

        sales_mask = keep & np.isfinite(price) & (price > 0)
        indep_mask = keep & np.isfinite(indep) & (indep > 0)

        sales = _ratio_study_summary(assessment[sales_mask], price[sales_mask], conf_level)
        independent = _ratio_study_summary(assessment[indep_mask], indep[indep_mask], conf_level)

        out = {'n_sales': sales['n'], 'n_independent': independent['n'],
               'n_overlap': int((sales_mask & indep_mask).sum()), 'n_excluded': n_excluded}
        if sales['n'] < min_n or independent['n'] < min_n:
            out.update(flag=None, conclusion=_TOO_SMALL)
            return out

        n_overlap = out['n_overlap']
        out.update(_compare_ratio_studies(sales, independent, alpha, tolerance,
                                          standards, two_sided=True))
        out['n_overlap'] = n_overlap
        out['n_excluded'] = n_excluded

        if out['flag']:
            out['conclusion'] = (
                'Value chasing indicated: ' + '; '.join(out['findings'])
                + '. Either the selling parcels or the independently valued ones were '
                  'treated differently; investigation is needed to establish which'
            )
        elif out['mannwhitney_p'] < alpha:
            out['flag'] = False
            out['conclusion'] = (f"{_INCONCLUSIVE}: the two sets of ratios differ (p="
                                 f"{out['mannwhitney_p']:.4f}) but no level or uniformity "
                                 'measure separates beyond its confidence interval')
        else:
            out['flag'] = False
            out['conclusion'] = _NO_EVIDENCE
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: sc_independent_values:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def detect_sales_chasing(df=None,
                         assessment_col=None,
                         sale_price_col=None,
                         asr=None,
                         group_by=None,
                         sold_col=None,
                         prior_value_col=None,
                         current_value_col=None,
                         size_col=None,
                         sale_date_col=None,
                         cutoff_date=None,
                         independent_value_col=None,
                         independent_technique='auto',
                         exclude_col=None,
                         min_achievable_cod=0.04,
                         band=0.02,
                         alpha=0.05,
                         tolerance=0.05,
                         conf_level=0.95,
                         standards=None,
                         min_n=SC_MIN_N):
    """
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
    """
    try:
        if independent_technique not in ('auto', 'B.5', 'B.6', 'both'):
            raise ValueError("detect_sales_chasing(): independent_technique must be one of "
                             "'auto', 'B.5', 'B.6', 'both'.")
        rows = []

        def add(technique, reference, scope, result, statistic=None, threshold=None,
                p_value=None, n=None):
            rows.append({
                'technique': technique,
                'reference': reference,
                'scope': scope,
                'n': n if n is not None else result.get('n', result.get('n_sold')),
                'statistic': statistic,
                'threshold': threshold,
                'p_value': p_value,
                'flag': result.get('flag'),
                'conclusion': result.get('conclusion'),
                'detail': result,
            })

        def not_run(technique, reference, reason):
            rows.append({
                'technique': technique, 'reference': reference, 'scope': 'overall',
                'n': None, 'statistic': None, 'threshold': None, 'p_value': None,
                'flag': None, 'conclusion': f'not run: {reason}', 'detail': {},
            })

        frame = None
        if df is not None:
            frame = nw_utils.require_pandas(df, reason='detect_sales_chasing()')

        # -- scopes: the whole sample, then each stratum -------------------
        # Frames rather than ratio arrays, because B.1/B.2/B.5/B.6 need
        # columns the ratio does not carry.
        frame_scopes = []
        if frame is not None:
            frame_scopes.append(('overall', frame))
            if group_by:
                keys = [group_by] if isinstance(group_by, str) else list(group_by)
                # A one-element list makes pandas emit each group label as a
                # 1-tuple (and warn about changing to do so); pass the bare
                # column so single- and multi-key grouping label the same way.
                by = keys[0] if len(keys) == 1 else keys
                for label, sub in frame.groupby(by, dropna=False):
                    name = ' | '.join(str(v) for v in (label if isinstance(label, tuple) else (label,)))
                    frame_scopes.append((name, sub))
        elif group_by:
            raise ValueError('detect_sales_chasing(): `group_by` requires `df`, not `asr`.')

        # -- B.4 and the CDF screen ----------------------------------------
        ratio_scopes = []
        if asr is not None:
            ratio_scopes.append(('overall', nw_utils.to_numpy_1d(asr).astype(float)))
        elif frame is not None and assessment_col and sale_price_col:
            for scope, sub in frame_scopes:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = sub[assessment_col].astype(float) / sub[sale_price_col].astype(float)
                ratio_scopes.append((scope, ratios.to_numpy()))

        if ratio_scopes:
            for scope, values in ratio_scopes:
                conc = sc_concentration(values, band=band, min_achievable_cod=min_achievable_cod,
                                        alpha=alpha, min_n=min_n)
                add('Ratio concentration', 'B.4', scope, conc,
                    statistic=conc.get('observed_proportion'),
                    threshold=conc.get('critical_proportion'),
                    p_value=conc.get('p_value'))
                gapres = sc_cdf_gap(values, min_n=min_n)
                add('CDF discontinuity', '--', scope, gapres,
                    statistic=gapres.get('max_gap'), threshold=gapres.get('gap_threshold'))
        else:
            not_run('Ratio concentration', 'B.4', 'need `asr`, or `assessment_col` and `sale_price_col`')
            not_run('CDF discontinuity', '--', 'need `asr`, or `assessment_col` and `sale_price_col`')

        # -- B.1 -----------------------------------------------------------
        if frame is not None and sold_col and prior_value_col and current_value_col:
            for scope, sub in frame_scopes:
                res = sc_value_change(sub, sold_col, prior_value_col, current_value_col,
                                      exclude_col=exclude_col, alpha=alpha, min_n=min_n)
                add('Value change, sold vs unsold', 'B.1', scope, res,
                    statistic=res.get('difference'), threshold=res.get('tolerance'),
                    p_value=res.get('mannwhitney_p'))
        else:
            not_run('Value change, sold vs unsold', 'B.1',
                    'need `sold_col`, `prior_value_col`, `current_value_col`')

        # -- B.2 -----------------------------------------------------------
        if frame is not None and sold_col and size_col and assessment_col:
            for scope, sub in frame_scopes:
                res = sc_unit_values(sub, sold_col, assessment_col, size_col,
                                     exclude_col=exclude_col, alpha=alpha, min_n=min_n)
                add('Unit value, sold vs unsold', 'B.2', scope, res,
                    statistic=res.get('relative_difference'), threshold=res.get('tolerance'),
                    p_value=res.get('mannwhitney_p'))
        else:
            not_run('Unit value, sold vs unsold', 'B.2',
                    'need `sold_col`, `size_col`, `assessment_col`')

        # -- B.3 -----------------------------------------------------------
        if frame is not None and sale_date_col and cutoff_date is not None and assessment_col and sale_price_col:
            for scope, sub in frame_scopes:
                res = sc_split_sample(sub, assessment_col, sale_price_col, sale_date_col,
                                      cutoff_date, conf_level=conf_level, alpha=alpha,
                                      tolerance=tolerance, min_n=min_n)
                add('Split sample', 'B.3', scope, res, n=res.get('n_known'),
                    statistic=res.get('cod_ratio'), p_value=res.get('mannwhitney_p'))
        else:
            not_run('Split sample', 'B.3', 'need `sale_date_col` and `cutoff_date`')

        # -- B.5 / B.6 -----------------------------------------------------
        if frame is not None and independent_value_col and assessment_col and sale_price_col:
            wanted = ({'B.5'} if independent_technique == 'B.5'
                      else {'B.6'} if independent_technique == 'B.6'
                      else {'B.5', 'B.6'} if independent_technique == 'both'
                      else {'B.5'} if sold_col else {'B.6'})
            if 'B.5' in wanted:
                for scope, sub in frame_scopes:
                    res = sc_mass_appraisal(sub, assessment_col, sale_price_col,
                                            independent_value_col, sold_col=sold_col,
                                            exclude_col=exclude_col, conf_level=conf_level,
                                            alpha=alpha, tolerance=tolerance,
                                            standards=standards, min_n=min_n)
                    add('Independent values, unsold parcels', 'B.5', scope, res,
                        n=res.get('n_independent'), statistic=res.get('median_difference'),
                        p_value=res.get('mannwhitney_p'))
            if 'B.6' in wanted:
                for scope, sub in frame_scopes:
                    res = sc_independent_values(sub, assessment_col, sale_price_col,
                                                independent_value_col, exclude_col=exclude_col,
                                                conf_level=conf_level, alpha=alpha,
                                                tolerance=tolerance, standards=standards,
                                                min_n=min_n)
                    add('Independent vs selling ratios', 'B.6', scope, res,
                        n=res.get('n_independent'), statistic=res.get('median_difference'),
                        p_value=res.get('mannwhitney_p'))
        else:
            not_run('Independent values', 'B.5 / B.6',
                    'need `independent_value_col`, `assessment_col`, `sale_price_col`')

        out = pd.DataFrame(rows)
        flagged = int(out['flag'].eq(True).sum())
        if flagged:
            warnings.warn(
                f'{flagged} sales-chasing screen(s) flagged. Appendix B: investigate before '
                'concluding that sales chasing has occurred.'
            )
        return out
    except Exception as ex:
        raise RuntimeError('error in ratio study: detect_sales_chasing:', ex) from ex
