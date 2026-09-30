import difflib
import logging
import math
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import skew, norm, kurtosis, randint, binom
import warnings
from . import nw_utils

# -------------------------------
#  Confidence Intervals Functions
# -------------------------------

# Indicators that get a bootstrapped confidence interval by default, in
# ``ratios()`` and ``ratios_report()``. Intersected with the study's own
# ``indicators`` so a narrower study does not pay for CIs it cannot show.
CI_DEFAULT_INDICATORS = ('Median', 'CoD', 'PRD', 'PRB')

# Every indicator ``_bootstrap_analysis`` can produce a ``<name>_ci`` for.
CI_SUPPORTED_INDICATORS = (
    'Median', 'Mean', 'CoD', 'CoD_rsd', 'CoDw', 'PRB', 'PRD', 'RMSE',
    'CoC_10', 'CoC_15', 'CoC_20', 'MAE', 'MAPE', 'MdAPE', 'CoV',
    'Gini_Sale_Prices', 'Gini_Assessments', 'Concentration_Index',
    'Redistributive_Effect', 'Total_ASR', 'MKI', 'RRI',
    'Modified_Kakwani_Index', 'Reranking_Index',
)

# ``ci_mode='distribution'`` builds every interval from a closed-form
# sampling distribution instead of by resampling. `Median`, `Mean`,
# `Weighted_Mean` and `CoD` each have a formula of their own, prescribed by
# the IAAO Exposure Draft (May 2026) Appendix D. `PRB` and `PRD` are not in
# Appendix D, but neither needs the bootstrap to get an interval: `PRB`
# keeps the interval its own slope regression always produces (see `prb`
# in `indicators.py`), and `PRD` -- the ratio of `Mean` to `Weighted_Mean`
# -- gets a delta-method interval (`prd_ci`, below) built from the other
# two formulas already here. There is no general-purpose distribution CI
# beyond these six, so the set is not extensible the way
# `CI_SUPPORTED_INDICATORS` is.
DISTRIBUTION_CI_SUPPORTED_INDICATORS = (
    'Median', 'Mean', 'Weighted_Mean', 'CoD', 'PRB', 'PRD',
)

# Distribution-mode counterpart of `CI_DEFAULT_INDICATORS`. All six are on
# by default because none of them costs more than a pass over the ratios.
DISTRIBUTION_CI_DEFAULT_INDICATORS = DISTRIBUTION_CI_SUPPORTED_INDICATORS

# The two families of interval `ratios()` / `ratios_report()` can build,
# and the spellings accepted for each.
_CI_MODES = {
    'bootstrap': 'bootstrap',
    'boot': 'bootstrap',
    'bootstrapped': 'bootstrap',
    'bootstrapping': 'bootstrap',
    'resample': 'bootstrap',
    'resampling': 'bootstrap',
    'distribution': 'distribution',
    'distributional': 'distribution',
    'dist': 'distribution',
    'analytic': 'distribution',
    'analytical': 'distribution',
    'closed_form': 'distribution',
    'closed-form': 'distribution',
    'formula': 'distribution',
    'parametric': 'distribution',
}

# Sample size at or below which the median confidence interval switches from
# the large-sample rank formula (Exposure Draft Appendix D.2) to the exact
# binomial tables (Appendix D.4).
MEDIAN_CI_SMALL_SAMPLE_MAX = 30

# Binomial ("Clapp 1989") interpolation parameters for the small-sample
# median confidence interval, Exposure Draft Tables D.4.1 (90 %) and D.4.2
# (95 %). Keyed by sample size; each value is ``(weight, rank_a, rank_b)``
# and means ``weight * R[rank_a] + (1 - weight) * R[rank_b]`` for the lower
# bound, with ``R`` the ratios in ascending order and the ranks 1-indexed.
#
# Only the lower bound is stored. The published upper bound is always the
# mirror image of the lower one -- the same weight applied to the ranks
# counted in from the top -- so deriving it as ``R[n + 1 - rank]`` keeps the
# interval symmetric in rank by construction and repairs two arithmetic
# slips in the printed 95 % table, where n=28 lists .7221/.2729 (sums to
# .9950) and n=29's upper row lists .3062 against .3063 in its lower row.
_MEDIAN_CI_CLAPP_LOWER = {
    90: {
        5:  (0.8800, 1, 2),
        6:  (0.6333, 1, 2),
        7:  (0.2286, 1, 2),
        8:  (0.8643, 2, 3),
        9:  (0.5667, 2, 3),
        10: (0.1067, 2, 3),
        11: (0.7855, 3, 4),
        12: (0.4282, 3, 4),
        13: (0.9558, 4, 5),
        14: (0.6511, 4, 5),
        15: (0.2217, 4, 5),
        16: (0.8261, 5, 6),
        17: (0.4603, 5, 6),
        18: (0.9735, 6, 7),
        19: (0.6480, 6, 7),
        20: (0.2072, 6, 7),
        21: (0.8048, 7, 8),
        22: (0.4156, 7, 8),
        23: (0.9413, 8, 9),
        24: (0.5884, 8, 9),
        25: (0.1203, 8, 9),
        26: (0.7371, 9, 10),
        27: (0.3161, 9, 10),
        28: (0.8687, 10, 11),
        29: (0.4831, 10, 11),
        30: (0.9876, 11, 12),
    },
    95: {
        6:  (0.9000, 1, 2),
        7:  (0.6857, 1, 2),
        8:  (0.3250, 1, 2),
        9:  (0.9222, 2, 3),
        10: (0.6756, 2, 3),
        11: (0.2873, 2, 3),
        12: (0.8936, 3, 4),
        13: (0.6056, 3, 4),
        14: (0.1659, 3, 4),
        15: (0.8218, 4, 5),
        16: (0.4827, 4, 5),
        17: (0.9899, 5, 6),
        18: (0.7076, 5, 6),
        19: (0.3059, 5, 6),
        20: (0.8835, 6, 7),
        21: (0.5479, 6, 7),
        22: (0.0697, 6, 7),
        23: (0.7381, 7, 8),
        24: (0.3373, 7, 8),
        25: (0.8958, 8, 9),
        26: (0.5481, 8, 9),
        27: (0.0677, 8, 9),
        28: (0.7221, 9, 10),
        29: (0.3063, 9, 10),
        30: (0.8709, 10, 11),
    },
}

# Confidence levels the published binomial tables cover, as percentages.
_MEDIAN_CI_CLAPP_LEVELS = tuple(sorted(_MEDIAN_CI_CLAPP_LOWER))

# "The caller did not mention this option", as distinct from a deliberate
# None (which several options give their own meaning to).
_UNSET = object()

# Sample sizes for which the published tables put the confidence limits on
# the very lowest and highest ratios in the sample. Property Appraisal and
# Assessment Administration (IAAO 1990, 609) advises caution there: the
# interval is bounded by single observations and says little about the
# median.
_MEDIAN_CI_CLAPP_EXTREME_N = {90: (5, 6, 7), 95: (6, 7, 8)}

# Canonical ``bootstrap_args`` keys and their accepted short forms, so that
# ``bootstrap_args={'groups': True, 'level': 0.9}`` works as well as the
# fully spelled-out ``{'bootstrap_cis_for_groups': True,
# 'bootstrap_confint_level': 0.9}``.
_CI_ARG_CANONICAL = (
    'bootstrap_cis_for_population', 'bootstrap_cis_for_groups',
    'bootstrap_method', 'bootstrap_resamplings', 'bootstrap_confint_level',
    'ci_only', 'bootstrap_concurrency', 'bootstrap_indicators',
    'min_count_threshold', 'bootstrap_seed', 'ci_mode',
)

_CI_ARG_ALIASES = {
    'population': 'bootstrap_cis_for_population',
    'overall': 'bootstrap_cis_for_population',
    'cis_for_population': 'bootstrap_cis_for_population',
    'for_population': 'bootstrap_cis_for_population',
    'groups': 'bootstrap_cis_for_groups',
    'cis_for_groups': 'bootstrap_cis_for_groups',
    'for_groups': 'bootstrap_cis_for_groups',
    'method': 'bootstrap_method',
    'resamplings': 'bootstrap_resamplings',
    'resamples': 'bootstrap_resamplings',
    'resample_pct': 'bootstrap_resample_pct',
    'resample_percent': 'bootstrap_resample_pct',
    'resample_size_pct': 'bootstrap_resample_pct',
    'sample_pct': 'bootstrap_resample_pct',
    'pct': 'bootstrap_resample_pct',
    'level': 'bootstrap_confint_level',
    'confint_level': 'bootstrap_confint_level',
    'confidence_level': 'bootstrap_confint_level',
    'indicators': 'bootstrap_indicators',
    'concurrency': 'bootstrap_concurrency',
    'min_count': 'min_count_threshold',
    'min_count_treshold': 'min_count_threshold',
    'seed': 'bootstrap_seed',
    'random_state': 'bootstrap_seed',
    'random_seed': 'bootstrap_seed',
    'mode': 'ci_mode',
    'family': 'ci_mode',
}

# Words accepted by ``confidence_intervals_aggs`` for the on/off switches.
_CI_SCOPE_WORDS = {
    'none': (False, False), 'off': (False, False), 'no': (False, False),
    'false': (False, False), '': (False, False),
    'overall': (True, False), 'population': (True, False),
    'pop': (True, False), 'on': (True, False), 'yes': (True, False),
    'true': (True, False),
    'groups': (False, True), 'group': (False, True),
    'both': (True, True), 'all': (True, True),
}

# ---------------------------------------------------------------------------------------------------------------------

# No real ratio study reports an interval narrower than this, so a level
# resolving below it is treated as a significance level passed by mistake
# rather than as a deliberate request. 0.5 is well clear of the lowest
# level anyone uses in practice (0.68, one standard error) and well above
# the significance levels that get confused with it (0.01, 0.05, 0.10).
CI_LEVEL_FLOOR = 0.5

# Floor on the size of a single bootstrap resample. `bootstrap_resample_pct`
# below 100% draws fewer sales per resample; below roughly 20 the indicators
# themselves stop being meaningful (a CoD on 8 sales is noise), so the
# percentage is applied and then raised back to this floor -- or to n, if the
# whole sample is smaller than the floor.
BOOTSTRAP_MIN_RESAMPLE_SIZE = 20


def confidence_level_as_fraction(level, argument='ci_conf_level'):
    """
    Normalise a confidence level to a fraction in ``(0, 1)``.

    The one gate every confidence level in the package passes through, so
    that ``0.95`` and ``95`` mean the same thing wherever a level is
    accepted and a significance level passed by mistake is caught rather
    than silently honoured.

    Parameters
    ----------
    level : float or int
        Confidence level, either as a fraction (``0.95``) or as a
        percentage (``95``). Values above 1 are read as percentages.
    argument : str, default 'ci_conf_level'
        Name of the argument being validated, used in the error message so
        it names what the caller actually passed. Always pass this at a
        call site that is validating a differently-named parameter.

    Returns
    -------
    float
        The confidence level as a fraction in ``[CI_LEVEL_FLOOR, 1)``.

    Raises
    ------
    ValueError
        If `level` is not a number, does not resolve into ``(0, 1)``, or
        resolves below `CI_LEVEL_FLOOR`. The last case almost always means
        a significance level was passed where a confidence level was
        wanted -- ``ci_conf_level=0.05`` rather than ``ci_conf_level=0.95`` -- so the
        message says so instead of quietly building a 5 % interval.

    Examples
    --------
    >>> confidence_level_as_fraction(0.95), confidence_level_as_fraction(95)
    (0.95, 0.95)
    >>> confidence_level_as_fraction(90)
    0.9
    >>> confidence_level_as_fraction(0.05)
    Traceback (most recent call last):
        ...
    ValueError: ci_conf_level=0.05 is a significance level, not a confidence level. Use 0.95 for a 95% interval (or pass 95). Confidence levels below 0.5 are rejected; `req_significance` is the argument that takes 0.05.
    """
    try:
        value = float(level)
    except (TypeError, ValueError):
        raise ValueError(
            f'{argument} must be a number, got {level!r}'
        ) from None

    # A percentage, e.g. 95 or 90. Values in (0, 1] are already fractions.
    if value > 1:
        value = value / 100.0

    if not 0 < value < 1:
        raise ValueError(
            f'{argument}={level!r} must be between 0 and 1, or a percentage '
            'such as 95'
        )

    if value < CI_LEVEL_FLOOR:
        raise ValueError(
            f'{argument}={level!r} is a significance level, not a confidence '
            f'level. Use {1 - value:g} for a {(1 - value) * 100:g}% interval '
            f'(or pass {(1 - value) * 100:g}). Confidence levels below '
            f'{CI_LEVEL_FLOOR:g} are rejected; `req_significance` is the '
            'argument that takes 0.05.'
        )

    return value


# Retained under the old private name: `indicators` imports it, and so may
# anything else pinned to a released version.
_ci_level_as_fraction = confidence_level_as_fraction


# ---------------------------------------------------------------------------------------------------------------------

def _canonical_ci_args(user_args, messages):
    """
    Map short-form ``bootstrap_args`` keys to their canonical names.

    Unrecognised keys are dropped and reported in `messages` with the
    closest canonical name as a suggestion, so silent typos such as
    ``'boostrap_indicators'`` surface instead of being ignored.

    Parameters
    ----------
    user_args : dict
        Bootstrap options as supplied by the caller.
    messages : list of str
        Accumulator appended to for every key that was renamed or dropped.

    Returns
    -------
    dict
        Options keyed by canonical ``bootstrap_args`` names.
    """
    canonical = {}
    known = set(_CI_ARG_CANONICAL)
    for key, value in (user_args or {}).items():
        if key in known:
            canonical[key] = value
            continue
        alias = _CI_ARG_ALIASES.get(str(key).strip().lower().lstrip('_'))
        if alias:
            canonical[alias] = value
            continue
        suggestions = difflib.get_close_matches(
            str(key), sorted(known | set(_CI_ARG_ALIASES)), n=1, cutoff=0.6
        )
        hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ''
        messages.append(
            f"input error: unknown bootstrap_args key '{key}' was ignored.{hint}"
        )
    return canonical


# ---------------------------------------------------------------------------------------------------------------------

def normalize_ci_args(bootstrap_args=None
                     , confidence_intervals_aggs=None
                     , ci_conf_level=None
                     , ci_indicators=None
                     , ci_seed=None
                     , ci_mode=None
                     , study_indicators=None
                     , has_groups=False
                     , defaults=None):
    """
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
    """
    messages = []

    resolved = {
        'bootstrap_cis_for_population': True,
        'bootstrap_cis_for_groups': False,
        'bootstrap_method': 'BCa',
        'bootstrap_resamplings': 1000,
        'bootstrap_resample_pct': 1.0,
        # 90 %, two-tailed: the level the Standard on Ratio Studies asks for
        # when an interval decides compliance (s.11.1.2 and s.13.6.1.3), and
        # the level its own demonstration report is printed at (Table 10.2).
        # It is also what `vei_conf_level` and `vem_conf_level` default to in
        # `ratios`, so one study no longer mixes a 90 % VEI interval with 95 %
        # intervals on everything else.
        'bootstrap_confint_level': 0.9,
        'ci_only': True,
        'bootstrap_concurrency': None,
        # ``_UNSET`` rather than a list, because an explicit
        # ``bootstrap_indicators=None`` has its own meaning -- "every study
        # indicator that supports an interval" -- and has to stay
        # distinguishable from "the caller never said".
        'bootstrap_indicators': _UNSET,
        'min_count_threshold': 20,
        'bootstrap_seed': None,
        'ci_mode': 'distribution',
    }
    if defaults:
        resolved.update(_canonical_ci_args(defaults, messages))

    user = _canonical_ci_args(bootstrap_args, messages)
    resolved.update(user)
    indicators_were_set = 'bootstrap_indicators' in user

    # -- high-level switches, applied last so they always win ----------
    if isinstance(confidence_intervals_aggs, dict):
        overrides = _canonical_ci_args(confidence_intervals_aggs, messages)
        resolved.update(overrides)
        indicators_were_set = indicators_were_set or 'bootstrap_indicators' in overrides
    elif confidence_intervals_aggs is not None:
        if isinstance(confidence_intervals_aggs, bool):
            scope = _CI_SCOPE_WORDS['on' if confidence_intervals_aggs else 'off']
        elif isinstance(confidence_intervals_aggs, str):
            word = confidence_intervals_aggs.strip().lower()
            if word not in _CI_SCOPE_WORDS:
                raise ValueError(
                    f'confidence_intervals_aggs={confidence_intervals_aggs!r} is not recognised; '
                    f'expected one of {sorted(w for w in _CI_SCOPE_WORDS if w)}, '
                    'a bool, a list of indicator names, or a dict of bootstrap options.'
                )
            scope = _CI_SCOPE_WORDS[word]
        elif isinstance(confidence_intervals_aggs, (list, tuple, set, frozenset)):
            scope = (True, False)
            if ci_indicators is None:
                ci_indicators = list(confidence_intervals_aggs)
        else:
            raise ValueError(
                f'confidence_intervals_aggs={confidence_intervals_aggs!r} is not recognised; '
                'expected a bool, a scope name, a list of indicator names, or a dict.'
            )
        resolved['bootstrap_cis_for_population'], resolved['bootstrap_cis_for_groups'] = scope

    if ci_conf_level is not None:
        resolved['bootstrap_confint_level'] = confidence_level_as_fraction(ci_conf_level, 'ci_conf_level')
    if ci_indicators is not None:
        resolved['bootstrap_indicators'] = (
            [ci_indicators] if isinstance(ci_indicators, str) else list(ci_indicators)
        )
        indicators_were_set = True
    if ci_seed is not None:
        resolved['bootstrap_seed'] = ci_seed
    if ci_mode is not None:
        resolved['ci_mode'] = ci_mode

    # -- validation and cross-checks -----------------------------------
    # The mode is settled first: it decides which indicators can carry an
    # interval, so every check below has to know it.
    mode_word = str(resolved['ci_mode']).strip().lower().replace(' ', '_')
    if mode_word in _CI_MODES:
        resolved['ci_mode'] = _CI_MODES[mode_word]
    else:
        messages.append(
            f"input error: ci_mode={resolved['ci_mode']!r} is not recognised; "
            "using 'distribution'. Supported modes: 'bootstrap', 'distribution'."
        )
        resolved['ci_mode'] = 'distribution'
    distribution_mode = resolved['ci_mode'] == 'distribution'
    supported = (DISTRIBUTION_CI_SUPPORTED_INDICATORS if distribution_mode
                 else CI_SUPPORTED_INDICATORS)

    method = str(resolved['bootstrap_method'])
    matched = [m for m in ('BCa', 'percentile') if m.lower() == method.lower()]
    if matched:
        resolved['bootstrap_method'] = matched[0]
    else:
        messages.append(
            f"input error: bootstrap method '{method}' is not supported; using 'BCa'. "
            "Supported methods: 'BCa', 'percentile'."
        )
        resolved['bootstrap_method'] = 'BCa'

    resolved['bootstrap_confint_level'] = confidence_level_as_fraction(
        resolved['bootstrap_confint_level'], 'bootstrap_confint_level'
    )

    try:
        resamplings = int(resolved['bootstrap_resamplings'])
    except (TypeError, ValueError):
        resamplings = 0
    if resamplings < 10:
        messages.append(
            f"input error: bootstrap_resamplings={resolved['bootstrap_resamplings']!r} "
            'is too low for a confidence interval; using 1000.'
        )
        resamplings = 1000
    resolved['bootstrap_resamplings'] = resamplings

    # Share of the sample drawn per resample. Accepted as a fraction (0.2) or
    # a percentage (20), the same as a confidence level, because both readings
    # are natural and only one of them is ever meant.
    pct = resolved['bootstrap_resample_pct']
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        messages.append(
            f"input error: bootstrap_resample_pct={resolved['bootstrap_resample_pct']!r} "
            'is not a number; using 100% of the sample.'
        )
        pct = 1.0
    if pct > 1:
        pct = pct / 100
    if not 0 < pct <= 1:
        messages.append(
            f"input error: bootstrap_resample_pct={resolved['bootstrap_resample_pct']!r} "
            'is outside (0, 1]; using 100% of the sample.'
        )
        pct = 1.0
    resolved['bootstrap_resample_pct'] = pct

    study = list(study_indicators) if study_indicators else []
    requested = resolved['bootstrap_indicators']
    if requested is _UNSET:
        # Nobody asked, so take the mode's own default list.
        requested = list(DISTRIBUTION_CI_DEFAULT_INDICATORS if distribution_mode
                         else CI_DEFAULT_INDICATORS)
    elif requested is None:
        # A deliberate None means "everything this mode can do".
        requested = [x for x in study if x in supported]
    elif isinstance(requested, str):
        requested = [requested]
    else:
        requested = list(requested)

    unsupported = [x for x in requested if x not in supported]
    if unsupported and indicators_were_set:
        family = ('distribution-based' if distribution_mode else 'bootstrap')
        messages.append(
            f'input error: no {family} confidence interval is available for '
            f'{unsupported}; they were dropped. Available: {list(supported)}.'
        )
    requested = [x for x in requested if x in supported]

    # Indicators the study has to start computing for the requested intervals
    # to have a point estimate to sit beside. Output only -- the caller never
    # sets this key; `ratios` / `ratios_report` read it back out.
    additions = []
    cis_on = bool(resolved['bootstrap_cis_for_population']
                  or resolved['bootstrap_cis_for_groups'])
    if study:
        missing = [x for x in requested if x not in study]
        if missing and distribution_mode and cis_on and indicators_were_set:
            # The caller asked for these intervals by name, and every
            # distribution-mode estimator is one pass over the ratios, so the
            # study takes the indicator on rather than dropping the interval
            # the caller asked for. Two limits keep this from widening a study
            # nobody widened: the bootstrap still drops (a resampled interval
            # on an indicator nobody asked for is real work), and a default CI
            # indicator list is still intersected, so choosing a narrow
            # `indicators` list keeps the results table narrow.
            additions = list(dict.fromkeys(missing))
            study = study + additions
        elif missing and indicators_were_set:
            messages.append(
                f'input error: confidence intervals were requested for {missing}, '
                'which the study does not compute; add them to `indicators` to see '
                'those intervals.'
            )
        requested = [x for x in requested if x in study]

    resolved['bootstrap_indicators'] = list(dict.fromkeys(requested))
    resolved['ci_study_indicator_additions'] = additions

    if not resolved['bootstrap_indicators'] and (
        resolved['bootstrap_cis_for_population'] or resolved['bootstrap_cis_for_groups']
    ):
        messages.append(
            'input error: confidence intervals were requested but no indicator '
            'supports one; skipping the '
            + ('distribution-based intervals.' if distribution_mode else 'bootstrap.')
        )
        resolved['bootstrap_cis_for_population'] = False
        resolved['bootstrap_cis_for_groups'] = False

    if resolved['bootstrap_cis_for_groups'] and not has_groups:
        messages.append(
            'input error: per-group confidence intervals were requested, but no '
            'group_by was provided. Proceeding with overall confidence intervals only.'
        )
        resolved['bootstrap_cis_for_groups'] = False

    try:
        resolved['min_count_threshold'] = max(int(resolved['min_count_threshold']), 4)
    except (TypeError, ValueError):
        resolved['min_count_threshold'] = 20

    return resolved, messages


def describe_ci_settings(resolved, count=None):
    """
    Render the resolved confidence-interval settings as readable lines.

    Most of what governs an interval is a default the caller never typed, so
    a published number can rest on settings nobody looked at. This lists every
    one of them and marks whether it came from the caller or from the package,
    which is the visibility `print_ci_settings` exists to give.

    Parameters
    ----------
    resolved : dict
        Canonical bootstrap arguments, as returned by `normalize_ci_args`.
    count : int, optional
        Sample size. When given, the resample line reports the actual number
        of sales drawn, including the `BOOTSTRAP_MIN_RESAMPLE_SIZE` floor.

    Returns
    -------
    str
        A multi-line summary, or ``''`` when no interval will be produced.
    """
    on_pop = resolved.get('bootstrap_cis_for_population')
    on_grp = resolved.get('bootstrap_cis_for_groups')
    if not (on_pop or on_grp):
        return 'Confidence intervals: off.'

    distribution = resolved.get('ci_mode') == 'distribution'
    default_indicators = list(DISTRIBUTION_CI_DEFAULT_INDICATORS if distribution
                              else CI_DEFAULT_INDICATORS)

    def mark(key, value, default):
        return '' if value == default else '   <-- set'

    rows = [
        ('Mode', resolved.get('ci_mode'), mark('ci_mode', resolved.get('ci_mode'), 'distribution')),
        ('Level', f"{resolved.get('bootstrap_confint_level', 0):.0%}",
         mark('level', resolved.get('bootstrap_confint_level'), 0.9)),
        ('Indicators', ', '.join(resolved.get('bootstrap_indicators') or []) or 'none',
         mark('ind', list(resolved.get('bootstrap_indicators') or []), default_indicators)),
        ('Overall intervals', 'on' if on_pop else 'off', mark('pop', bool(on_pop), True)),
        ('Per-group intervals', 'on' if on_grp else 'off', mark('grp', bool(on_grp), False)),
        ('Minimum sales per row', f"{resolved.get('min_count_threshold'):,}",
         mark('min', resolved.get('min_count_threshold'), 20)),
    ]

    if distribution:
        rows.append(('Estimators',
                     'Appendix D closed forms, plus PRB (regression) and PRD '
                     '(delta method); bootstrap options ignored', ''))
    else:
        pct = resolved.get('bootstrap_resample_pct', 1.0)
        if count and pct < 1:
            drawn = min(count, max(BOOTSTRAP_MIN_RESAMPLE_SIZE, int(round(count * pct))))
            floored = ' (floor applied)' if drawn > round(count * pct) else ''
            size = f'{pct:.0%} of {count:,} = {drawn:,} sales per resample{floored}'
        elif count:
            size = f'100% of {count:,} = {count:,} sales per resample'
        else:
            size = f'{pct:.0%} of the sample per resample'
        seed = resolved.get('bootstrap_seed')
        rows += [
            ('Method', resolved.get('bootstrap_method'),
             mark('method', resolved.get('bootstrap_method'), 'BCa')),
            ('Resamplings', f"{resolved.get('bootstrap_resamplings'):,}",
             mark('res', resolved.get('bootstrap_resamplings'), 1000)),
            ('Resample size', size, mark('pct', pct, 1.0)),
            ('Seed', str(seed) if seed is not None
             else 'not set - intervals vary slightly between runs',
             mark('seed', seed, None)),
        ]

    width = max(len(label) for label, _, _ in rows)
    lines = ['Confidence interval settings  (<-- marks a value you set or is a default setting)']
    lines += [f'  {label:<{width}} : {value}{flag}' for label, value, flag in rows]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------------------------------

def _bca_ci(series, population_estimate, confidence_level, interpolation_method='median_unbiased'):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Parameters
    ----------
    series : array-like
        Pandas, numpy, or narwhals series with the statistic computed
        for each bootstrap sample or observation.
    population_estimate : float
        Statistic estimate for the population, without resampling.
    confidence_level : float
        Confidence level for the confidence interval calculation.
    interpolation_method : str, default 'median_unbiased'
        Interpolation method passed to `numpy.percentile`.
        'median_unbiased' is argued to be the best choice, a
        rounding-equivalent method is 'closest_observation', and
        'linear' is an obvious alternative. See
        https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#r08bde0ebf37b-1
        and https://www.amherst.edu/media/view/129116/original/Sample+Quantiles.pdf
        for a detailed analysis of the available methods.

    Returns
    -------
    tuple
        Two-element tuple ``(lower_ci, upper_ci)`` with the confidence
        interval bounds. Returns ``(nan, nan)`` if the calculation
        fails.
    """
    try:
        # Coerce input (pandas Series, narwhals Series, list, ndarray, ...) to a 1D numpy array.
        series = nw_utils.to_numpy_1d(series)
        confidence_level_tail = (1 - confidence_level) / 2
        # handle cases where all results are the same, and would yield a bug in processing: acceleration is NaN, bias_correction is infinite
        unique_vals = np.unique(series)
        if len(unique_vals) == 1:
            val = float(unique_vals[0])
            return (val, val)
        else:
            try:
                acceleration = skew(series) / 6
                n = len(series)
                bias_correction = norm.ppf((series >= population_estimate).sum() / n)
                # bias_correction = norm.ppf(len([x for x in series if x >= population_estimate]) / n)
                if bias_correction > 20:
                    bias_correction = 20
                elif bias_correction < -20:
                    bias_correction = -20
                alpha1 = norm.cdf(bias_correction + ((bias_correction + (norm.ppf(confidence_level_tail))) / (1 - acceleration * (bias_correction + norm.ppf(confidence_level_tail)))))
                alpha2 = norm.cdf(bias_correction + ((bias_correction + (norm.ppf(1 - confidence_level_tail))) / (1 - acceleration * (bias_correction + norm.ppf(1 - confidence_level_tail)))))
                try:
                    lower_ci = round(np.percentile(a=series, q=alpha1 * 100, method=interpolation_method), 4)
                    upper_ci = round(np.percentile(a=series, q=alpha2 * 100, method=interpolation_method), 4)
                except: # for older numpy versions
                    lower_ci = round(np.percentile(a=series, q=alpha1 * 100, interpolation = 'linear'), 4)
                    upper_ci = round(np.percentile(a=series, q=alpha2 * 100, interpolation = 'linear'), 4)

                return ((lower_ci, upper_ci)) if lower_ci < upper_ci else ((upper_ci, lower_ci))
            except Exception as ex:
                logging.warning("BCa CI calculation failed: %s", ex)
                return (np.nan, np.nan)
    except Exception as ex:
        raise RuntimeError('error in ratio study: bca_ci:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _percentile_ci(series, confidence_level, interpolation_method='median_unbiased'):
    """
    Percentile confidence interval.

    Parameters
    ----------
    series : array-like
        Pandas, numpy, or narwhals series with the statistic indicator
        for each sample or observation.
    confidence_level : float
        Confidence level for the confidence interval calculation.
    interpolation_method : str, default 'median_unbiased'
        Interpolation method passed to `numpy.percentile`.
        'median_unbiased' is argued to be the best choice, a
        rounding-equivalent method is 'closest_observation', and
        'linear' is an obvious alternative. See
        https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#r08bde0ebf37b-1
        and https://www.amherst.edu/media/view/129116/original/Sample+Quantiles.pdf
        for a detailed analysis of the available methods.

    Returns
    -------
    tuple
        Two-element tuple ``(lower_ci, upper_ci)`` with the confidence
        interval bounds. Returns ``(nan, nan)`` if the calculation
        fails.
    """
    try:
        # Coerce input to a 1D numpy array for backend-agnostic handling.
        series = nw_utils.to_numpy_1d(series)
        confidence_level_tail = (1 - confidence_level) / 2
        # handle cases where all results are the same, and would yield a bug in processing
        unique_vals = np.unique(series)
        if len(unique_vals) == 1:
            val = float(unique_vals[0])
            return (val, val)
        else:
            try:
                try:
                    lower_ci = round(np.percentile(a=series, q=confidence_level_tail * 100, method=interpolation_method), 4)
                    upper_ci = round(np.percentile(a=series, q=(1 - confidence_level_tail) * 100, method=interpolation_method), 4)
                except: # for older numpy versions
                    lower_ci = round(np.percentile(a=series, q=confidence_level_tail * 100, interpolation='linear'), 4)
                    upper_ci = round(np.percentile(a=series, q=(1 - confidence_level_tail) * 100, interpolation='linear'), 4)

                return ((lower_ci, upper_ci)) if lower_ci < upper_ci else ((upper_ci, lower_ci))
            except:
                return ((np.nan, np.nan))
    except Exception as ex:
        raise RuntimeError('error in ratio study: percentile ci:', ex)
    

# ---------------------------------------------------------------------------------------------------------------------

def _format_ci(tpl, pattern, rounding=None):
    """
    Format the numbers inside a confidence interval tuple.

    Parameters
    ----------
    tpl : tuple
        Two-element tuple (or list/ndarray) with the confidence
        interval.
    pattern : str
        Format pattern applied to each bound (e.g. a ``str.format``
        pattern such as ``'{:.2f}'``).
    rounding : int, optional
        Number of decimal places to round each bound to before
        formatting. If None, the bounds are formatted without
        rounding.

    Returns
    -------
    tuple
        Two-element tuple with the formatted confidence interval
        bounds. Returns `tpl` unchanged if it is not a two-element
        tuple/list/ndarray, or if formatting fails.
    """
    try:
        if isinstance(tpl, (tuple, list, np.ndarray)) and len(tpl) == 2:
            try:
                if rounding is not None:
                    return (pattern.format(round(tpl[0], rounding)), pattern.format(round(tpl[1], rounding)))
                else:
                    return (pattern.format(tpl[0]), pattern.format(tpl[1]))
            except:
                return tpl
        else:
            return tpl
    except Exception as ex:
        raise RuntimeError('error in ratio study: format_ci', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _format_ci_string(tpl, pattern, rounding=None):
    """
    Format the numbers inside a confidence interval tuple and convert it to a string.

    Parameters
    ----------
    tpl : tuple
        Two-element tuple (or list/ndarray) with the confidence
        interval.
    pattern : str
        Format pattern applied to each bound (e.g. a ``str.format``
        pattern such as ``'{:.2f}'``).
    rounding : int, optional
        Number of decimal places to round each bound to before
        formatting. If None, the bounds are formatted without
        rounding.

    Returns
    -------
    str
        String of the form ``'(lower, upper)'`` with the formatted
        confidence interval bounds. Returns `tpl` unchanged if it is
        not a two-element tuple/list/ndarray, or if formatting fails.
    """
    try:
        if isinstance(tpl, (tuple, list, np.ndarray)) and len(tpl) == 2:
            try:
                if rounding is not None:
                    first_item = pattern.format(round(tpl[0], rounding))
                    second_item = pattern.format(round(tpl[1], rounding))
                else:
                    first_item = pattern.format(tpl[0])
                    second_item = pattern.format(tpl[1])
                return '(' + first_item + ', ' + second_item + ')'
            except:
                return tpl
        else:
            return tpl
    except Exception as ex:
        raise RuntimeError('error in ratio study: format_ci_string', ex)
    

# ---------------------------------------------------------------------------------------------------------------------

def log_median_diff_ci(
    low_ratios,
    high_ratios,
    overall_median,
    conf_level=0.90,
    threshold=10,
):
    """
    Confidence interval for the Value Equity Indicator (VEI).

    Computes VEI = (median_high - median_low) / overall_median,
    expressed as a percentage, together with a confidence interval
    derived from a log-normal delta-method approximation to the
    standard error of each group's median.

    Parameters
    ----------
    low_ratios : array-like
        Ratios in the low-value group.
    high_ratios : array-like
        Ratios in the high-value group.
    overall_median : float
        Median ratio of the full ratio-study sample.
    conf_level : float, default 0.90
        Confidence level.
    threshold : float, default 10
        Minimum-effect threshold, in percentage points, used to
        compute `flag_min_effect`.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * n_low : int -- number of valid observations in the low-value group.
        * n_high : int -- number of valid observations in the high-value group.
        * median_low : float -- median of `low_ratios`.
        * median_high : float -- median of `high_ratios`.
        * overall_median : float -- `overall_median`, echoed back.
        * vei_pct : float -- VEI estimate, as a percentage.
        * vei_ci_lower : float -- lower bound of the VEI confidence interval.
        * vei_ci_upper : float -- upper bound of the VEI confidence interval.
        * se_diff_raw : float -- standard error of the raw median difference.
        * conf_level : float -- `conf_level`, echoed back.
        * flag_min_effect : list of bool -- single-element list flagging
          whether the confidence interval clears `threshold` in either
          direction.
    """

    if overall_median is None or np.size(overall_median) != 1:
        raise ValueError(
            "overall_median must be a single numeric value: "
            "the median ratio of the full ratio-study sample."
        )

    low_ratios = np.asarray(low_ratios, dtype=float)
    high_ratios = np.asarray(high_ratios, dtype=float)

    n_low_raw = len(low_ratios)
    n_high_raw = len(high_ratios)

    # Keep only positive, non-missing values
    low_ratios = low_ratios[np.isfinite(low_ratios) & (low_ratios > 0)]
    high_ratios = high_ratios[np.isfinite(high_ratios) & (high_ratios > 0)]

    n_low = len(low_ratios)
    n_high = len(high_ratios)

    if n_low < n_low_raw or n_high < n_high_raw:
        warnings.warn(
            f"Dropped {n_low_raw - n_low} low-group and "
            f"{n_high_raw - n_high} high-group non-positive/missing values "
            "before logging."
        )

    if n_low < 2 or n_high < 2:
        raise ValueError(
            "Each group needs at least 2 valid observations "
            "to estimate a standard error."
        )

    med_low = np.median(low_ratios)
    med_high = np.median(high_ratios)

    # Step 1: SE of log-median
    are_const = np.sqrt(np.pi / 2)  # ~1.2533

    se_log_low = (
        are_const * np.std(np.log(low_ratios), ddof=1) / np.sqrt(n_low)
    )
    se_log_high = (
        are_const * np.std(np.log(high_ratios), ddof=1) / np.sqrt(n_high)
    )

    # Step 2: Delta-method transfer to raw scale
    se_raw_low = med_low * se_log_low
    se_raw_high = med_high * se_log_high

    # Step 3: CI for raw difference
    diff = med_high - med_low
    se_diff = np.sqrt(se_raw_low**2 + se_raw_high**2)

    conf_level = confidence_level_as_fraction(conf_level, 'conf_level')
    z = norm.ppf(1 - (1 - conf_level) / 2)

    diff_lo = diff - z * se_diff
    diff_hi = diff + z * se_diff

    # Step 4: Express as percent VEI
    vei_pct = 100 * diff / overall_median
    vei_ci_lower = 100 * diff_lo / overall_median
    vei_ci_upper = 100 * diff_hi / overall_median

    return {
            "n_low": n_low,
            "n_high": n_high,
            "median_low": med_low,
            "median_high": med_high,
            "overall_median": overall_median,
            "vei_pct": vei_pct,
            "vei_ci_lower": vei_ci_lower,
            "vei_ci_upper": vei_ci_upper,
            "se_diff_raw": se_diff,
            "conf_level": conf_level,
            "flag_min_effect": [
                (vei_ci_lower > threshold)
                or (vei_ci_upper < -threshold)
            ],
        }

# ---------------------------------------------------------------------------------------------------------------------

def _compare_ci(stat_ci
               , standard_ci):
    """
    Compare two confidence intervals.

    Both `stat_ci` and `standard_ci` must be sorted as (lower, upper).

    Parameters
    ----------
    stat_ci : tuple
        Two-element tuple with the confidence interval being tested.
    standard_ci : tuple
        Two-element tuple with the standard/reference confidence
        interval.

    Returns
    -------
    str
        One of 'confidence interval outside standards', 'confidence
        interval within standards', or 'confidence interval overlaps
        standards'.
    """
    try:
        if stat_ci[0] > standard_ci[1] or stat_ci[1] < standard_ci[0]:
            return 'confidence interval outside standards'
        elif stat_ci[0] >= standard_ci[0] and stat_ci[1] <= standard_ci[1]:
            return 'confidence interval within standards'
        else:
            return 'confidence interval overlaps standards'
    except Exception as ex:
        raise RuntimeError('error in ratio study: compare_ci', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _check_acceptance(stat, standard_ci):
    """
    Compare a statistic with a standard interval.

    `standard_ci` must be sorted as (lower, upper).

    Parameters
    ----------
    stat : float
        Statistic being tested.
    standard_ci : tuple
        Two-element tuple with the standard/reference interval.

    Returns
    -------
    str
        'statistic outside standards' or 'statistic within
        standards'. Returns 'could not check acceptance' if the
        comparison fails.
    """
    try:
        if stat > standard_ci[1] or stat < standard_ci[0]:
            return 'statistic outside standards'
        else:
            return 'statistic within standards'
    except:
        return 'could not check acceptance'
    
# ---------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _binom_median_rank_bounds(n, alpha):
    """
    Order-statistic indices and achieved coverage for the ``'binom'``
    median CI, at sample size `n` and significance `alpha`.

    Split out of `compute_median_ci` and cached because none of this
    depends on the sample's actual values -- only on `n` and `alpha` --
    so a caller that runs the same sample size repeatedly (`vem`'s moving
    window recomputes a median CI once per row, at the same window length
    for every interior row) would otherwise pay for the same
    ``scipy.stats.binom.ppf`` / ``.cdf`` calls, whose generic
    (non-vectorised) machinery dominates the per-call cost, over and over.

    Parameters
    ----------
    n : int
        Sample size.
    alpha : float
        Significance level, ``1 - conf_level``.

    Returns
    -------
    tuple
        ``(lower_idx, upper_idx, achieved_coverage)`` -- 0-indexed
        positions into the sorted sample, and the exact coverage of the
        interval those positions bound.
    """
    k_lo = int(binom.ppf(alpha / 2, n, 0.5))
    k_hi = int(binom.ppf(1 - alpha / 2, n, 0.5))
    lower_idx = max(k_lo - 1, 0)
    upper_idx = min(k_hi, n - 1)
    # Back to 1-indexed ranks, so the coverage describes this interval
    # and not the one rank narrower it used to be computed from.
    rank_lo = lower_idx + 1
    rank_hi = upper_idx + 1
    achieved_coverage = float(
        binom.cdf(rank_hi - 1, n, 0.5) - binom.cdf(rank_lo - 1, n, 0.5)
    )
    return lower_idx, upper_idx, achieved_coverage


def compute_median_ci(asr, conf_level=0.9, method='binom'):
    """
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
    """
    asr = np.asarray(asr)
    n = len(asr)
    if n < 1:
        raise ValueError("No data provided")
    if n == 1:
        v = float(asr[0])
        return {"median": v, "lower": v, "upper": v, "achieved_coverage": conf_level}

    sorted_asr = np.sort(asr)
    # The array is already sorted, so the middle order statistic(s) give
    # the median directly -- `np.median` would re-sort/partition it as if
    # it weren't. Cheap on its own, but this runs once per row in `vem`'s
    # moving window, so the redundant pass adds up.
    mid = n // 2
    median = float(sorted_asr[mid] if n % 2 else
                   (sorted_asr[mid - 1] + sorted_asr[mid]) / 2.0)
    method_up = str(method).upper()
    # Through the shared gate, so 0.95 and 95 both mean 95% here as well,
    # and a significance level passed by mistake is rejected rather than
    # turned into a negative alpha and a NaN.
    conf_level = confidence_level_as_fraction(conf_level, 'conf_level')
    alpha = 1 - conf_level
    q_lo = (alpha / 2.0) * 100.0
    q_hi = (1.0 - alpha / 2.0) * 100.0

    if method_up in ('R6', 'R7'):
        # A confidence interval for the *median* is a pair of order
        # statistics either side of the middle of the sample, at rank
        # distance ``z * sqrt(n) / 2`` from it (Appendix D.2; Gloudemans &
        # Almy 2011, p. 365). Expressed as percentile ranks of the data
        # that is
        #
        #     p = 0.5 -/+ z / (2 * sqrt(n))
        #
        # and R6/R7 are then just the rule for turning ``p`` into a
        # position in the sorted sample. Feeding ``alpha/2`` in as ``p``
        # instead -- the 5th and 95th percentiles of the ratios for a 90%
        # interval -- returns the spread of the data rather than the
        # uncertainty of its median: it does not narrow as ``n`` grows,
        # and the coverage it reports goes to 1 regardless of
        # `conf_level`.
        z = norm.ppf(1.0 - alpha / 2.0)
        spread = z / (2.0 * math.sqrt(n))
        p_lo = min(max(0.5 - spread, 0.0), 1.0)
        p_hi = min(max(0.5 + spread, 0.0), 1.0)
        if method_up == 'R6':
            h_lo = p_lo * (n + 1)
            h_hi = p_hi * (n + 1)
        else:  # R7
            h_lo = p_lo * (n - 1) + 1
            h_hi = p_hi * (n - 1) + 1
        # R6 and R7 differ only in this position mapping; both then read
        # the value at a fractional position by interpolating between the
        # two order statistics that bracket it. `median_ci_rank` is the
        # conservative integer-rank variant D.2 spells out, which rounds
        # the positions outward instead of interpolating.
        lower = _order_statistic_at(sorted_asr, h_lo)
        upper = _order_statistic_at(sorted_asr, h_hi)
        if lower > upper:
            lower, upper = upper, lower
        # Coverage of ``[x_(j), x_(k)]`` is ``P(j <= B <= k-1)`` for
        # ``B ~ Bin(n, 0.5)``, the count of observations below the true
        # median. Approximating B by ``N(n/2, sqrt(n)/2)`` with the usual
        # half-unit continuity correction and re-centring on
        # ``(n + 1) / 2`` cancels both shifts, so the bounds go in
        # unadjusted. Checked against the exact binomial: 0.886 vs 0.892
        # at n=25, 0.898 vs 0.896 at n=200, both at a nominal 90%.
        mu = (n + 1) / 2.0
        sigma = max(math.sqrt(n) / 2.0, 1e-12)
        achieved_coverage = float(
            norm.cdf(h_hi, loc=mu, scale=sigma)
            - norm.cdf(h_lo, loc=mu, scale=sigma)
        )
        return {
            "median": median,
            "lower": lower,
            "upper": upper,
            "achieved_coverage": achieved_coverage,
        }

    if method_up == 'BINOM':
        # Sign-test / order-statistic CI for the median. The bounds are the
        # 1-indexed order statistics ``j`` and ``k``, and the exact coverage
        # of ``[x_(j), x_(k)]`` is ``P(j <= B <= k - 1)`` for ``B ~
        # Bin(n, 0.5)``, so the coverage has to be derived from the ranks
        # actually returned rather than from ``k_lo`` / ``k_hi`` directly.
        # None of that depends on the sample's values, only on `n` and
        # `alpha`, so it is cached: a caller running the same window length
        # over and over (`vem`'s moving median) would otherwise repeat the
        # same ``binom.ppf`` / ``.cdf`` calls, whose fixed per-call cost
        # dominates a small window's own sort.
        lower_idx, upper_idx, achieved_coverage = _binom_median_rank_bounds(n, alpha)
        lower = float(sorted_asr[lower_idx])
        upper = float(sorted_asr[upper_idx])
        return {
            "median": median,
            "lower": lower,
            "upper": upper,
            "achieved_coverage": achieved_coverage,
        }

    raise ValueError(f"Unknown compute_median_ci method: {method!r}; expected 'R6', 'R7', or 'binom'.")


# ---------------------------------------------------------------------------------------------------------------------
#  Distribution-based confidence intervals
#
#  IAAO Exposure Draft (May 2026) Appendix D. Each indicator gets the
#  formula prescribed for it rather than one generic interval:
#
#    D.1  mean            -- Student's t about the mean ratio
#    D.2  median (n > 30) -- large-sample rank formula
#    D.3  weighted mean   -- ratio-estimator standard error
#    D.4  median (n <= 30)-- exact binomial tables (Clapp 1989)
#    D.6  CoD            -- Bonett-Seier log-scale interval
#
#  All of them return a plain ``(lower, upper)`` tuple, the same shape
#  ``_bca_ci`` / ``_percentile_ci`` return, so the results table cannot tell
#  which family filled a ``<indicator>_ci`` column. ``(nan, nan)`` comes back
#  whenever the sample is too small or too degenerate for the formula, which
#  leaves the column blank exactly as a skipped bootstrap does.
# ---------------------------------------------------------------------------------------------------------------------

def _order_statistic_at(sorted_values, position):
    """
    Read the value at a fractional 1-indexed position in a sorted sample.

    Linear interpolation between the two order statistics that bracket
    `position`, clamped to the ends of the sample. This is the second half
    of a Hyndman-Fan quantile estimator: the ``h(p)`` mapping (which is
    what distinguishes R6 from R7) picks the position, and this turns the
    position into a value.

    Parameters
    ----------
    sorted_values : numpy.ndarray
        Sample values in ascending order.
    position : float
        1-indexed position, so ``1`` is the smallest value and ``len`` the
        largest. Positions outside that range are clamped rather than
        extrapolated -- an order-statistic bound can never leave the
        observed range, which is the property that keeps a ratio interval
        from reaching below zero.

    Returns
    -------
    float
        The interpolated value.
    """
    n = len(sorted_values)
    if n == 0:
        return float('nan')
    position = min(max(float(position), 1.0), float(n))
    lower_index = int(math.floor(position))
    if lower_index >= n:
        return float(sorted_values[n - 1])
    fraction = position - lower_index
    low = float(sorted_values[lower_index - 1])
    high = float(sorted_values[lower_index])
    return low + fraction * (high - low)


# ---------------------------------------------------------------------------------------------------------------------

def _critical_value(conf_level, n=None, use_t=True):
    """
    Two-tailed critical value for a confidence level.

    Parameters
    ----------
    conf_level : float
        Confidence level as a fraction, e.g. 0.95.
    n : int, optional
        Sample size. Required when `use_t` is True, which uses ``n - 1``
        degrees of freedom.
    use_t : bool, default True
        If True, return Student's t; if False, the normal deviate.
        Exposure Draft D.5.1 tabulates t and notes that 1.96 (95 %) and
        1.645 (90 %) may be used once the sample passes about 30. Using
        the exact t throughout reproduces those values to three decimals
        in the large-sample limit, so this never disagrees with the table
        by more than its own rounding.

    Returns
    -------
    float
        The critical value, or NaN when the degrees of freedom are
        non-positive.
    """
    alpha = 1.0 - confidence_level_as_fraction(conf_level, 'conf_level')
    if not use_t:
        return float(norm.ppf(1.0 - alpha / 2.0))
    if n is None:
        raise ValueError('n is required to pick a t critical value')
    df = int(n) - 1
    if df < 1:
        return float('nan')
    from scipy.stats import t as _t
    return float(_t.ppf(1.0 - alpha / 2.0, df))


# ---------------------------------------------------------------------------------------------------------------------

def mean_ci(asr, conf_level=0.95, weights=None):
    """
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
    """
    try:
        values = nw_utils.to_numpy_1d(asr).astype(float)
        if weights is not None:
            w = nw_utils.to_numpy_1d(weights).astype(float)
            keep = np.isfinite(values) & np.isfinite(w) & (w > 0)
            values, w = values[keep], w[keep]
        else:
            values = values[np.isfinite(values)]
            w = None

        n = len(values)
        if n < 2:
            return (np.nan, np.nan)

        if w is None:
            centre = float(np.mean(values))
            variance = float(np.var(values, ddof=1))
        else:
            # Reliability weights: the package normalises them to mean 1,
            # so ``sum(w)`` is the sample size and the standard error is
            # the weighted variance over n, as in the unweighted case.
            w = w / np.mean(w)
            centre = float(np.average(values, weights=w))
            variance = float(np.sum(w * (values - centre) ** 2) / (np.sum(w) - 1))

        std_error = math.sqrt(variance / n)
        critical = _critical_value(conf_level, n=n, use_t=True)
        if not np.isfinite(critical) or not np.isfinite(std_error):
            return (np.nan, np.nan)
        margin = critical * std_error
        # An ASR cannot be negative -- it is a ratio of two positive
        # amounts -- but this interval is symmetric in value, unlike the
        # order-statistic median intervals whose bounds are drawn from the
        # sample. On a small, sharply right-skewed sample the margin can
        # exceed the mean, so the lower bound is clamped at zero rather
        # than published as an impossible negative ratio. The upper bound
        # needs no clamp: the ASR has no ceiling.
        return (max(centre - margin, 0.0), centre + margin)
    except Exception as ex:
        raise RuntimeError('error in ratio study: mean_ci:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def median_ci_rank(asr, conf_level=0.95):
    """
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
    """
    try:
        values = nw_utils.to_numpy_1d(asr).astype(float)
        values = values[np.isfinite(values)]
        n = len(values)
        if n == 0:
            return {'median': np.nan, 'lower': np.nan, 'upper': np.nan,
                    'lower_rank': None, 'upper_rank': None, 'method': 'rank'}

        sorted_asr = np.sort(values)
        median = float(np.median(sorted_asr))
        z = _critical_value(conf_level, use_t=False)
        half_width = z * math.sqrt(n) / 2.0

        if n % 2:
            # Odd n: the median is the single rank (n + 1) / 2, and j is
            # counted out from it in both directions.
            j = int(math.ceil(half_width))
            middle = (n + 1) // 2
            lower_rank, upper_rank = middle - j, middle + j
        else:
            # Even n: the median falls between ranks n/2 and n/2 + 1. The
            # +0.5 makes j land on an integer number of ranks measured from
            # the two central ones outward.
            j = int(math.ceil(half_width + 0.5))
            lower_rank, upper_rank = n // 2 - j + 1, n // 2 + j

        lower_rank = max(1, min(n, lower_rank))
        upper_rank = max(1, min(n, upper_rank))
        return {
            'median': median,
            'lower': float(sorted_asr[lower_rank - 1]),
            'upper': float(sorted_asr[upper_rank - 1]),
            'lower_rank': lower_rank,
            'upper_rank': upper_rank,
            'method': 'rank',
        }
    except Exception as ex:
        raise RuntimeError('error in ratio study: median_ci_rank:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def median_ci_clapp(asr, conf_level=0.95):
    """
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
    """
    try:
        level = confidence_level_as_fraction(conf_level, 'conf_level')
        level_pct = int(round(level * 100))
        table = _MEDIAN_CI_CLAPP_LOWER.get(level_pct)
        if table is None:
            return None

        values = nw_utils.to_numpy_1d(asr).astype(float)
        values = values[np.isfinite(values)]
        n = len(values)
        params = table.get(n)
        if params is None:
            return None

        weight, rank_a, rank_b = params
        sorted_asr = np.sort(values)

        def _blend(first_rank, second_rank):
            return float(
                weight * sorted_asr[first_rank - 1]
                + (1.0 - weight) * sorted_asr[second_rank - 1]
            )

        lower = _blend(rank_a, rank_b)
        # The published upper row is the lower row mirrored about the
        # middle of the array, so the ranks are counted in from the top.
        upper = _blend(n + 1 - rank_a, n + 1 - rank_b)
        if lower > upper:
            lower, upper = upper, lower

        return {
            'median': float(np.median(sorted_asr)),
            'lower': lower,
            'upper': upper,
            'method': 'binomial_table',
            'caution': n in _MEDIAN_CI_CLAPP_EXTREME_N.get(level_pct, ()),
        }
    except Exception as ex:
        raise RuntimeError('error in ratio study: median_ci_clapp:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def median_ci_distribution(asr, conf_level=0.95, messages=None):
    """
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
    """
    try:
        values = nw_utils.to_numpy_1d(asr).astype(float)
        values = values[np.isfinite(values)]
        n = len(values)
        if n < 2:
            return (np.nan, np.nan)

        level = confidence_level_as_fraction(conf_level, 'conf_level')

        if n > MEDIAN_CI_SMALL_SAMPLE_MAX:
            out = median_ci_rank(values, level)
            return (out['lower'], out['upper'])

        out = median_ci_clapp(values, level)
        if out is not None:
            if out['caution'] and messages is not None:
                messages.append(
                    f'note: with {n} sales the {level:.0%} median confidence '
                    'limits rest on the most extreme ratios in the sample, so '
                    'they say little about the median (IAAO 1990, 609).'
                )
            return (out['lower'], out['upper'])

        # Either a level the tables do not cover, or n below the smallest
        # tabulated size. The exact binomial interval is defined for both.
        if messages is not None:
            level_list = ', '.join(f'{p}%' for p in _MEDIAN_CI_CLAPP_LEVELS)
            messages.append(
                'note: the small-sample median confidence interval table covers '
                f'{level_list} only, and n from 5; with n={n} at {level:.0%} '
                'the exact binomial interval was used instead.'
            )
        exact = compute_median_ci(values, level, method='binom')
        return (exact['lower'], exact['upper'])
    except Exception as ex:
        raise RuntimeError('error in ratio study: median_ci_distribution:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def weighted_mean_ci(assessment, market_value, conf_level=0.95, weights=None):
    """
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
    """
    try:
        a = nw_utils.to_numpy_1d(assessment).astype(float)
        s = nw_utils.to_numpy_1d(market_value).astype(float)
        keep = np.isfinite(a) & np.isfinite(s)
        if weights is not None:
            w = nw_utils.to_numpy_1d(weights).astype(float)
            keep &= np.isfinite(w) & (w > 0)
            a, s, w = a[keep], s[keep], w[keep]
            w = w / np.mean(w)
        else:
            a, s = a[keep], s[keep]
            w = np.ones(len(a))

        n = len(a)
        if n < 2:
            return (np.nan, np.nan)

        sum_s = float(np.sum(w * s))
        if sum_s == 0:
            return (np.nan, np.nan)

        weighted_mean = float(np.sum(w * a) / sum_s)
        mean_s = sum_s / float(np.sum(w))

        sum_a2 = float(np.sum(w * a * a))
        sum_s2 = float(np.sum(w * s * s))
        sum_as = float(np.sum(w * a * s))

        # sum((A - R*S)^2), expanded so it reads as the draft's steps.
        residual_ss = sum_a2 - 2.0 * weighted_mean * sum_as \
            + weighted_mean ** 2 * sum_s2
        if residual_ss < 0:
            # Only reachable through floating-point cancellation on a
            # perfectly fitting sample.
            residual_ss = 0.0

        critical = _critical_value(conf_level, n=n, use_t=True)
        if not np.isfinite(critical) or mean_s == 0:
            return (np.nan, np.nan)

        margin = critical * math.sqrt(residual_ss) / (
            mean_s * math.sqrt(n * (n - 1))
        )
        # Clamped at zero for the same reason as `mean_ci`: a ratio of two
        # positive totals cannot be negative, and this interval is
        # symmetric in value so a wide margin on a small sample could push
        # the lower bound below the floor.
        return (max(weighted_mean - margin, 0.0), weighted_mean + margin)
    except Exception as ex:
        raise RuntimeError('error in ratio study: weighted_mean_ci:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def prd_ci(assessment, market_value, conf_level=0.95, weights=None):
    """
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
    """
    try:
        a = nw_utils.to_numpy_1d(assessment).astype(float)
        s = nw_utils.to_numpy_1d(market_value).astype(float)
        keep = np.isfinite(a) & np.isfinite(s) & (s != 0)
        if weights is not None:
            w = nw_utils.to_numpy_1d(weights).astype(float)
            keep &= np.isfinite(w) & (w > 0)
            a, s, w = a[keep], s[keep], w[keep]
            w = w / np.mean(w)
        else:
            a, s = a[keep], s[keep]
            w = np.ones(len(a))

        n = len(a)
        if n < 2:
            return (np.nan, np.nan)

        # Same subset feeds both `asr` (for M) and `a`/`s` (for W), so the
        # cross term below pairs each sale with itself rather than with a
        # differently-filtered sample.
        asr = a / s

        sum_w = float(np.sum(w))
        sum_s = float(np.sum(w * s))
        if sum_s == 0:
            return (np.nan, np.nan)
        mean_s = sum_s / sum_w

        mean_ratio = float(np.sum(w * asr) / sum_w)              # M
        weighted_mean = float(np.sum(w * a) / sum_s)              # W
        if weighted_mean == 0:
            return (np.nan, np.nan)

        var_m = float(np.sum(w * (asr - mean_ratio) ** 2) / (sum_w - 1) / n)

        residual = a - weighted_mean * s
        residual_ss = float(np.sum(w * residual ** 2))
        if residual_ss < 0:
            residual_ss = 0.0
        var_w = residual_ss / (mean_s ** 2 * n * (n - 1))

        cov_mw = float(np.sum(w * (asr - mean_ratio) * residual)) / (
            mean_s * n * (n - 1)
        )

        prd_value = mean_ratio / weighted_mean
        variance = (var_m / weighted_mean ** 2
                    + mean_ratio ** 2 * var_w / weighted_mean ** 4
                    - 2.0 * mean_ratio * cov_mw / weighted_mean ** 3)
        if variance < 0:
            # Only reachable through floating-point cancellation, same as
            # `residual_ss` above.
            variance = 0.0

        critical = _critical_value(conf_level, n=n, use_t=True)
        std_error = math.sqrt(variance)
        if not np.isfinite(critical) or not np.isfinite(std_error):
            return (np.nan, np.nan)
        margin = critical * std_error
        # Clamped at zero for the same reason as `mean_ci` and
        # `weighted_mean_ci`: PRD is a ratio of two positive quantities.
        return (max(prd_value - margin, 0.0), prd_value + margin)
    except Exception as ex:
        raise RuntimeError('error in ratio study: prd_ci:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def cod_ci(asr, conf_level=0.95):
    """
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
    """
    try:
        values = nw_utils.to_numpy_1d(asr).astype(float)
        values = values[np.isfinite(values)]
        n = len(values)
        if n < 4:
            return (np.nan, np.nan)

        sorted_asr = np.sort(values)
        median = float(np.median(sorted_asr))
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=1))

        if median <= 0:
            return (np.nan, np.nan)

        # tau: the mean absolute deviation from the median.
        tau = float(np.mean(np.abs(values - median)))
        if tau <= 0:
            return (np.nan, np.nan)

        delta = (mean - median) / tau              # mean-median standardised difference
        gamma = variance / (tau ** 2)              # relative dispersion ratio

        # Variance ranks, one root-n either side of the middle.
        astar = int(round((n + 1) / 2.0 - math.sqrt(n)))
        astar = max(1, min(n, astar))
        bstar = n - astar + 1
        ar = float(sorted_asr[astar - 1])
        br = float(sorted_asr[bstar - 1])
        if ar <= 0 or br <= 0:
            return (np.nan, np.nan)

        varleta = ((math.log(ar) - math.log(br)) / 4.0) ** 2
        se1 = math.sqrt(varleta)
        varltau = ((gamma + delta ** 2) - 1.0) / n
        if varltau <= 0 or (se1 + math.sqrt(max(varltau, 0.0))) == 0:
            return (np.nan, np.nan)
        se2 = math.sqrt(varltau)
        covltle = (delta * math.sqrt(varleta)) / math.sqrt(n)

        calibration_var = varleta + varltau - 2.0 * covltle
        if calibration_var < 0:
            return (np.nan, np.nan)
        k = math.sqrt(calibration_var) / (se1 + se2)

        # D.6 quotes 1.96 / 1.645 and says to use a t value below n = 30.
        critical = _critical_value(conf_level, n=n, use_t=n < 30)
        if not np.isfinite(critical):
            return (np.nan, np.nan)

        rank_low = int(round((n + 1) / 2.0 - k * critical * math.sqrt(n / 4.0)))
        rank_low = max(1, min(n, rank_low))
        rank_high = max(1, min(n, n - rank_low + 1))
        ratio_low = float(sorted_asr[rank_low - 1])
        ratio_high = float(sorted_asr[rank_high - 1])
        if ratio_low <= 0 or ratio_high <= 0:
            return (np.nan, np.nan)

        scaled_tau = (n / (n - 1.0)) * tau
        if scaled_tau <= 0:
            return (np.nan, np.nan)

        margin = k * critical * se2
        log_centre = math.log(scaled_tau)
        lower = math.exp((log_centre - margin) - math.log(ratio_high))
        upper = math.exp((log_centre + margin) - math.log(ratio_low))
        if lower > upper:
            lower, upper = upper, lower
        return (lower, upper)
    except Exception as ex:
        raise RuntimeError('error in ratio study: cod_ci:', ex) from ex


# ---------------------------------------------------------------------------------------------------------------------

def distribution_analysis(input_df
                        , assessment_header
                        , market_value_header
                        , weights_header=None
                        , confint_level=0.95
                        , indicators=None
                        , messages=None):
    """
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
    """
    try:
        level = confidence_level_as_fraction(confint_level, 'confint_level')
        wanted = (list(DISTRIBUTION_CI_SUPPORTED_INDICATORS) if indicators is None
                  else [x for x in indicators if x in DISTRIBUTION_CI_SUPPORTED_INDICATORS])
        if not wanted:
            return {}

        assessment = nw_utils.to_numpy_1d(input_df[assessment_header]).astype(float)
        market_value = nw_utils.to_numpy_1d(input_df[market_value_header]).astype(float)
        weights = (nw_utils.to_numpy_1d(input_df[weights_header]).astype(float)
                   if weights_header else None)
        with np.errstate(divide='ignore', invalid='ignore'):
            asr = assessment / market_value

        result = {}
        if 'Mean' in wanted:
            result['Mean_ci'] = mean_ci(asr, level, weights=weights)
        if 'Median' in wanted:
            result['Median_ci'] = median_ci_distribution(asr, level, messages=messages)
        if 'CoD' in wanted:
            result['CoD_ci'] = cod_ci(asr, level)
        if 'Weighted_Mean' in wanted:
            result['Weighted_Mean_ci'] = weighted_mean_ci(
                assessment, market_value, level, weights=weights
            )
        if 'PRD' in wanted:
            result['PRD_ci'] = prd_ci(assessment, market_value, level, weights=weights)
        # PRB has no branch here: its interval comes from the slope
        # regression `prb()` always builds (see `rts_calculator` in
        # `ratio_study.py`), not from a per-sample-array formula like the
        # others in this function. Being in `DISTRIBUTION_CI_SUPPORTED_
        # INDICATORS` keeps it from tripping the "no indicator supports
        # one" fallback and from being reported as unsupported; the
        # regression CI reaches the results table on its own.
        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: distribution_analysis:', ex) from ex


