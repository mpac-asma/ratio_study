from datetime import datetime
import logging
import warnings
import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
import tarfile
import zipfile
from ratio_study.indicators import *
from ratio_study.helpers import _add_needed_variables, iqr_filter, report_iqr_trim, save_errors_to_text, _split_list, custom_format, _add_suffixes_to_list, leaf_values
from ratio_study.confint import (normalize_ci_args, distribution_analysis,
                                 confidence_level_as_fraction,
                                 describe_ci_settings,
                                 BOOTSTRAP_MIN_RESAMPLE_SIZE)
from . import nw_utils

gen_log = False

# Reading order for the results table, independent of the order `indicators`
# was passed in. Grouped the way the Standard on Ratio Studies presents them:
# level of valuation (s.7), then uniformity within the stratum (s.8.1), then
# vertical equity (s.8.2). Each indicator's `_ci` / `_Pvalue` companion is
# inserted immediately after it by `_add_suffixes_to_list`.
INDICATOR_DISPLAY_ORDER = [
    # level of valuation
    'Median', 'Mean', 'Weighted_Mean',
    # uniformity within the stratum (horizontal equity)
    'CoD', 'CoD_rsd', 'CoDw', 'CoV', 'CoC_10', 'CoC_15', 'CoC_20', 'FSD',
    # vertical equity -- PRB before PRD: PRB is the supported measure, PRD is
    # retired after the September 2026 Standard
    'PRB', 'PRD',
]

# Accuracy measures trail the acceptance indicators: they describe how far
# individual valuations land from the market, not whether the study passes.
ACCURACY_DISPLAY_ORDER = ['RMSE', 'MAE', 'MAPE', 'MdAPE']

# The indicator set a study computes when the caller does not choose one.
# Shared, because `ratios_report` reads `indicators` before handing it to
# `ratios` -- with the list written out twice they drifted, and passing
# ``indicators=None`` to the report raised on ``x in None``.
#
# All three measures of valuation level are here, not just the median: the
# Standard asks a published study for measures of central tendency and their
# confidence intervals (s.10.1.2), and its own demonstration report reports
# mean, median and weighted mean together (Table 10.2). Leaving `Mean` and
# `Weighted_Mean` out also cost their intervals -- the CI list is intersected
# with the study -- so a default distribution-mode study silently had no
# `Mean_ci` or `Weighted_Mean_ci` despite Appendix D giving a formula for both.
DEFAULT_INDICATORS = ['Median', 'Mean', 'Weighted_Mean', 'CoD', 'PRD', 'PRB',
                      'VEI', 'VEM']


def _report_ci_messages(messages, error_messages):
    """
    Surface the input problems `normalize_ci_args` found.

    Each message is warned about (so it shows up in an interactive
    session), added to the study's `error_messages` list (so it reaches the
    errors file and the PDF's ending comments), and logged when logging is
    on.

    Parameters
    ----------
    messages : list of str
        Messages returned by
        :func:`ratio_study.confint.normalize_ci_args`.
    error_messages : list of str
        The study's error-message accumulator, appended to in place.
    """
    for message in messages:
        error_messages.append(message)
        warnings.warn(message, stacklevel=3)
        if gen_log:
            logging.warning(message)


def _drop_unbootstrapped_prb_ci(ratios_dict, bootstrap_args):
    """
    Remove a `PRB_ci` the bootstrap did not produce.

    `rts_calculator` fills `PRB_ci` from the PRB slope's own regression, at
    `req_significance` rather than at the bootstrap's confidence level. When
    PRB is one of the bootstrapped indicators, keeping that value for the
    rows the bootstrap skipped -- CIs off for this scope, or a sample below
    ``min_count_threshold`` -- would put an interval from a different
    estimator, at a different level, in the same column as the rest.

    Parameters
    ----------
    ratios_dict : dict
        Indicator results for one row, modified in place.
    bootstrap_args : dict
        Resolved bootstrap options, as returned by
        :func:`ratio_study.confint.normalize_ci_args`.
    """
    if 'PRB' in bootstrap_args['bootstrap_indicators']:
        ratios_dict.pop('PRB_ci', None)


def _confidence_intervals_for_scope(scope_df
                                  , assessment_header
                                  , market_value_header
                                  , weights_header
                                  , bootstrap_args
                                  , ratios_dict
                                  , count
                                  , req_significance
                                  , vei_conf_level
                                  , vei_use_log_normal_delta
                                  , messages):
    """
    Build the ``<indicator>_ci`` entries for one scope, in whichever mode
    the caller asked for.

    One place decides bootstrap-versus-distribution, so the whole sample
    and each group cannot drift apart. Both modes return the same
    ``{'<indicator>_ci': (lower, upper)}`` shape, so the results table is
    assembled identically either way.

    The minimum-count gate applies to the bootstrap only: fewer than
    ``min_count_threshold`` sales cannot support a resampled interval, and
    the columns are left unset so they come out blank. The distribution
    formulas need no such gate -- each one is defined down to its own
    sample-size floor (the binomial median tables reach down to 5 or 6
    sales) and returns ``(nan, nan)`` below it.

    Parameters
    ----------
    scope_df : pandas DataFrame
        Sales for this scope -- the whole study, or one group.
    assessment_header, market_value_header : str
        Assessed-value and sale-amount columns.
    weights_header : str or None
        Sample-weight column.
    bootstrap_args : dict
        Resolved CI options, as returned by
        :func:`ratio_study.confint.normalize_ci_args`.
    ratios_dict : dict
        Indicator results already computed for this scope. The BCa method
        bias-corrects against them.
    count : int
        Number of sales in this scope.
    req_significance, vei_conf_level, vei_use_log_normal_delta
        Passed through to the per-resample recalculation.
    messages : list of str
        Accumulator for notes a formula raises about the sample.

    Returns
    -------
    dict or None
        The ``<indicator>_ci`` entries, or None when no interval was built
        for this scope (so the caller can clear a stale `PRB_ci`).
    """
    if bootstrap_args['ci_mode'] == 'distribution':
        return distribution_analysis(scope_df
                                     , assessment_header
                                     , market_value_header
                                     , weights_header=weights_header
                                     , confint_level=bootstrap_args['bootstrap_confint_level']
                                     , indicators=bootstrap_args['bootstrap_indicators']
                                     , messages=messages)

    if count < bootstrap_args['min_count_threshold']:
        return None

    return _bootstrap_analysis(scope_df
                               , assessment_header
                               , market_value_header
                               , weights_header
                               , confint_level=bootstrap_args['bootstrap_confint_level']
                               , bootstrap_method=bootstrap_args['bootstrap_method']
                               , bootstrap_resamplings=bootstrap_args['bootstrap_resamplings']
                               , resample_pct=bootstrap_args['bootstrap_resample_pct']
                               , population_ratios=ratios_dict
                               , ci_only=bootstrap_args['ci_only']
                               , req_significance=req_significance
                               , vei_conf_level=vei_conf_level
                               , vei_use_log_normal_delta=vei_use_log_normal_delta
                               , indicators=bootstrap_args['bootstrap_indicators']
                               , concurrency=bootstrap_args['bootstrap_concurrency']
                               , seed=bootstrap_args['bootstrap_seed'])


def ratios(input_df
          , assessment_header
          , market_value_header
          , mkt_value_proxy_header = None
          , prb_mvp_header = None
          , mvp_sale_price_share = DEFAULT_MVP_SALE_PRICE_SHARE
          , group_by=None
          , log_transform=False
          , weights_header=None
          , export_directory=Path.cwd()
          , descriptive=False
          , dict_results=None
          , model_name='model'
          , indicators=None
          , gini_args = None
          , bo_args = None
          , confidence_intervals_aggs=None
          , ci_mode='distribution'
          , ci_conf_level=None
          , ci_indicators=None
          , ci_seed=None
          , bootstrap_args = None
          , print_ci_settings=True
          , vei_percentile_method='R7'
          , vei_compare_all_buckets=False
          , vei_conf_level=0.9
          , vei_plot=False
          , vei_plot_ref=''
          , vei_use_log_normal_delta=True
          , ve_plot_ref = ''
          , vem_acceptable_inequity_level = 0.1
          , vem_conf_level = 0.9
          , vem_ci_method = 'binom'
          , vem_plot = False
          , vem_title = "Vertical Equity Moving Median"
          , round_values=True
          , req_significance=0.05
          , assessment_standards=None
          , graphs_args=None
          , discrete_variables=None
          , iqr_trim_multiple=0
          , iqr_log_transform=True
          , iqr_group_by=None
          , iqr_min_cod=0.06
          , iqr_filter_result='print'
          , return_dict=False
          , VE_bundle_comments=False
          ):

    """
    Compute a ratio study (assessed-to-market-value ratios) with optional
    grouping, indicators, confidence intervals, Gini decomposition, and charts.

    Parameters
    ----------
    input_df : DataFrame (pandas, polars, or other narwhals-supported backend)
        Sales data containing assessed values, market values, and any
        additional columns needed for grouping or indicators.
    assessment_header : str
        Column name in `input_df` for the assessed / predicted value.
    market_value_header : str
        Column name in `input_df` for the market value / sale amount.
    mkt_value_proxy_header : str, optional
        Column to use as a proxy for market value (e.g. in VEI calculations)
        when `market_value_header` isn't the appropriate reference.
    prb_mvp_header : str, optional
        Column to use as the market-value proxy in the PRB regression,
        overriding `prb`'s default blend of `market_value_header`
        and the ASR-adjusted assessment. Independent of
        `mkt_value_proxy_header`, which is for VEI.
    mvp_sale_price_share : float, default 0.5
        Share of the sale price in the market value proxy built for the
        PRB regression and the VEI grouping -- see
        :func:`ratio_study.market_value_proxy`. The default reproduces the
        Standard's 50/50 proxy; ``0.3`` gives a 30/70 one. Ignored for
        whichever of the two proxies is supplied as a column via
        `prb_mvp_header` / `mkt_value_proxy_header`.
    group_by : str or list, default None
        Column(s) to group the ratio study by.

        * Simple grouping : a list of column names, e.g.
          ``['propcode', 'neighbourhood']`` computes ratios separately for
          each one.
        * Nested grouping : a list of lists, e.g.
          ``[['propcode', 'neighbourhood']]`` computes ratios for every
          combination of the given columns.

        Groups with fewer than 4 sales skip some indicators; groups with
        fewer than ``bootstrap_args['min_count_threshold']`` sales (20 by
        default) also skip bootstrap confidence intervals.
        Defaults to ratios computed for the whole sample only.
    log_transform : bool, default False
        Set to True if the assessment/market value columns are log-transformed,
        so they're converted back to price scale.
    weights_header : str, optional
        Column with sample weights. Rescaled internally to have mean 1.
    export_directory : str or Path, default current working directory
        Directory where output files are saved.
    descriptive : bool, default False
        If True, add automatic acceptance comments against `assessment_standards`.
    dict_results : dict, optional
        Existing results dictionary to append to. A new one is created if None.
    model_name : str, default 'model'
        Name used for output file names and chart titles.
    indicators : list of str, optional
        Indicators to compute. Defaults to
        ``['Median', 'Mean', 'Weighted_Mean', 'CoD', 'PRD', 'PRB', 'VEI',
        'VEM']``.
    gini_args : dict, optional
        Options for the Gini decomposition:

        * run_regression : bool, default True -- also compute Reranking
          Effect, Reranking Index, Kakwani Index, and associated p-values.
        * gini_decomposition : bool, default False -- run the Gini decomposition.
        * gini_factors_variables : list of str, default [] -- independent
          model variables to decompose the Gini coefficient by.
        * gini_lorenz_plots : bool, default False -- generate Lorenz curve plots.
    bo_args : dict, optional
        Experimental; under testing and not yet documented. Leave unset.
    confidence_intervals_aggs : bool, str, list, or dict, default None
        Which bootstrap confidence intervals to compute. This is the
        shorthand for `bootstrap_args`, and it is applied last, so
        ``confidence_intervals_aggs=False`` switches every interval off
        whatever else was passed.

        * None -- keep the defaults (overall CIs on) or whatever
          `bootstrap_args` sets.
        * False / ``'none'`` / ``'off'`` -- no confidence intervals.
        * True / ``'overall'`` -- overall CIs only. This is the default.
        * ``'groups'`` -- per-group CIs only.
        * ``'both'`` / ``'all'`` -- overall and per-group CIs.
        * a list of indicator names -- overall CIs for those indicators,
          e.g. ``['Median', 'CoD']``.
        * a dict -- treated as `bootstrap_args`, short keys included.
    ci_mode : {'bootstrap', 'distribution'}, default 'distribution'
        Which family of confidence interval to build. `ci_conf_level`,
        `confidence_intervals_aggs` and `ci_indicators` mean the same thing in
        both; only the estimator behind each ``<indicator>_ci`` changes.

        * ``'bootstrap'`` -- resample the sample and read the interval off
          the resampling distribution, BCa by default. Works for every
          indicator in
          :data:`ratio_study.confint.CI_SUPPORTED_INDICATORS`, and needs
          ``min_count_threshold`` sales (20 by default) before it will
          produce one.
        * ``'distribution'`` -- use the closed-form formula the IAAO
          Exposure Draft (May 2026) Appendix D prescribes for each
          indicator: Student's t about the mean (D.1), the rank formula
          for the median (D.2), the ratio-estimator standard error for the
          weighted mean (D.3), the binomial tables for a median on 30
          sales or fewer (D.4), and Bonett-Seier for the CoD (D.6). Covers
          :data:`ratio_study.confint.DISTRIBUTION_CI_SUPPORTED_INDICATORS`
          -- ``Median``, ``Mean``, ``Weighted_Mean``, ``CoD``, ``PRB`` and
          ``PRD``. PRB and PRD are not in Appendix D: PRB keeps the
          interval its own slope regression always builds, at
          `ci_conf_level` rather than a separate formula, and PRD -- the
          ratio of ``Mean`` to ``Weighted_Mean`` -- gets a delta-method
          interval built from those two formulas
          (:func:`ratio_study.confint.prd_ci`). Nothing outside these six
          gets an interval in this mode. Carries no minimum-count gate:
          each formula is defined down to its own floor, and the binomial
          median tables reach 5 sales (90 %) or 6 (95 %).
          ``bootstrap_method``, ``bootstrap_resamplings``,
          ``bootstrap_concurrency`` and `ci_seed` are ignored.

        All three measures of valuation level -- ``Median``, ``Mean`` and
        ``Weighted_Mean`` -- are in `indicators` by default, so this mode
        reports the six intervals of the Standard's own demonstration
        report (Table 10.2) without being asked.
    ci_conf_level : float, default None
        Confidence level for the intervals, as a fraction (``0.9``) or a
        percentage (``90``). Defaults to 0.90, two-tailed, the level the
        Standard uses to decide compliance (s.11.1.2) and the level
        `vei_conf_level` and `vem_conf_level` default to. Note that
        ``ci_mode='distribution'`` has published tables for the median on
        small samples at 90 % and 95 % only; any other level there falls
        back to the exact binomial interval, with a message.
    ci_indicators : str or list of str, default None
        Indicators to build intervals for. Defaults to
        ``['Median', 'CoD', 'PRD', 'PRB']`` in bootstrap mode and to all
        six of
        :data:`ratio_study.confint.DISTRIBUTION_CI_SUPPORTED_INDICATORS`
        in distribution mode, intersected in both cases with
        `indicators`. Anything the mode cannot produce an interval for is
        dropped with a message. An indicator named here that the study
        does not compute is added to `indicators` in distribution mode --
        the closed forms are cheap -- and dropped with a message in
        bootstrap mode.
    ci_seed : int, default None
        Seed for the bootstrap resampler. Left unset, two identical runs
        publish slightly different intervals; set it when the numbers have
        to be reproducible. Unused when ``ci_mode='distribution'``, whose
        formulas are exact.
    bootstrap_args : dict, optional
        Full control over the bootstrap, for anything
        `confidence_intervals_aggs` / `ci_conf_level` / `ci_indicators` do not
        cover. Both the canonical keys and the short forms in
        ``ratio_study.confint._CI_ARG_ALIASES`` are accepted, so
        ``{'groups': True, 'level': 0.9}`` and
        ``{'bootstrap_cis_for_groups': True,
        'bootstrap_confint_level': 0.9}`` are equivalent. Unrecognised
        keys are reported rather than silently ignored.

        * bootstrap_cis_for_population : bool, default True -- overall CIs.
        * bootstrap_cis_for_groups : bool, default False -- per-group CIs.
        * bootstrap_method : str, default 'BCa' -- 'BCa' or 'percentile'.
        * bootstrap_resamplings : int, default 1000
        * bootstrap_resample_pct : float, default 1.0 -- share of the
          sample drawn in each resample, as a fraction (``0.2``) or a
          percentage (``20``). The classical bootstrap draws n of n; below
          1.0 this becomes an m-out-of-n bootstrap, which widens the
          intervals roughly as sqrt(n/m) and is the honest way to ask
          "how certain would I be on a smaller sample?". Each draw is
          floored at 20 sales (or n, if the sample is smaller), so a 20%
          draw from 50 sales takes 20, not 10.
        * bootstrap_confint_level : float, default 0.9
        * bootstrap_indicators : list of str, default
          ``['Median', 'CoD', 'PRD', 'PRB']``
        * ci_only : bool, default True -- if False, also compute mean/median/
          std on bootstrap samples.
        * min_count_threshold : int, default 20 -- minimum sales needed for
          bootstrap CIs. The whole sample and each group are gated on it
          independently; anything smaller is skipped and its CI columns
          come out blank rather than carrying an interval built on too few
          resampled sales.
        * bootstrap_seed : int, default None -- seed for the resampler.
        * bootstrap_concurrency -- 'processes', 'threads', or None.
    print_ci_settings : bool, default True
        Print the resolved confidence-interval settings before the study
        runs, marking any value that differs from the package default.
        Most of what governs an interval -- level, method, resamplings,
        resample size, the minimum count, the seed -- is a default nobody
        typed, so a published interval can rest on settings that were
        never reviewed. Set False in loops, or when the study produces no
        intervals.
    vei_percentile_method : str, default 'R7'
        Percentile interpolation method used in VEI calculations.
    vei_conf_level : float, default 0.9
        Confidence level for VEI confidence intervals.
    vei_compare_all_buckets : bool, default False
        Have the VEI compare the two quantile groups with the widest gap
        in median ratio, rather than the first and last. A **diagnostic,
        not the Appendix E acceptance test** -- see
        :func:`ratio_study.vei_analysis`. Reported once per study when
        `print_ci_settings` is on.
    vei_plot : bool, default False
        If True, generate the VEI plot.
    vei_plot_ref : str, default ''
        Reference label added to the VEI plot title/filename.
    vei_use_log_normal_delta : bool, default True
        If True, use the log-normal delta method for VEI confidence intervals.
    ve_plot_ref : str, default ''
        Reference label added to the vertical equity bands plot.
    vem_acceptable_inequity_level : float, default 0.1
        Acceptable inequity threshold used for the vertical equity
        moving-median indicator (VEM).
    vem_conf_level : float, default 0.9
        Confidence level for the band around the local (moving) median in
        the VEM, and for the `ci_out_of_bounds` test that decides the VEM
        verdict: a stretch of the value range only counts as inequitable
        if its whole local-median interval clears
        `vem_acceptable_inequity_level`. Also labels the band in the
        moving-median chart and the Vertical Equity Panel, so the legend
        and the verdict always quote the same level. Pass None to skip the
        band and compute the moving median alone -- the VEM verdict then
        needs `vem_acceptable_inequity_level` to be falsy too, since there
        is no interval left to test against.

        Separate from `ci_conf_level`, which sets the level of the indicator
        intervals in the results table, and from `vei_conf_level`. Set all
        three to the same number for a study that reports one level
        throughout.
    vem_ci_method : {'binom', 'R6', 'R7'}, default 'binom'
        Which median-CI family the VEM band uses, passed to
        :func:`ratio_study.confint.compute_median_ci`. ``'binom'`` is the
        exact sign-test interval; ``'R6'`` / ``'R7'`` follow the
        percentile-rank presentation of Appendix D.2. All three are
        order-statistic intervals bounded by the observed ratios.
    vem_plot : bool, default False
        If True, generate the vertical equity moving-median plot.
    vem_title : str, default 'Vertical Equity Moving Median'
        Title used for the vertical equity moving-median plot.
    round_values : bool, default True
        If True, round and cast sale/assessment values to integers for
        performance.
    req_significance : float, default 0.05
        Required significance level (p-value) for statistical tests.
    assessment_standards : dict, optional
        Acceptance thresholds keyed by indicator name, e.g.
        ``{'Median': (0.9, 1.1), 'PRD': (0.98, 1.03)}``. If not provided and
        `descriptive=True`, IAAO's strictest standards are used.
    graphs_args : dict, optional
        Options controlling chart generation:

        * pdf_overall_asr_graphs : bool, default True
        * pdf_group_asr_graphs : bool, default False
        * pdf_mosaic_group_asr_graphs : bool, default False
        * pdf_asr_graphs_x_vars_and_aliases : dict -- headers/aliases for
          SAR/ASR graphs.
        * pdf_grouping_aliases : dict -- display aliases for group labels.
        * facet_col_limit_per_page, facet_row_limit_per_page : int, optional
        * pdf_vertical_equity_panel : bool, default False in ``ratios``,
          True in ``ratios_report`` -- generate the
          overall Vertical Equity Panel (PRB / PRD / VEI / VEM gauges over
          the VEM moving-median and VEI boxplot charts). Titled
          ``'Vertical Equity Panel — Overall'``. Added to
          ``'overall_graphs'`` and also returned under
          ``'vertical_equity_panel'``. See
          :func:`ratio_study.graphs.vertical_equity_panel`.
        * pdf_vertical_equity_panel_groups : bool, default False -- also
          generate one panel per group of every ``group_by`` level, titled
          ``'Vertical Equity Panel — <level>: <group>'`` so a reader
          flipping through the report always knows which stratum a dial
          belongs to. Appended to ``'group_graphs'``.
        * pdf_vertical_equity_panel_min_count : int, default 30 -- skip the
          per-group panel for groups with fewer sales than this, noting the
          skip in the error messages. VEI and VEM both need about 30 sales
          before their verdicts mean anything.
        * pdf_asr_predictability : bool, default False -- add the ML/SHAP
          diagnostics panel: how much of the ASR a gradient-boosted model
          can predict from property characteristics, and which
          characteristics carry it. Titled
          ``'ASR Predictability — Overall'``. Requires the ``ml`` extra
          (``pip install "ratio_study[ml]"``); if it is missing, the panel
          is skipped with a note rather than failing the study. Added to
          ``'overall_graphs'``, with the full result dict under
          ``'asr_predictability'``. See
          :func:`ratio_study.ml_diagnostics.asr_predictability`.
        * asr_predictability_variables : list of str, default [] --
          characteristics to test. Required by
          ``pdf_asr_predictability``; sale price is excluded by design.
        * asr_predictability_shap_sample_pct : float, default 0.20 -- share
          of the sales frame explained with SHAP, as a fraction, sampled
          across the sorted ratio distribution. The main runtime dial.
        * asr_predictability_group_by : str, optional -- column for the
          CMBI / CDR subgroup breakdown inside the diagnostics.

    discrete_variables : list of str, optional
        Variables to treat as discrete rather than continuous.
    iqr_trim_multiple : float, default 0
        IQR multiple used to flag ASR outliers for trimming. 0 disables trimming.
    iqr_log_transform : bool, default True
        Compute the IQR trimming bounds on log(ASR) rather than the raw
        ratio, since ASR is right-skewed and raw-scale quartiles push the
        upper bound out, making high outliers easy to miss. No effect
        when ``iqr_trim_multiple`` is 0.
    iqr_group_by : str or list, optional
        Column(s) to compute the IQR trim independently within, instead
        of once over the whole sample. Same shape as ``group_by``:

        * a column name, or a list of column names -- each defines its
          own grouping, and a sale is trimmed if it is an outlier under
          any of them.
        * a list of lists -- the columns inside a sublist are combined
          into one composite grouping (e.g. county within neighbourhood).

        No effect when ``iqr_trim_multiple`` is 0.
    iqr_min_cod : float, default 0.06
        Minimum CoD a sample (or, with ``iqr_group_by``, a group) must
        have for the IQR trim to run on it. Below this, dispersion is
        already tight enough that trimming would just remove legitimate
        sales rather than genuine outliers. Set to 0 to always trim.
    iqr_filter_result : str or None, default 'print'
        What to do with the sales the IQR trim removed. Appendix C.6 of the
        Standard requires a report to state the basis for excluding outliers
        and identify what was removed, so the default is to say something
        rather than trim silently.

        * ``'print'`` -- print a summary: how many sales went, what share of
          the sample that is (Appendix C.5 caps it at 5-10%), and their total
          assessed and market value.
        * ``'save'`` -- also write the removed records to a ``.csv.gz`` in
          `export_directory`, then print the path and the summary.
        * a directory -- as ``'save'``, but written there.
        * a file path -- written exactly there, forced to ``.csv.gz``.
        * ``None`` / ``''`` -- silent. Use in loops and CI.

        The removed records are always available under
        ``'IQR_trimmed_sales'`` in the results dict regardless of this
        setting; the path written, if any, comes back under
        ``'IQR_trimmed_sales_export'``. No effect when ``iqr_trim_multiple``
        is 0.
    return_dict : bool, default False
        If True (or set automatically when plots/decompositions are
        produced), return the expanded results dict instead of just the
        ratios DataFrame.
    VE_bundle_comments : bool, default False
        If True, bundle vertical equity descriptive comments together.

    Returns
    -------
    DataFrame or dict
        DataFrame with the table of all ratios if
        `return_dict` is False.

        If True, a dict containing:

        * population ratios and Gini package for the population
        * Gini decomposition results (values, dataframes, plots)
        * DataFrame with the table of all ratios
        * graph objects

    Examples
    --------
    Overall 95 % distribution-mode intervals for Median, CoD, PRD and PRB
    come out of the box, as the ``Median_ci`` / ``CoD_ci`` / ``PRD_ci`` /
    ``PRB_ci`` columns:

    >>> ratios(sales, 'assessment', 'sale_price')          # doctest: +SKIP

    Turn them off when the run needs to be fast:

    >>> ratios(sales, 'assessment', 'sale_price',
    ...        confidence_intervals_aggs=False)                  # doctest: +SKIP

    Per-group intervals too, at 90 %:

    >>> ratios(sales, 'assessment', 'sale_price', group_by='nbhd',
    ...        confidence_intervals_aggs='both', ci_conf_level=90)    # doctest: +SKIP

    Just the two indicators that matter here:

    >>> ratios(sales, 'assessment', 'sale_price',
    ...        confidence_intervals_aggs=['Median', 'CoD'])      # doctest: +SKIP

    Anything else, through `bootstrap_args` (short keys accepted):

    >>> ratios(sales, 'assessment', 'sale_price', group_by='nbhd',
    ...        bootstrap_args={'groups': True, 'resamplings': 5000,
    ...                        'method': 'percentile'})     # doctest: +SKIP
    """

    # Track the input backend so we can return a frame in the same
    # backend when the user passed a polars / pyarrow frame.  Most
    # downstream operations (groupby, rolling, .str accessors, etc.)
    # require a real pandas DataFrame, so we convert at the entry
    # point and re-convert on the way out.
    _input_backend = nw_utils.native_backend(input_df)
    if not isinstance(input_df, pd.DataFrame):
        input_df = nw_utils.require_pandas(
            input_df, reason="ratios() uses pandas groupby/rolling/string accessors"
        )

    group_by = [] if group_by is None else list(group_by) if not isinstance(group_by, str) else group_by
    gini_args = {} if gini_args is None else dict(gini_args)
    bo_args = {} if bo_args is None else dict(bo_args)
    bootstrap_args = {} if bootstrap_args is None else dict(bootstrap_args)
    graphs_args = {} if graphs_args is None else dict(graphs_args)
    assessment_standards = {} if assessment_standards is None else dict(assessment_standards)
    discrete_variables = [] if discrete_variables is None else list(discrete_variables)
    indicators = list(DEFAULT_INDICATORS) if indicators is None else list(indicators)


    # LAZY IMPORTS
    if (graphs_args and 'pdf_overall_asr_graphs' in graphs_args and graphs_args['pdf_overall_asr_graphs']) \
        or ('pdf_group_asr_graphs' in graphs_args and graphs_args['pdf_group_asr_graphs']) or \
        ('pdf_mosaic_group_asr_graphs' in graphs_args and graphs_args['pdf_mosaic_group_asr_graphs']):
        try:
            from ratio_study.graphs import _reg_graph, _facet_grid_graphs
        except Exception as ex:
            raise RuntimeError('Error in ratios:', ex)

    # Opened before anything that can fail, so the handler at the bottom of
    # `ratios` always has a list to report through.
    error_messages = []

    # Resolve every way of asking for confidence intervals -- the
    # `confidence_intervals_aggs` / `ci_conf_level` / `ci_indicators` shorthands and the
    # `bootstrap_args` dict, short keys included -- into one canonical dict.
    # Deliberately outside the try below: this reads arguments only, never the
    # data, so a value it cannot make sense of is the caller's mistake and
    # should raise at the call site rather than come back as an error dict
    # where a DataFrame was expected. Recoverable problems come back as
    # messages instead.
    # Every confidence level goes through the one shared gate, here rather
    # than deep in the indicator that consumes it: the try below turns any
    # exception into an error dict, so a level validated downstream comes
    # back as a failed study instead of telling the caller what they typed
    # wrong. Same reasoning as `normalize_ci_args` immediately after.
    for _level_name, _level_value in (('vei_conf_level', vei_conf_level),
                                      ('vem_conf_level', vem_conf_level)):
        if _level_value is not None:
            confidence_level_as_fraction(_level_value, _level_name)

    # Same reasoning for the market value proxy's blend: validated here, out
    # in front of the try, so a share the caller cannot have meant names
    # itself instead of arriving as a failed study from inside `prb`.
    mvp_sale_price_share = mvp_sale_price_share_as_fraction(mvp_sale_price_share)

    bootstrap_args, ci_messages = normalize_ci_args(
        bootstrap_args=bootstrap_args
        , confidence_intervals_aggs=confidence_intervals_aggs
        , ci_conf_level=ci_conf_level
        , ci_indicators=ci_indicators
        , ci_seed=ci_seed
        , ci_mode=ci_mode
        , study_indicators=indicators
        , has_groups=bool(group_by)
    )
    _report_ci_messages(ci_messages, error_messages)

    # An interval the caller asked for by name, on an indicator the study was
    # not computing, brings that indicator into the study in distribution mode
    # rather than being dropped -- otherwise the interval would have nothing to
    # sit beside. See `normalize_ci_args`, `study_indicators`.
    indicators += [x for x in bootstrap_args.get('ci_study_indicator_additions', [])
                   if x not in indicators]

    try:
        # set default values for dict arguments not provided, otherwise use what was provided.
        std_gini_args = {'run_regression': True, 'gini_decomposition': False, 'gini_factors_variables': [], 'gini_lorenz_plots': False}
        std_graphs_args = {'pdf_overall_asr_graphs': True, 'pdf_group_asr_graphs': False, 'pdf_mosaic_group_asr_graphs': False, 'pdf_asr_graphs_x_vars_and_aliases': {'overall':[], 'groups':[], 'groups_mosaic':[]}, 'pdf_grouping_aliases':{}
            , 'facet_col_limit_per_page': None, 'facet_row_limit_per_page': None
            , 'pdf_vertical_equity_panel': False, 'pdf_vertical_equity_panel_groups': False
            , 'pdf_vertical_equity_panel_min_count': 30
            , 'pdf_asr_predictability': False, 'asr_predictability_variables': []
            , 'asr_predictability_shap_sample_pct': 0.20
            , 'asr_predictability_group_by': None}

        for key in std_gini_args.keys():
            if key not in gini_args: gini_args.update({key: std_gini_args[key]})
        for key in std_graphs_args.keys():
            if key not in graphs_args: graphs_args.update({key: std_graphs_args[key]})

        if dict_results is None:
            dict_results = {'exported_files': []}

        # set default IAAO's strictest assessment standards if they weren't provided
        if not assessment_standards and descriptive:
            assessment_standards = {'Median': (0.9, 1.1), 'PRB': (-0.05, 0.05), 'PRD': (0.98, 1.03), 'CoD': (0.05, 0.15), 'Mkt_desc': ''}

        if isinstance(group_by, str): #make it user friendly by accepting a single column as input for group_by
            group_by = [group_by]

        # check that inputs exist and all needed columns exist
        required_cols = set()
        required_cols, error_messages = _add_needed_variables(vars=[assessment_header, market_value_header], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        if len(required_cols) < 2:
            return {'errors': [save_errors_to_text(error_messages, os.path.join(export_directory, 'errors_in_ratio_study.txt')), error_messages]}

        if weights_header is not None:
            required_cols, error_messages = _add_needed_variables(vars=[weights_header], required_cols=required_cols, input_df=input_df, error_messages=error_messages)

        if mkt_value_proxy_header is not None:
            required_cols, error_messages = _add_needed_variables(vars=[mkt_value_proxy_header], required_cols=required_cols, input_df=input_df, error_messages=error_messages)

        if prb_mvp_header is not None:
            required_cols, error_messages = _add_needed_variables(vars=[prb_mvp_header], required_cols=required_cols, input_df=input_df, error_messages=error_messages)

        # force assessment, market_value_header and weights to be numeric
        try:
            for col in required_cols:
                input_df = input_df[pd.to_numeric(input_df[col], errors='coerce').notnull()]
        except Exception as ex:
            print('error', ex)

        # determine all necessary columns
        required_cols, error_messages = _add_needed_variables(vars=group_by, required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        if iqr_group_by:
            iqr_group_by_cols = [iqr_group_by] if isinstance(iqr_group_by, str) else list(iqr_group_by)
            required_cols, error_messages = _add_needed_variables(vars=iqr_group_by_cols, required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        required_cols, error_messages = _add_needed_variables(vars=graphs_args['pdf_asr_graphs_x_vars_and_aliases']['overall'], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        if 'groups_mosaic' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups_mosaic']:
            for mosaic_grouping in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups_mosaic']:
                required_cols, error_messages = _add_needed_variables(vars=mosaic_grouping, required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        if 'groups' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups']:
            for grouping in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups']:
                required_cols, error_messages = _add_needed_variables(vars=grouping, required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        if 'hist_groups' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['hist_groups']:
            for mosaic_grouping in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['hist_groups']:
                required_cols, error_messages = _add_needed_variables(vars=mosaic_grouping, required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        if 'dist_groups_mosaic' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['dist_groups_mosaic']:
            for grouping in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['dist_groups_mosaic']:
                required_cols, error_messages = _add_needed_variables(vars=grouping, required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        required_cols, error_messages = _add_needed_variables(vars=gini_args['gini_factors_variables'], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        # Characteristics the ML/SHAP diagnostics will test have to survive
        # the narrowing to required_cols, or they are gone by the time
        # asr_predictability runs.
        if graphs_args.get('pdf_asr_predictability'):
            required_cols, error_messages = _add_needed_variables(vars=graphs_args.get('asr_predictability_variables') or [], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
            if graphs_args.get('asr_predictability_group_by'):
                required_cols, error_messages = _add_needed_variables(vars=[graphs_args['asr_predictability_group_by']], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
        if bo_args:
            required_cols, error_messages = _add_needed_variables(vars=bo_args['characteristics'], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
            if 'stratify_by' in bo_args:
                required_cols, error_messages = _add_needed_variables(vars=bo_args['stratify_by'], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
            required_cols, error_messages = _add_needed_variables(vars=[bo_args['group_col']] if isinstance(bo_args['group_col'], str) else bo_args['group_col'], required_cols=required_cols, input_df=input_df, error_messages=error_messages)
            if 'detail_groups' in bo_args:
                required_cols, error_messages = _add_needed_variables(vars= leaf_values(bo_args['detail_groups']), required_cols=required_cols, input_df=input_df, error_messages=error_messages)
            if 'dummy_sets' in bo_args:
                required_cols, error_messages = _add_needed_variables(vars= leaf_values(bo_args['dummy_sets']), required_cols=required_cols, input_df=input_df, error_messages=error_messages)


        # for performance, drop all columns not deemed necessary. Keep a
        # reference to the wide frame first: if the IQR trim removes sales,
        # the audit export wants the whole record, not just the handful of
        # columns this study happens to need.
        # Retained only when a trim can actually happen -- on a large sales
        # file this is the whole frame, and there is no reason to hold it
        # otherwise. `.copy()` keeps the narrowed frame independent, so the
        # in-place cleaning below does not warn about writing to a slice.
        _records_before_narrowing = input_df if iqr_trim_multiple else None
        input_df = input_df[list(required_cols)].copy()

        input_df.dropna(subset=[assessment_header, market_value_header], inplace=True)
        input_df = input_df[(input_df[market_value_header] > 0) & (input_df[assessment_header] > 0)]

        if print_ci_settings:
            # After cleaning, so the reported resample size is the count the
            # bootstrap will actually draw from.
            print(describe_ci_settings(bootstrap_args, count=len(input_df)))
            if 'VEI' in indicators:
                # Appendix E.5: R6 and R7 can put boundary sales in different
                # groups and produce different VEIs, so which one ran is part
                # of the result. Reported here rather than from `vei_analysis`,
                # which runs once per group and once per bootstrap resample.
                _vei_note = (f'VEI quantile method: {vei_percentile_method}')
                if vei_compare_all_buckets:
                    _vei_note += ('; compare_all_buckets ON - widest pairwise gap, '
                                  'a diagnostic rather than the acceptance test')
                print(_vei_note)

        if iqr_trim_multiple:
            asr = input_df[assessment_header] / input_df[market_value_header]
            iqr_group_defs = [] if not iqr_group_by else (
                [iqr_group_by] if isinstance(iqr_group_by, str) else list(iqr_group_by)
            )
            total_assessed_value = input_df[assessment_header].sum()
            total_market_value = input_df[market_value_header].sum()
            if iqr_group_defs:
                iqr_flags = np.zeros(len(input_df), dtype=int)
                for group_header in iqr_group_defs:
                    if isinstance(group_header, list):
                        group_keys = input_df[group_header[0]].astype(str)
                        for col in group_header[1:]:
                            group_keys = group_keys + ', ' + input_df[col].astype(str)
                    else:
                        group_keys = input_df[group_header].astype(str)
                    iqr_flags = iqr_flags | iqr_filter(asr, iqr_mult=iqr_trim_multiple, log_transform=iqr_log_transform, groups=group_keys, min_cod=iqr_min_cod)
                input_df['asr_outlier'] = iqr_flags
            else:
                input_df['asr_outlier'] = iqr_filter(asr, iqr_mult=iqr_trim_multiple, log_transform=iqr_log_transform, min_cod=iqr_min_cod)
            _trimmed = input_df.loc[input_df['asr_outlier'] == 1].copy()
            _asr_trimmed = _trimmed[assessment_header] / _trimmed[market_value_header]
            # Widen back to the caller's full record where the index allows it,
            # so the excluded sales can actually be investigated -- Appendix C.2
            # asks for them to be scrutinised, which needs more than two columns.
            if _records_before_narrowing.index.is_unique:
                _trimmed = _records_before_narrowing.loc[_trimmed.index].copy()
            else:
                _trimmed = _trimmed.drop(columns=['asr_outlier'], errors='ignore')
            _trimmed['ASR'] = _asr_trimmed
            dict_results['IQR_trimmed_sales'] = _trimmed
            # Appendix C.6: a trim has to be reported, not just performed.
            dict_results['IQR_trimmed_sales_export'] = report_iqr_trim(
                dict_results['IQR_trimmed_sales']
                , assessment_header=assessment_header
                , market_value_header=market_value_header
                , total_assessed_value=total_assessed_value
                , total_market_value=total_market_value
                , result=iqr_filter_result
                , export_directory=export_directory
                , model_name=model_name
                , total_count=len(input_df))
            input_df=input_df.loc[input_df['asr_outlier'] == 0].drop(columns=['asr_outlier']).copy()

        if weights_header:
            input_df.dropna(subset=[weights_header], inplace=True)
            input_df[weights_header] = input_df[weights_header] / input_df[weights_header].mean()

        if log_transform:
            input_df[assessment_header] = np.exp(input_df[assessment_header])
            input_df[market_value_header] = np.exp(input_df[market_value_header])

        if round_values:
            input_df[[assessment_header, market_value_header]] = input_df[[assessment_header, market_value_header]].round(0).astype(float)
            input_df[[assessment_header, market_value_header]] = input_df[[assessment_header, market_value_header]].astype(int)

        if gini_args['gini_decomposition'] and not group_by:
            gini_args['gini_decomposition'] = False
            error_messages.append('input error: gini_decomposition requested, but no valid group_by was provided. Proceding without Gini Decomposition')
            if gen_log:
                logging.warning('input error: gini_decomposition requested, but no valid group_by was provided. Proceding without Gini Decomposition')

        # Initialised unconditionally: ``gini_factors_variables`` writes into
        # this dict without ``gini_decomposition`` ever running, and the
        # earlier indicator-gated guard left it unbound in that case.
        gini_decomp_results = {}

        # run population ratios
        population_ratios = ratios_calculator(input_df=input_df
                                              , market_value_header=market_value_header
                                              , assessment_header=assessment_header
                                              , mkt_value_proxy_header=mkt_value_proxy_header
                                              , prb_mvp_header=prb_mvp_header
                                              , mvp_sale_price_share=mvp_sale_price_share
                                              , weights_header=weights_header
                                              , req_significance=req_significance
                                              , run_regression=gini_args['run_regression']
                                              , indicators=indicators
                                              , vei_percentile_method=vei_percentile_method
                                              , vei_compare_all_buckets=vei_compare_all_buckets
                                              , ve_plot_ref = ve_plot_ref
                                              , vem_acceptable_inequity_level = vem_acceptable_inequity_level
                                              , vem_conf_level = vem_conf_level
                                              , vem_ci_method = vem_ci_method
                                              , ci_conf_level = bootstrap_args['bootstrap_confint_level']
                                              , vem_plot = vem_plot
                                              , vem_title=vem_title
                                              , gini_lorenz_plots=gini_args['gini_lorenz_plots']
                                              , gini_plot_ref=' '.join([model_name, 'Overall'])
                                              , vei_conf_level=vei_conf_level
                                              , vei_use_log_normal_delta=vei_use_log_normal_delta
                                              , vei_plot=vei_plot
                                              , vei_plot_ref=vei_plot_ref
                                              , export_directory=export_directory
                                              )

        if gini_args['gini_lorenz_plots']:
            dict_results['gini_lorenz_plots'] = [population_ratios['gini_lorenz_plot']]

        # The whole sample goes through the same gate as the groups, so the
        # overall row and the group rows always carry intervals from the
        # same estimator at the same level.
        population_cis = None
        if bootstrap_args['bootstrap_cis_for_population']:
            population_cis = _confidence_intervals_for_scope(
                input_df, assessment_header, market_value_header, weights_header
                , bootstrap_args=bootstrap_args
                , ratios_dict=population_ratios
                , count=len(input_df)
                , req_significance=req_significance
                , vei_conf_level=vei_conf_level
                , vei_use_log_normal_delta=vei_use_log_normal_delta
                , messages=error_messages)
        if population_cis:
            population_ratios.update(population_cis)
        else:
            _drop_unbootstrapped_prb_ci(population_ratios, bootstrap_args)
        dict_results['dict_population_ratios'] = {'ratios': population_ratios, 'gini_pkg': population_ratios}

        # Overall Gini package for the decomposition. ``gini_decomp`` /
        # ``_format_gini_decomp`` want a 2-sequence: [0] carries the Total row
        # (the Gini statistics plus ``Count``), [1] is the ungrouped package the
        # residual (interaction) effect is measured against. Only built when a
        # decomposition was actually asked for, since it costs a gini_stats run.
        _population_gini_pair = None
        if gini_args['gini_decomposition']:
            from ratio_study.gini import gini_stats as _gini_stats
            _overall_gini = _gini_stats(
                input_df[market_value_header], input_df[assessment_header]
                , weights=input_df[weights_header] if weights_header else None
                , run_regression=True)
            _population_gini_pair = ({**_overall_gini, 'Count': len(input_df)}, _overall_gini)

        population_ratios['Level'] = 'Overall'
        population_ratios['Group'] = ''

        df_results = pd.DataFrame([population_ratios])
        # ``custom_format`` (called later for ``df_results_formatted``)
        # relies on pandas; we restore the original backend at the
        # very end of the function.  This intermediate DataFrame is
        # always pandas so the formatting / merge / dropna passes
        # downstream are unchanged from the original behaviour.

        # make graphs
        dict_results['overall_graphs'] = []
        dict_results['group_mosaic_graphs'] = []
        dict_results['group_graphs'] = []
        dict_results['dist_plot_group_mosaic_graphs'] = []
        dict_results['histogram_overall_graphs'] = []

        # make overall asr scatter/boxplot graphs
        if 'pdf_overall_asr_graphs' in graphs_args and graphs_args['pdf_overall_asr_graphs'] and 'pdf_asr_graphs_x_vars_and_aliases' in graphs_args \
            and 'overall' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['overall']:
            input_df['graph_ASR'] = input_df[assessment_header] / input_df[market_value_header]
            for graph_variable in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['overall'].keys():
                x_ticks = None
                if graph_variable == 'rsMnthNum':  # special case for rsMnthNum
                    input_df['rsMnthNumLabel'] = input_df[graph_variable].str[3:]  # for full desc
                    input_df['rsMnthNum'] = input_df[graph_variable].str[:3].astype(int)
                    x_ticks = 'rsMnthNumLabel'

                elif graph_variable == 'rsYrNum':  # special case for rsYrNum
                    input_df['rsYrNumLabel'] = input_df[graph_variable].str[3:]  # for year only
                    input_df['rsYrNum'] = input_df[graph_variable].str[:3].astype(int)
                    x_ticks = 'rsYrNumLabel'

                if graph_variable in ['quality', 'building_class_cmv'] or input_df[graph_variable].dtype == 'O' or graph_variable[:7] == 'pctile_':
                    x_is_discrete = True
                else:
                    x_is_discrete = False

                graph = _reg_graph(df=input_df
                                   , col_y='graph_ASR'
                                   , col_x=graph_variable
                                   , y_label='Time-Adjusted ASR'
                                   , x_label=graphs_args['pdf_asr_graphs_x_vars_and_aliases']['overall'][graph_variable]
                                   , x_ticks=x_ticks
                                   , title='Sales Ratio by ' + graphs_args['pdf_asr_graphs_x_vars_and_aliases']['overall'][graph_variable]
                                   , x_is_discrete=x_is_discrete)

                if graph:
                    dict_results['overall_graphs'].append(graph)
                    return_dict = True

        # make overal asr histogram
        if 'pdf_overall_asr_graphs' in graphs_args and graphs_args['pdf_overall_asr_graphs'] and 'pdf_asr_graphs_x_vars_and_aliases' in graphs_args and 'overall_histogram' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['overall_histogram']:
            if 'graph_ASR' not in input_df:
                input_df['graph_ASR'] = input_df[assessment_header] / input_df[market_value_header]

            graph = _reg_graph(df=input_df
                               , col_x='graph_ASR'
                               , x_label='Time-Adjusted ASR'
                               , title='Histogram of Sales Ratio - Overall'
                               , histogram=True)
            if graph:
                dict_results['overall_graphs'].append(graph)
                return_dict = True

        # make overall VEI boxplot
        if 'pdf_overall_VEI_graphs' in graphs_args and graphs_args['pdf_overall_VEI_graphs']:
            asr = input_df[assessment_header] / input_df[market_value_header]
            if mkt_value_proxy_header is None:
                mkt_value_proxy = market_value_proxy(input_df[market_value_header],
                                                     input_df[assessment_header],
                                                     asr_md=population_ratios['Median'],
                                                     mvp_sale_price_share=mvp_sale_price_share)
            else:
                mkt_value_proxy = input_df[mkt_value_proxy_header]
            
            title_overall_vei = f'Vertical Equity Analysis - Boxplot of Ratios - Overall'

            vei_results = vei_analysis(asr=asr,
                 mkt_value_proxy=mkt_value_proxy,
                 output_format='dict',
                 metrics_only=True,
                 plot_ref = title_overall_vei,
                 gen_plot=True,
                 plot_return_object=True)

            if vei_results and 'vei_plot' in vei_results and vei_results['vei_plot']:
                dict_results['overall_graphs'].append(vei_results['vei_plot'])
                return_dict = True

        # make the overall Vertical Equity Panel: PRB / PRD / VEI / VEM
        # gauges over the VEI boxplot and VEM moving-median charts.
        if graphs_args.get('pdf_vertical_equity_panel'):
            from ratio_study.graphs import vertical_equity_panel

            panel = vertical_equity_panel(
                df=input_df
                , assessment_header=assessment_header
                , market_value_header=market_value_header
                , mkt_value_proxy_header=mkt_value_proxy_header
                , prb_mvp_header=prb_mvp_header
                , mvp_sale_price_share=mvp_sale_price_share
                , weights_header=weights_header
                # Reuse the population figures so the gauges cannot drift
                # from the ratios table.
                , prd_value=population_ratios.get('PRD')
                , prb_value=population_ratios.get('PRB')
                , cod_value=population_ratios.get('CoD')
                , assessment_standards=assessment_standards
                , req_significance=req_significance
                , vei_conf_level=vei_conf_level
                , vei_percentile_method=vei_percentile_method
                , vei_use_log_normal_delta=vei_use_log_normal_delta
                , vem_acceptable_inequity_level=vem_acceptable_inequity_level
                , vem_conf_level=vem_conf_level
                , title='Vertical Equity Panel — Overall'
                , subtitle=model_name if model_name and model_name != 'model' else ''
                , return_object=True
            )

            if panel:
                dict_results['vertical_equity_panel'] = panel
                dict_results['overall_graphs'].append(panel)
                return_dict = True

        # ML/SHAP: how much of the ASR is predictable from property
        # characteristics, and by what. Optional and off by default -- it
        # needs the 'ml' extra, takes seconds rather than milliseconds, and
        # requires the caller to nominate the characteristics to test.
        if graphs_args.get('pdf_asr_predictability'):
            ml_vars = [v for v in (graphs_args.get('asr_predictability_variables') or [])
                       if v in input_df.columns]
            missing_ml = [v for v in (graphs_args.get('asr_predictability_variables') or [])
                          if v not in input_df.columns]
            for v in missing_ml:
                error_messages.append(
                    'ASR predictability: ' + str(v) + ' is not contained in '
                    'input_df and was dropped from the diagnostics.')
            if not ml_vars:
                error_messages.append(
                    'ASR predictability skipped: no valid columns in '
                    "graphs_args['asr_predictability_variables'].")
            else:
                try:
                    from ratio_study.ml_diagnostics import (
                        asr_predictability, asr_predictability_panel)

                    ml_diag = asr_predictability(
                        input_df
                        , market_value_header=market_value_header
                        , assessment_header=assessment_header
                        , independent_variables=ml_vars
                        , group_by=graphs_args.get('asr_predictability_group_by')
                        , shap_sample_pct=graphs_args.get(
                            'asr_predictability_shap_sample_pct', 0.20)
                    )
                    ml_panel = asr_predictability_panel(
                        ml_diag
                        , title='ASR Predictability — Overall'
                        , return_object=True
                    )
                    if ml_panel:
                        dict_results['asr_predictability'] = ml_diag
                        dict_results['asr_predictability_panel'] = ml_panel
                        dict_results['overall_graphs'].append(ml_panel)
                        return_dict = True
                except ImportError as ex:
                    error_messages.append(
                        'ASR predictability skipped: ' + str(ex).splitlines()[0])
                except Exception as ex:
                    error_messages.append(
                        'ASR predictability failed: ' + str(ex))

        # if group_by specified, run group ratios
        if group_by:
            valid_group_by = []

            for group_header in group_by:
                if isinstance(group_header, list):
                    missing = [header for header in group_header if header not in input_df.columns]
                    if missing:
                        for header in missing:
                            error_messages.append('Error in grouped ratios: ' + header + ' is not contained in input_df. Please check for misspellings on headers')
                    else:
                        valid_group_by.append(group_header)
                else:
                    if group_header not in input_df.columns:
                        error_messages.append('Error in grouped ratios: ' + group_header + ' is not contained in input_df. Please check for misspellings on headers')
                    else:
                        valid_group_by.append(group_header)

            group_by = valid_group_by
            if group_by:

                # check if each group_by variable exists in input_df
                for group_header in group_by:
                    if isinstance(group_header, list):
                        for header in group_header:
                            if header not in input_df:
                                error_messages.append('Error in grouped ratios: ' + header + ' is not contained in input_df. Please check for misspellings on headers')
                                if gen_log:
                                    logging.warning('Error in grouped ratios: ' + header + ' is not contained in input_df. Please check for misspellings on headers')
                                group_by.remove(group_header)
                    else:
                        if group_header not in input_df:
                            error_messages.append('Error in grouped ratios: ' + group_header + ' is not contained in input_df. Please check for misspellings on headers')
                            if gen_log:
                                logging.warning('Error in grouped ratios: ' + group_header + ' is not contained in input_df. Please check for misspellings on headers')
                            group_by.remove(group_header)
                # group data
                for group_header in group_by:
                    group_df = input_df.copy()
                    group_df['group'] = ''
                    if isinstance(group_header, list):  # if there are nested hierarchical groups
                        level = ', '.join(group_header)
                        group_df['ratio_group'] = group_df[group_header[0]].astype(str)
                        i = 1
                        while i < len(group_header):
                            group_df['ratio_group'] = group_df['ratio_group'] + ', ' + group_df[group_header[i]].astype(str)
                            i += 1
                    else:
                        level = group_header
                        group_df['ratio_group'] = group_df[group_header]

                    ratio_groups = group_df['ratio_group'].drop_duplicates().tolist()  # determine groups

                    # run ratios for each grouping
                    ratios_results = []
                    for ratio_group in ratio_groups:
                        str_group_header = ','.join(str(g) for g in group_header) if isinstance(group_header, list) else group_header
                        group_ratios = ratios_calculator(input_df=group_df[group_df['ratio_group'] == ratio_group]
                                                        , market_value_header=market_value_header
                                                        , assessment_header=assessment_header
                                                        , mkt_value_proxy_header=mkt_value_proxy_header
                                                        , prb_mvp_header=prb_mvp_header
                                                        , mvp_sale_price_share=mvp_sale_price_share
                                                        , weights_header=weights_header
                                                        , req_significance=req_significance
                                                        , run_regression=True
                                                        , indicators=indicators
                                                        , gini_lorenz_plots=gini_args['gini_lorenz_plots']
                                                        , gini_plot_ref=model_name + ' by ' + str_group_header + '-' + str(ratio_group)
                                                        , vei_percentile_method=vei_percentile_method
                                                        , vei_compare_all_buckets=vei_compare_all_buckets
                                                        , ve_plot_ref = ve_plot_ref
                                                        , vem_acceptable_inequity_level = vem_acceptable_inequity_level
                                                        , vem_conf_level = vem_conf_level
                                                        , vem_ci_method = vem_ci_method
                                                        , ci_conf_level = bootstrap_args['bootstrap_confint_level']
                                                        , vem_plot = vem_plot
                                                        , vem_title=vem_title
                                                        , vei_conf_level=vei_conf_level
                                                        , vei_use_log_normal_delta=vei_use_log_normal_delta
                                                        , vei_plot=vei_plot
                                                        , vei_plot_ref=vei_plot_ref
                                                        , export_directory=export_directory)

                        if gini_args['gini_lorenz_plots']:
                            if 'gini_lorenz_plot' in group_ratios:
                                dict_results['gini_lorenz_plots'].append(group_ratios['gini_lorenz_plot'])

                        group_cis = None
                        if bootstrap_args['bootstrap_cis_for_groups']:
                            group_cis = _confidence_intervals_for_scope(
                                group_df[group_df['ratio_group'] == ratio_group]
                                , assessment_header, market_value_header, weights_header
                                , bootstrap_args=bootstrap_args
                                , ratios_dict=group_ratios
                                , count=group_ratios['Count']
                                , req_significance=req_significance
                                , vei_conf_level=vei_conf_level
                                , vei_use_log_normal_delta=vei_use_log_normal_delta
                                , messages=error_messages)
                        if group_cis:
                            group_ratios.update(group_cis)
                        else:
                            _drop_unbootstrapped_prb_ci(group_ratios, bootstrap_args)
                        # per-group Vertical Equity Panel, titled with the
                        # level and group so a reader flipping through the PDF
                        # always knows which stratum a dial belongs to.
                        if graphs_args.get('pdf_vertical_equity_panel_groups'):
                            min_panel_n = graphs_args.get(
                                'pdf_vertical_equity_panel_min_count', 30) or 0
                            if group_ratios.get('Count', 0) >= min_panel_n:
                                from ratio_study.graphs import vertical_equity_panel

                                group_slice = group_df[group_df['ratio_group'] == ratio_group]
                                group_panel = vertical_equity_panel(
                                    df=group_slice
                                    , assessment_header=assessment_header
                                    , market_value_header=market_value_header
                                    , mkt_value_proxy_header=mkt_value_proxy_header
                                    , prb_mvp_header=prb_mvp_header
                                    , mvp_sale_price_share=mvp_sale_price_share
                                    , weights_header=weights_header
                                    , prd_value=group_ratios.get('PRD')
                                    , prb_value=group_ratios.get('PRB')
                                    , cod_value=group_ratios.get('CoD')
                                    , assessment_standards=assessment_standards
                                    , req_significance=req_significance
                                    , vei_conf_level=vei_conf_level
                                    , vei_percentile_method=vei_percentile_method
                                    , vei_use_log_normal_delta=vei_use_log_normal_delta
                                    , vem_acceptable_inequity_level=vem_acceptable_inequity_level
                                    , vem_conf_level=vem_conf_level
                                    , title=f'Vertical Equity Panel — {level}: {ratio_group}'
                                    , subtitle=model_name if model_name and model_name != 'model' else ''
                                    , return_object=True
                                )
                                if group_panel:
                                    dict_results['group_graphs'].append(group_panel)
                                    return_dict = True
                            else:
                                error_messages.append(
                                    'Vertical Equity Panel skipped for ' + str(level)
                                    + ': ' + str(ratio_group) + ' — only '
                                    + str(int(group_ratios.get('Count', 0)))
                                    + ' sales, below the '
                                    + str(int(min_panel_n)) + ' minimum.')

                        group_ratios['Level'] = level
                        group_ratios['Group'] = ratio_group
                        ratios_results.append(group_ratios)
                    group_results = pd.DataFrame(ratios_results)
                    group_results.sort_values(by=['Group'], inplace=True)

                    if level == 'rsSaleDate':  # special formatting for sale date originally as int YYYYMM
                        group_results.loc[group_results['Group'] <= 0, 'Group'] = 200001
                        group_results['Group'] = pd.to_datetime(group_results['Group'], format='%Y%m')  # convert to datetime
                        group_results['Group'] = group_results['Group'].dt.strftime('%b-%Y')  # convert to string in MMM-YYYY format

                    df_results = pd.concat([df_results, group_results], ignore_index=True, axis=0)

                    # gini decomposition
                    if gini_args['gini_decomposition']:
                        # Lazy Import
                        from ratio_study.gini import gini_decomp, _format_gini_decomp
                        # ``gini_decomp`` and ``_format_gini_decomp`` both index
                        # ``population_ratios`` as a 2-sequence: [0] the overall
                        # Gini package plus ``Count`` for the Total row, [1] the
                        # ungrouped package the residual is measured against.
                        # The ratios dict is neither, so build the pair here.
                        gini_decomp_results[level], dict_results = gini_decomp(input_df[market_value_header]
                                                                    , input_df[assessment_header], group_df['ratio_group'], level
                                                                    , _population_gini_pair, export_directory
                                                                    ,dict_results = dict_results, unbiased=False, weights=input_df[weights_header] if weights_header else None)

                        gini_decomp_results[level]['dataframe'] = _format_gini_decomp(gini_decomp_results[level], level, _population_gini_pair)

        if len(gini_args['gini_factors_variables']) > 0:
            # Lazy Import
            from ratio_study.gini import gini_factors
            gini_decomp_results['gini_factors'] = gini_factors(data=input_df, depvar_header=assessment_header, indvar_headers=gini_args['gini_factors_variables'])

        if gini_args['gini_decomposition'] or len(gini_args['gini_factors_variables']) > 0:
            dict_results['dict_gini_decomposition'] = gini_decomp_results
            return_dict = True

# ── Blinder-Oaxaca Decomposition ───────────────────────────────
        if bo_args:
            std_bo_args = {
                'run_bo': False, 'group_col': None, 'characteristics': [],
                'log_asr': True, 'reference': 'sloczynski',
                'references': ['group_0', 'group_1', 'sloczynski'],
                'threshold': None, 'group_high_label': None,
                'group_low_label': None, 'dummy_sets': None,
                'detail_groups': None, 'alpha': 0.05,
                'stratify_by': None, 'bo_coefficient_plot': True,
                'bo_decomposition_plot': True, 'bo_detail_plot': True,
            }
            for key in std_bo_args:
                if key not in bo_args:
                    bo_args[key] = std_bo_args[key]

            if (bo_args.get('run_bo')
                and bo_args.get('group_col')
                and bo_args.get('characteristics')):
                # Lazy import
                from ratio_study.blinder_oaxaca import (
                    bo_decompose, bo_multi_decomposition, bo_format_table,
                    bo_coefficient_plot, bo_decomposition_plot, bo_detail_plot,
                )
                if 'graph_ASR' not in input_df.columns:
                    input_df['graph_ASR'] = (
                        input_df[assessment_header] / input_df[market_value_header]
                    )

                # Single decomposition (overall)
                bo_single = bo_decompose(
                    input_df, asr_col='graph_ASR',
                    group_col=bo_args['group_col'],
                    characteristics=bo_args['characteristics'],
                    log_asr=bo_args['log_asr'],
                    reference=bo_args['reference'],
                    threshold=bo_args.get('threshold'),
                    group_high_label=bo_args.get('group_high_label'),
                    group_low_label=bo_args.get('group_low_label'),
                    dummy_sets=bo_args.get('dummy_sets'),
                    detail_groups=bo_args.get('detail_groups'),
                    alpha=bo_args.get('alpha', 0.05),
                )
                if bo_single:
                    dict_results['bo_single'] = bo_single
                    dict_results['bo_formatted'] = bo_format_table(bo_single)
                    return_dict = True

                # Multi-strata / multi-model
                if (bo_args.get('stratify_by')
                    or len(bo_args.get('references', [])) > 1):
                    bo_multi = bo_multi_decomposition(
                        input_df, asr_col='graph_ASR',
                        group_col=bo_args['group_col'],
                        characteristics=bo_args['characteristics'],
                        stratify_by=bo_args.get('stratify_by'),
                        references=bo_args.get('references'),
                        log_asr=bo_args['log_asr'],
                        threshold=bo_args.get('threshold'),
                        group_high_label=bo_args.get('group_high_label'),
                        group_low_label=bo_args.get('group_low_label'),
                        dummy_sets=bo_args.get('dummy_sets'),
                        detail_groups=bo_args.get('detail_groups'),
                        alpha=bo_args.get('alpha', 0.05),
                    )
                    if bo_multi:
                        dict_results['bo_multi'] = bo_multi
                        return_dict = True

                # Plots
                bo_plots = []
                if bo_args.get('bo_coefficient_plot'):
                    fig = bo_coefficient_plot(bo_single,
                        filepath=os.path.join(export_directory,
                            'bo_coefficient_plot.png') if export_directory else None,
                        return_object=True)
                    if fig: bo_plots.append(fig)
                if bo_args.get('bo_detail_plot'):
                    for comp in ['Q', 'U']:
                        fig = bo_detail_plot(bo_single, component=comp,
                            filepath=os.path.join(export_directory,
                                f'bo_detail_{comp}.png') if export_directory else None,
                            return_object=True)
                        if fig: bo_plots.append(fig)
                if (bo_args.get('bo_decomposition_plot')
                    and 'bo_multi' in dict_results):
                    fig = bo_decomposition_plot(dict_results['bo_multi'],
                        filepath=os.path.join(export_directory,
                            'bo_decomposition_plot.png') if export_directory else None,
                        return_object=True)
                    if fig: bo_plots.append(fig)
                if bo_plots:
                    dict_results['bo_plots'] = bo_plots
                    return_dict = True


    # ── End Blinder-Oaxaca block ───────────────────────────────────

        # make group ASR graphs
        if 'pdf_group_asr_graphs' in graphs_args and graphs_args['pdf_group_asr_graphs'] and 'pdf_asr_graphs_x_vars_and_aliases' in graphs_args and 'groups' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups']:
            if 'graph_ASR' not in input_df:  # avoid re-calculating what was calculated in the overall graphs
                input_df['graph_ASR'] = input_df[assessment_header] / input_df[market_value_header]

            for group_graph_specs in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups']:
                if isinstance(group_graph_specs, dict):
                    keys_list = list(group_graph_specs.keys())
                else:
                    keys_list = group_graph_specs
                for grouped_graph_vars in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups']:
                    x_ticks = None
                    if 'rsMnthNum' in keys_list[2]:  # special case for rsMnthNum
                        if 'rsMnthNum' not in input_df and 'rsMnthNum' not in input_df:  # avoid re-calculating what was calculated in the overall graphs
                            input_df['rsMnthNumLabel'] = input_df[keys_list[2]].str[3:]  # for full desc
                            input_df['rsMnthNumLabel'] = input_df[keys_list[2]].str[3:]  # for full desc
                            input_df['rsMnthNum'] = input_df[keys_list[2]].str[:3].astype(int)
                        x_ticks = 'rsMnthNumLabel'

                    if 'rsYrNum' in keys_list[2]:  # special case for rsMnthNum
                        if 'rsMnthNum' not in input_df and 'rsMnthNum' not in input_df:  # avoid re-calculating what was calculated in the overall graphs
                            input_df['rsYrNumLabel'] = input_df[keys_list[2]].str[3:]  # for full desc
                            input_df['rsYrNum'] = input_df[keys_list[2]].str[:3].astype(int)
                        x_ticks = 'rsYrNumLabel'

                if keys_list[2] in discrete_variables or input_df[keys_list[2]].dtype == 'O' or keys_list[2][:7] == 'pctile_':
                    x_is_discrete = True
                else:
                    x_is_discrete = False

                row_cases = list(input_df[keys_list[0]].drop_duplicates())
                for row_case in row_cases:
                    title = 'Assessment To Sales Ratio by ' + group_graph_specs[keys_list[2]] + ': ' + group_graph_specs[keys_list[0]] + ' ' + str(row_case) + ', by ' + group_graph_specs[keys_list[1]]
                    graph = _facet_grid_graphs(df=input_df[input_df[keys_list[0]] == row_case]
                                              , group_col=keys_list[1]
                                              , col_y='graph_ASR'
                                              , col_x=keys_list[2]
                                              , y_label='Time-Adjusted ASR'
                                              , x_label=group_graph_specs[keys_list[2]]
                                              , group_col_label=''
                                              , title=title
                                              , lowess=0.5
                                              , zoom_in=False
                                              , x_is_discrete=x_is_discrete
                                              , x_ticks=x_ticks
                                              , facet_col_limit_per_page=graphs_args['facet_col_limit_per_page'])
                    title=None
                    if graph:
                        dict_results['group_graphs'].append(graph)
                        return_dict = True

        # make group VEI graphs
        if 'pdf_group_vei_graphs' in graphs_args and graphs_args['pdf_group_vei_graphs'] and 'pdf_VEI_boxplots_x_vars_and_aliases' in graphs_args and \
            'groups' in graphs_args['pdf_VEI_boxplots_x_vars_and_aliases'] and graphs_args['pdf_VEI_boxplots_x_vars_and_aliases']['groups']:

            for grouping_var, alias in graphs_args['pdf_VEI_boxplots_x_vars_and_aliases']['groups'].items():
                groups = sorted(list(input_df[grouping_var].drop_duplicates()))

                for group in groups:
                    title = f'Vertical Equity Analysis - Boxplot of Ratios within {alias} {str(group)}'
                    df_group = input_df[input_df[grouping_var] == group]
                    asr_group = df_group[assessment_header] / df_group[market_value_header]
                    if mkt_value_proxy_header is None:
                        mkt_value_proxy_group = market_value_proxy(
                            df_group[market_value_header],
                            df_group[assessment_header],
                            asr=asr_group,
                            mvp_sale_price_share=mvp_sale_price_share,
                            weights=df_group[weights_header] if weights_header else None)
                    else:
                        mkt_value_proxy_group = df_group[mkt_value_proxy_header]

                    vei_results_group = vei_analysis(asr=asr_group,
                        mkt_value_proxy=mkt_value_proxy_group,
                        output_format='dict',
                        metrics_only=True,
                        gen_plot=True,
                        plot_ref=title,
                        plot_return_object=True)

                    if vei_results_group and 'vei_plot' in vei_results_group and vei_results_group['vei_plot']:
                        dict_results['group_graphs'].append(vei_results_group['vei_plot'])
                        return_dict = True

        # make mosaic group graphs
        if 'pdf_mosaic_group_asr_graphs' in graphs_args and graphs_args['pdf_mosaic_group_asr_graphs'] and 'pdf_asr_graphs_x_vars_and_aliases' in graphs_args and 'groups_mosaic' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups_mosaic']:
            if 'graph_ASR' not in input_df:  # avoid re-calculating what was calculated in the overall graphs
                input_df['graph_ASR'] = input_df[assessment_header] / input_df[market_value_header]
            for mosaic_graph_specs in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['groups_mosaic']:
                keys_list = list(mosaic_graph_specs.keys())

                x_ticks = None
                if 'rsMnthNum' in keys_list[2]:  # special case for rsMnthNum
                    if 'rsMnthNum' not in input_df and 'rsMnthNum' not in input_df:  # avoid re-calculating what was calculated in the overall graphs
                        input_df['rsMnthNumLabel'] = input_df[keys_list[2]].str[3:]  # for full desc
                        input_df['rsMnthNum'] = input_df[keys_list[2]].str[:3].astype(int)
                    x_ticks = 'rsMnthNumLabel'

                if 'rsYrNum' in keys_list[2]:  # special case for rsMnthNum
                    if 'rsMnthNum' not in input_df and 'rsMnthNum' not in input_df:  # avoid re-calculating what was calculated in the overall graphs
                        input_df['rsYrNumLabel'] = input_df[keys_list[2]].str[3:]  # for full desc
                        input_df['rsYrNum'] = input_df[keys_list[2]].str[:3].astype(int)
                    x_ticks = 'rsYrNumLabel'

                if keys_list[2] in discrete_variables or input_df[keys_list[2]].dtype == 'O' or keys_list[2][:7] == 'pctile_':
                # if keys_list[2] in discrete_variables or input_df[graph_variable].dtype == 'O' or graph_variable[:7] == 'pctile_':
                    x_is_discrete = True
                else:
                    x_is_discrete = False
                
                # set groups of groups if image needs to fit in page.
                if 'facet_col_limit_per_page' in graphs_args and graphs_args['facet_col_limit_per_page']:
                    col_groups = list(input_df[keys_list[1]].drop_duplicates())
                    if len(col_groups) > graphs_args['facet_col_limit_per_page']:
                        col_groups = _split_list(col_groups, graphs_args['facet_col_limit_per_page'])
                    else:
                        col_groups = [col_groups]
                if 'facet_row_limit_per_page' in graphs_args and graphs_args['facet_row_limit_per_page']:
                    row_groups = list(input_df[keys_list[0]].drop_duplicates())
                    if len(row_groups) > graphs_args['facet_row_limit_per_page']:
                        row_groups = _split_list(row_groups, graphs_args['facet_row_limit_per_page'])
                    else:
                        row_groups = [row_groups]
                title = 'Assessment To Sales Ratio by ' + mosaic_graph_specs[keys_list[2]] + ', stratified by ' + mosaic_graph_specs[keys_list[0]] + ' and ' + mosaic_graph_specs[keys_list[1]]
                if row_groups and col_groups:
                    for col_group in col_groups:
                        for row_group in row_groups:
                            subset_df = input_df[(input_df[keys_list[0]].isin(row_group)) & (input_df[keys_list[1]].isin(col_group))]
                            if len(subset_df) >= 1:
                                graph = _facet_grid_graphs(df=subset_df
                                                    , group_row=keys_list[0]
                                                    , group_col=keys_list[1]
                                                    , col_y='graph_ASR'
                                                    , col_x=keys_list[2]
                                                    , y_label='Time-Adjusted ASR'
                                                    , x_label=mosaic_graph_specs[keys_list[2]]
                                                    , title=title
                                                        # 'Sales Ratio by ' + mosaic_graph_specs[keys_list[2]]
                                                    , lowess=0.5
                                                    , zoom_in=False
                                                    , x_is_discrete=x_is_discrete
                                                    , facet_col_limit_per_page=graphs_args['facet_col_limit_per_page'])
                                title=None
                                if graph:
                                    dict_results['group_mosaic_graphs'].append(graph)
                                    return_dict = True

        # make mosaic distribution group graphs
        if 'pdf_mosaic_group_asr_graphs' in graphs_args and graphs_args['pdf_mosaic_group_asr_graphs'] and 'pdf_asr_graphs_x_vars_and_aliases' in graphs_args and 'dist_groups_mosaic' in graphs_args['pdf_asr_graphs_x_vars_and_aliases'] and graphs_args['pdf_asr_graphs_x_vars_and_aliases']['dist_groups_mosaic']:
            if 'graph_ASR' not in input_df:  # avoid re-calculating what was calculated in the overall graphs
                input_df['graph_ASR'] = input_df[assessment_header] / input_df[market_value_header]
            for mosaic_graph_specs in graphs_args['pdf_asr_graphs_x_vars_and_aliases']['dist_groups_mosaic']:
                keys_list = list(mosaic_graph_specs.keys())

                hue_groups = list(input_df[keys_list[1]].drop_duplicates().sort_values())
                if len(hue_groups) > 7:
                    hue_groups = _split_list(hue_groups, 7)
                else:
                    hue_groups = [hue_groups]

                if 'facet_row_limit_per_page' in graphs_args and graphs_args['facet_row_limit_per_page']:
                    row_groups = list(input_df[keys_list[0]].drop_duplicates().sort_values())
                    if len(row_groups) > graphs_args['facet_row_limit_per_page']:
                        row_groups = _split_list(row_groups, graphs_args['facet_row_limit_per_page'])
                    else:
                        row_groups = [row_groups]

                title = 'Distribution of Assessment To Sales Ratio, stratified by ' + mosaic_graph_specs[keys_list[0]] + ' and ' + mosaic_graph_specs[keys_list[1]]
                if row_groups and hue_groups:
                    for hue_group in hue_groups:
                        for row_group in row_groups:
                            subset_df = input_df[(input_df[keys_list[0]].isin(row_group)) & (input_df[keys_list[1]].isin(hue_group))]
                            if len(subset_df) >= 1:
                                graph = _facet_grid_graphs(df=subset_df
                                                        , group_row=keys_list[0]
                                                        , group_col=keys_list[1]
                                                        , col_x='graph_ASR'
                                                        , x_label='Time-Adjusted ASR'
                                                        , group_row_label=mosaic_graph_specs[keys_list[1]]
                                                        , group_col_label=mosaic_graph_specs[keys_list[0]]
                                                        , title=title
                                                        , dist_plot=True
                                                        , facet_col_limit_per_page=graphs_args['facet_col_limit_per_page']
                                                        )
                                title = None
                                if graph:
                                    dict_results['dist_plot_group_mosaic_graphs'].append(graph)
                                    return_dict = True

        # Format results dataframe
        # Columns follow a fixed reading order rather than the order the caller
        # happened to list `indicators` in, so every study in a series presents
        # the same way: identification, then level of valuation (Standard on
        # Ratio Studies s.7), then uniformity within the stratum (s.8.1), then
        # vertical equity (s.8.2), then accuracy.
        indicators_headers = [x for x in INDICATOR_DISPLAY_ORDER if x in indicators]
        accuracy_headers = [x for x in ACCURACY_DISPLAY_ORDER if x in indicators]
        if {'MKI', 'RRI'}.intersection(set(indicators)):
            basic_gini_headers = ['Reranking_Index', 'Modified_Kakwani_Index']
            regression_gini_headers = ['Redistributive_Effect', 'Reranking_Effect', 'Kakwani_Index']
            additional_gini_headers = ['Gini_Sale_Prices', 'Gini_Assessments', 'Concentration_Index']
        
        additional_bootstrap_headers = [x for x in df_results.columns if len(re.findall('_btst_md$|_btst_mn$|_btst_std$|_btst_krt$|_btst_skw$', x))]

        # compose list of headers in resulting summary
        results_cols = ['Level', 'Count']
        if group_by:
            results_cols.append('Group')
            results_cols = ['Level', 'Group', 'Count'] + [col for col in results_cols if col not in ['Level', 'Group', 'Count']]

        if 'gini_complete' in indicators:
            gini_headers = additional_gini_headers + basic_gini_headers
            if gini_args['run_regression']:
                gini_headers = gini_headers + regression_gini_headers
        elif {'MKI', 'RRI'}.intersection(set(indicators)):
            gini_headers = basic_gini_headers
        else:
            gini_headers = []

        if 'VEI' in indicators:
            # VEI_kw_statistic / VEI_kw_pvalue: Kruskal-Wallis across all
            # quantile groups (Appendix E.4), supplementary to the VEI
            # itself. Reaches the CSV/Excel export via `results_cols` below,
            # but not the PDF report, which selects its own columns via
            # `pdf_ratios_table_headers` and does not default to including
            # them -- room on the printed page is scarce.
            vei_headers = ['VEI', 'VEI_conclusion', 'VEI_kw_statistic', 'VEI_kw_pvalue']
        else:
            vei_headers = []

        if 'VEM' in indicators:
            ve_headers = ['Inequity_conclusion','Inequity_severity','Inequity_prevalence']
        else:
            ve_headers = []

        results_cols = (results_cols + indicators_headers + vei_headers + ve_headers
                        + gini_headers + accuracy_headers)

        suffixes_to_add = {}
        if bootstrap_args['bootstrap_cis_for_population'] or bootstrap_args['bootstrap_cis_for_groups']:
            ci_columns = list(bootstrap_args['bootstrap_indicators'])
            # PRB is the one indicator with an interval of its own, from the
            # slope regression, and it is now built at `ci_conf_level` like
            # everything else -- so it can be shown whenever the bootstrap
            # is not already filling the column. When PRB *is* bootstrapped
            # the bootstrap owns the column and
            # `_drop_unbootstrapped_prb_ci` clears the regression value from
            # the rows it skipped, so one estimator fills any given column
            # for the whole run.
            if 'PRB' in indicators and 'PRB' not in ci_columns:
                ci_columns.append('PRB')
            suffixes_to_add['_ci'] = ci_columns
        if gini_args['run_regression']:
            suffixes_to_add['_Pvalue'] = ['Gini_Sale_Prices', 'Gini_Assessments', 'Concentration_Index', 'Kakwani_Index', 'Modified_Kakwani_Index']

        results_cols = _add_suffixes_to_list(results_cols, suffixes_to_add)
        if bootstrap_args['ci_only'] == False and len(additional_bootstrap_headers) > 0:
            results_cols = results_cols + additional_bootstrap_headers

        results_cols = [x for x in results_cols if x in df_results.columns]
        df_results_formatted = df_results.loc[:, results_cols]

        results_format = {'Count': '{:,.0f}', 'CoD': '{:.2%}', 'CoD_rsd': '{:.2%}', 'CoDw': '{:.2%}', 'RRI': '{:.2%}', 'Reranking_Index': '{:.2%}', 'CoV': '{:.2%}', 'CoC': '{:.2%}', 'MAPE': '{:.2%}',
                          'MdAPE': '{:.2%}', 'RMSE': '{:,.0f}', 'MAE': '{:,.0f}', 'VEI': '{:.2f}', 'Inequity_conclusion': '{:.2f}', 'Inequity_severity': '{:.2f}', 'Inequity_prevalence': '{:.2f}',  'MAE': '{:,.0f}', 'All others': '{:,.4f}'}

        if descriptive:
            # Lazy import
            from ratio_study.reporting import _add_comments
            # One minimum count for the whole study: the row that is too small
            # to carry a confidence interval is the row that is too small to
            # comment on, so the comments and the bootstrap share the gate
            # rather than each keeping its own default.
            df_results_formatted = _add_comments(df_results_formatted, req_significance, assessment_standards, results_cols, results_format, skip=['Level', 'Group', 'FSD']
                                                 , ci_conf_level=bootstrap_args['bootstrap_confint_level']
                                                 , min_count_threshold=bootstrap_args['min_count_threshold']
                                                 , VE_bundle_comments=VE_bundle_comments)
        else:
            df_results_formatted = custom_format(df_results_formatted, results_cols, results_format, skip=['Level', 'Group', 'FSD'])

        # rename for MKI and RRI
        if 'Reranking_Index' in df_results_formatted.columns:
            df_results_formatted.rename(columns={'Reranking_Index': 'RRI'}, inplace=True)
        if 'Modified_Kakwani_Index' in df_results_formatted.columns:
            df_results_formatted.rename(columns={'Modified_Kakwani_Index': 'MKI'}, inplace=True)
        if 'Modified_Kakwani_Index_Pvalue' in df_results_formatted.columns:
            df_results_formatted.rename(columns={'Modified_Kakwani_Index_Pvalue': 'MKI_pvalue'}, inplace=True)

        if 'Reranking_Index_ci' in df_results_formatted.columns:
            df_results_formatted.rename(columns={'Reranking_Index_ci': 'RRI_ci'}, inplace=True)
        if 'Modified_Kakwani_Index_ci' in df_results_formatted.columns:
            df_results_formatted.rename(columns={'Modified_Kakwani_Index_ci': 'MKI_ci'}, inplace=True)

        # A blank cell reads as "not computed here"; 'NaN' reads as a result.
        # Confidence intervals skipped on a small group, indicators skipped on
        # a group of fewer than four sales, and unwritten acceptance comments
        # all land here, so the presentation frame carries empty strings.
        # ``.fillna('')`` alone raises on the nullable-boolean
        # ``<indicator>_acceptable`` flags, hence the cast.
        df_results_formatted = (df_results_formatted.astype(object)
                                .where(df_results_formatted.notna(), ''))

        # `VEI_pg_stats` is the per-quantile-group detail behind the VEI: 
        # list of dicts per row, useful in the results *dict* (the panel reads
        # it) but meaningless as a table column. It never belongs in either
        # output frame.
        df_results = df_results.drop(columns=['VEI_pg_stats'], errors='ignore')

        # Give the unformatted frame the same reading order as the formatted
        # one, so `df_all_ratios` and `df_all_ratios_formatted` line up.
        _ordered = [c for c in results_cols if c in df_results.columns]
        df_results = df_results.loc[:, _ordered + [c for c in df_results.columns if c not in _ordered]]

        # Restore the original backend on the way out, so a polars /
        # pyarrow input gets a polars / pyarrow DataFrame back.
        if _input_backend == "polars":
            try:
                import polars as pl
                # The formatted DataFrame contains string-typed columns
                # (e.g. ``'-0.05896*'`` for PRB non-significant).  Cast
                # any object-dtype columns to ``str`` first so that
                # ``pl.from_pandas`` does not try to coerce them to
                # numeric.
                df_to_convert = df_results.copy()
                for col in df_to_convert.columns:
                    if df_to_convert[col].dtype == object:
                        df_to_convert[col] = df_to_convert[col].astype(str)
                df_to_convert_fmt = df_results_formatted.copy()
                for col in df_to_convert_fmt.columns:
                    if df_to_convert_fmt[col].dtype == object:
                        df_to_convert_fmt[col] = df_to_convert_fmt[col].astype(str)
                df_results = pl.from_pandas(df_to_convert)
                df_results_formatted = pl.from_pandas(df_to_convert_fmt)
            except Exception as ex:
                warnings.warn(f"ratios(): could not restore polars backend: {ex}")
        elif _input_backend == "pyarrow":
            try:
                import pyarrow as pa
                df_results = pa.Table.from_pandas(df_results)
                df_results_formatted = pa.Table.from_pandas(df_results_formatted)
            except Exception as ex:
                warnings.warn(f"ratios(): could not restore pyarrow backend: {ex}")

        dict_results['df_all_ratios'] = df_results
        dict_results['df_all_ratios_formatted'] = df_results_formatted
        dict_results['error_messages'] = error_messages
        if return_dict == False:
            return dict_results['df_all_ratios_formatted']
        else:
            return dict_results

    except Exception as ex:
        error_messages += ['Exception: error in ratios:' + str(ex)]
        return {'errors': [save_errors_to_text(error_messages, os.path.join(export_directory, 'errors_in_ratio_study.txt')), error_messages]}

# ---------------------------------------------------------------------------------------------------------------------

def rts_calculator(input_df=None
                      , sale_count=None
                      , assessment_header=None
                      , market_value_header=None
                      , mkt_value_proxy_header=None
                      , prb_mvp_header=None
                      , mvp_sale_price_share=DEFAULT_MVP_SALE_PRICE_SHARE
                      , market_value_array=None
                      , assessment_array=None
                      , weights_header=None
                      , weights_array=None
                      , asr_array=None
                      , sale_error_array=None
                      , indicators=['Median', 'Mean', 'Weighted_Mean', 'CoD', 'PRD', 'PRB', 'VEI', 'VEM']  # keep in step with DEFAULT_INDICATORS
                      , req_significance=0.05
                      , vei_conf_level=0.9
                      , vei_percentile_method='R7'
                      , vei_compare_all_buckets=False
                      , vei_plot=False
                      , vei_plot_ref=''
                      , vei_use_log_normal_delta = True
                      , vem_acceptable_inequity_level = 0.05
                      , vem_conf_level = 0.9
                      , vem_ci_method = 'binom'
                      , vem_plot = False
                      , vem_title = 'Vertical Equity Moving Median'
                      , ve_plot_ref = ''
                      , ci_conf_level = None
                      , gini_lorenz_plots=False
                      , gini_plot_ref=''
                      , export_directory=Path.cwd()
                      , run_regression=True
                      , skip_fsd=False
                      , skip_count=False
                      , for_bootstrapping=False
                      , prb_non_significant_marker='*'
                      , log_transform=False):
    """
    Calculate ratio study indicators (Median, CoD, PRD, PRB, VEI, etc.)
    from a dataframe or raw arrays, and optionally Gini-related statistics.

    Parameters
    ----------
    input_df : DataFrame, optional
        Pandas dataframe with sales, assessments, and any additional columns
        required for the selected indicators. Not required if the relevant
        ``*_array`` parameters are supplied instead.
    sale_count : int, optional
        Explicit sale count. If not provided, the number of rows in
        `input_df` is used.
    assessment_header : str, optional
        Column name in `input_df` for the assessed / predicted value.
    market_value_header : str, optional
        Column name in `input_df` for the market value / sale amount.
    mkt_value_proxy_header : str, optional
        Column name in `input_df` for a market value proxy used in VEI
        calculations. If None, a proxy is constructed internally.
    prb_mvp_header : str, optional
        Column name in `input_df` for the market-value proxy `prb` uses in
        its regression, overriding the blend it builds itself. Independent
        of `mkt_value_proxy_header`.
    mvp_sale_price_share : float, default 0.5
        Share of the sale price in the market value proxy built for the PRB
        regression and the VEI grouping -- see
        :func:`ratio_study.market_value_proxy`. Ignored for whichever proxy
        is supplied as a column.
    market_value_array : ndarray, optional
        Array of sale amounts. Overrides `market_value_header` if provided.
    assessment_array : ndarray, optional
        Array of assessed / predicted values. Overrides `assessment_header`
        if provided.
    weights_header : str, optional
        Column name in `input_df` for sample weights, assumed to have mean 1.
    weights_array : ndarray, optional
        Array of weights. Overrides `weights_header` if provided.
    asr_array : ndarray, optional
        Array of assessment-to-sale ratios. If not provided, ratios are
        computed as assessment / market value.
    sale_error_array : ndarray, optional
        Array of relative sale errors, defined as
        (assessment - market_value) / market_value. Computed internally
        if required by the selected indicators.
    indicators : list of str, default ['Median', 'Mean', 'Weighted_Mean', 'CoD', 'PRD', 'PRB', 'VEI', 'VEM']
        Indicators to compute.
    req_significance : float, default 0.05
        Required significance level (p-value) for PRB calculations
        (e.g. 0.05 for 5%).
    vei_conf_level : float, default 0.9
        Confidence level used for VEI and vertical equity bands analysis.
    vei_percentile_method : str, default 'R7'
        Percentile interpolation method used in VEI calculations.
    vei_plot : bool, default False
        If True, generate the VEI plot.
    vei_plot_ref : str, default ''
        Reference label added to the VEI plot title/filename.
    vei_use_log_normal_delta : bool, default True
        If True, use the log-normal delta method for VEI confidence
        intervals.
    vem_acceptable_inequity_level : float, default 0.1
        Acceptable inequity threshold used for the vertical equity
        moving-median indicator (VEM).
    vem_conf_level : float, default 0.9
        Confidence level for the band around the local (moving) median in
        the VEM, and for the `ci_out_of_bounds` test that decides the VEM
        verdict: a stretch of the value range only counts as inequitable
        if its whole local-median interval clears
        `vem_acceptable_inequity_level`. Also labels the band in the
        moving-median chart and the Vertical Equity Panel, so the legend
        and the verdict always quote the same level. Pass None to skip the
        band and compute the moving median alone -- the VEM verdict then
        needs `vem_acceptable_inequity_level` to be falsy too, since there
        is no interval left to test against.

        Separate from `ci_conf_level`, which sets the level of the indicator
        intervals in the results table, and from `vei_conf_level`. Set all
        three to the same number for a study that reports one level
        throughout.
    vem_ci_method : {'binom', 'R6', 'R7'}, default 'binom'
        Which median-CI family the VEM band uses, passed to
        :func:`ratio_study.confint.compute_median_ci`. ``'binom'`` is the
        exact sign-test interval; ``'R6'`` / ``'R7'`` follow the
        percentile-rank presentation of Appendix D.2. All three are
        order-statistic intervals bounded by the observed ratios.
    vem_plot : bool, default False
        If True, generate the vertical equity moving-median plot.
    vem_title : str, default 'Vertical Equity Moving Median'
        Title used for the vertical equity moving-median plot.
    ve_plot_ref : str, default ''
        Reference label added to the vertical equity bands and
        moving-median plot titles/filenames.
    gini_lorenz_plots : bool, default False
        If True, generate Lorenz curve plots for Gini-related statistics.
    gini_plot_ref : str, default ''
        Reference label added to the Lorenz curve plot title/filename.
    export_directory : str or Path, default current working directory
        Directory where generated plots are saved.
    run_regression : bool, default True
        If True, also compute regression-based Gini metrics such as the
        Reranking Index and Kakwani-related statistics.
    skip_fsd : bool, default False
        If True, skip computing FSD. Used for bootstrap-based confidence
        intervals, where the string-based FSD result would be invalid.
    skip_count : bool, default False
        If True, skip adding the sale count to the output. Used for
        bootstrap-based confidence intervals.
    for_bootstrapping : bool, default False
        If True, skip PRB significance warnings and return numeric outputs
        only, suitable for bootstrap calculations.
    prb_non_significant_marker : str, default '*'
        Marker appended to the PRB result when the statistic is not
        significant.
    log_transform : bool, default False
        If True, `market_value_header`/`assessment_header` values are
        assumed to be in log scale and are exponentiated before analysis.

    Returns
    -------
    dict
        Dictionary of the requested ratio study indicators, plus, when
        applicable, Gini-related statistics and VEI/VE-bands/Lorenz-curve
        plot objects.
    """
    try:
        if sale_count:
            n = sale_count
        else:
            n = len(input_df.index)
        results = {}
        gini = {}

        if input_df is not None:
            # Coerce to pandas at the boundary; ``rts_calculator`` mutates
            # ``input_df`` and is called by ``ratios()`` which has already
            # converted upstream.
            if not isinstance(input_df, pd.DataFrame):
                input_df = nw_utils.require_pandas(
                    input_df, reason="rts_calculator() uses pandas column assignment"
                )
            input_df = input_df.copy()

        if log_transform:
            input_df[market_value_header] = np.exp(input_df[market_value_header])
            input_df[assessment_header] = np.exp(input_df[assessment_header])
        market_value = market_value_array if market_value_array is not None else np.array(input_df[market_value_header])
        assessment = assessment_array if assessment_array is not None else np.array(input_df[assessment_header])
        if prb_mvp_header:
            prb_mvp = np.array(input_df[prb_mvp_header])
        else:
            prb_mvp = None

        if weights_array is not None:
            weights = weights_array
        else:
            weights = np.array(input_df[weights_header]) if weights_header is not None else None
        if asr_array is not None:
            asr = asr_array
        else:
            asr = assessment / market_value

        if sale_error_array is not None:
            sale_error = sale_error_array
        else:
            if any(item in indicators for item in ['MdAPE', 'MAPE', 'FSD']):
                sale_error = (assessment - market_value) / market_value

        if skip_count == False:
            results['Count'] = n

        results['Median'] = asr_median(asr, weights=weights)
        if 'Mean' in indicators: results['Mean'] = asr_mean(asr, weights=weights)
        if 'CoD' in indicators: results['CoD'] = cod(asr, results['Median'], weights=weights) if n > 3 else np.nan
        if 'CoD_rsd' in indicators: results['CoD_rsd'] = cod_residual(market_value, assessment, weights=weights) if n > 3 else np.nan
        if 'CoDw' in indicators:
            # Value-weighted COD. Sample weights compound with the value
            # weights rather than being replaced by them, so a supplied
            # weights_header is not silently dropped for this one indicator.
            codw_weights = market_value if weights is None else market_value * weights
            results['CoDw'] = cod(asr, results['Median'], weights=codw_weights) if n > 3 else np.nan
        if 'PRD' in indicators: results['PRD'] = prd(market_value, asr, weights=weights) if n > 3 else np.nan

        if for_bootstrapping:
            if 'PRB' in indicators and n > 3:
                results['PRB'] = prb(market_value, assessment, asr, market_value_proxy=prb_mvp, asr_md=results['Median'], req_significance=0, non_significant_marker='', weights=weights, mvp_sale_price_share=mvp_sale_price_share)[0] if n > 3 else np.nan
        else:
            if 'PRB' in indicators:
                if n <= 3:
                    results['PRB'] = results['PRB_ci'] = np.nan
                else:
                    prb_results = prb(market_value, assessment, asr, market_value_proxy=prb_mvp, asr_md=results['Median'], req_significance=req_significance, non_significant_marker=prb_non_significant_marker, weights=weights, ci_conf_level=ci_conf_level, mvp_sale_price_share=mvp_sale_price_share)
                    results['PRB'] = prb_results[0] if n > 3 else np.nan
                    results['PRB_ci'] = prb_results[2] if n > 3 else np.nan

        if 'VEI' in indicators: 
            if mkt_value_proxy_header is None:
                mkt_value_proxy = market_value_proxy(input_df[market_value_header],
                                                     input_df[assessment_header],
                                                     asr_md=results['Median'],
                                                     mvp_sale_price_share=mvp_sale_price_share)
            else:
                mkt_value_proxy = input_df[mkt_value_proxy_header]

            vei_results = vei_analysis(asr=asr
                                       , mkt_value_proxy=mkt_value_proxy
                                       , conf_level=vei_conf_level
                                       , output_format='dict'
                                       , metrics_only=True
                                       , gen_plot=vei_plot
                                       , percentile_method=vei_percentile_method
                                       , plot_ref = vei_plot_ref
                                       , plot_filepath = os.path.join(export_directory, 'VEI_boxplot_' + vei_plot_ref.replace(' ', '_') + '.png')
                                       , vei_use_log_normal_delta=vei_use_log_normal_delta
                                       , compare_all_buckets=vei_compare_all_buckets
                                       , print_settings=False)

            
            results['VEI'] = vei_results['vei']
            results['VEI_sig'] = vei_results['vei_sig']
            results['VEI_conclusion'] = vei_results['conclusion']
            results['VEI_pg_stats'] = vei_results['pg_stats']

            # Kruskal-Wallis across all quantile groups (Appendix E.4,
            # "Further Analysis") -- supplementary to the VEI, not part of
            # its pass/fail test. NaN ratiosratioselow the 20-sale minimum, where
            # `vei_analysis` never formed the groups to test.
            kruskal_wallis = vei_results.get('kruskal_wallis')
            results['VEI_kw_statistic'] = kruskal_wallis['statistic'] if kruskal_wallis else np.nan
            results['VEI_kw_pvalue'] = kruskal_wallis['pvalue'] if kruskal_wallis else np.nan

            if vei_plot:
                results['vei_plot'] = vei_results['vei_plot']

        if 'VEM' in indicators:
            input_df['asr'] = asr
            window_size = 1000 if n > 10000 else int(n/10)
            vem_results = vem(df=input_df,
                                        market_value_proxy=mkt_value_proxy_header if mkt_value_proxy_header else market_value_header,
                                        window_size=window_size,
                                        generate_plot=vem_plot,
                                        plot_title=vem_title,
                                        plot_filepath=os.path.join(export_directory, 'vem_Plot_' + ve_plot_ref.replace(' ', '_') + '.png'),
                                        plot_return_object=True,
                                        plot_ref=ve_plot_ref,
                                        acceptable_inequity_level=vem_acceptable_inequity_level,
                                        conf_level=vem_conf_level,
                                        ci_method=vem_ci_method
                                        )
            results['Inequity_conclusion'] = vem_results['Inequity_conclusion']
            results['Inequity_severity'] = vem_results['Inequity_severity']
            results['Inequity_prevalence'] = vem_results['Inequity_prevalence']
            if vem_plot:
                results['vem_plot'] = vem_results['plot']

        if 'RMSE' in indicators: results['RMSE'] = rmse(market_value, assessment, weights=weights) if n > 3 else np.nan
        if 'MAE' in indicators: results['MAE'] = mae(market_value, assessment, weights=weights) if n > 3 else np.nan
        if 'CoV' in indicators: results['CoV'] = cov_adj(asr, weights=weights) if n > 3 else np.nan
        if 'CoC_10' in indicators: results['CoC_10'] = coc(asr, results['Median'], within_prcnt=0.10, weights=weights) if n > 3 else np.nan
        if 'CoC_15' in indicators: results['CoC_15'] = coc(asr, results['Median'], within_prcnt=0.15, weights=weights) if n > 3 else np.nan
        if 'CoC_20' in indicators: results['CoC_20'] = coc(asr, results['Median'], within_prcnt=0.20, weights=weights) if n > 3 else np.nan
        if 'MdAPE' in indicators: results['MdAPE'] = mdmape(sale_error, weights=weights) if n > 3 else np.nan
        if 'MAPE' in indicators: results['MAPE'] = mape(sale_error, weights=weights) if n > 3 else np.nan
        if skip_fsd == False:
            if 'FSD' in indicators: results['FSD'] = fsd(sale_error, weights=weights) if n > 3 else np.nan
        # The weighted mean used to be a by-product of the Gini block, so
        # asking for it on its own produced no column -- and, with
        # distribution-based CIs, a `Weighted_Mean_ci` with nothing to sit
        # beside. It is a ratio of two totals, so it costs nothing to
        # compute directly; the Gini block below recomputes the same value.
        if 'Weighted_Mean' in indicators:
            if weights is None:
                results['Weighted_Mean'] = float(assessment.sum() / market_value.sum())
            else:
                results['Weighted_Mean'] = float(
                    (assessment * weights).sum() / (market_value * weights).sum()
                )
        if n > 3 and {'MKI', 'RRI', 'gini_complete', 'Gini_Sale_Prices', 'Gini_Assessments', 'Concentration_Index', 'Redistributive_Effect', 'Reranking_Effect', 'Kakwani_Index', 'Modified_Kakwani_Index'}.intersection(set(indicators)):
            # Lazy Import
            from ratio_study.gini import gini_stats
            gini = gini_stats(market_value, assessment, run_regression=run_regression, plot_ref=gini_plot_ref, gini_lorenz_plots=gini_lorenz_plots, export_directory=export_directory, weights=weights)
            if 'gini_complete' in indicators:
                results['Gini_Sale_Prices'] = gini['Gini_Sale_Prices']
                results['Gini_Assessments'] = gini['Gini_Assessments']
                results['Concentration_Index'] = gini['Concentration_Index']
            results['Redistributive_Effect'] = gini['Redistributive_Effect']
            results['Weighted_Mean'] = assessment.sum() / market_value.sum()

            if run_regression:
                if 'gini_complete' in indicators:
                    results['Reranking_Effect'] = gini['Reranking_Effect']
                    results['Kakwani_Index'] = gini['Kakwani_Index']
                    results['Kakwani_Index_Pvalue'] = gini['Regressions']['Kakwani_Index']['slope']['pvalue']
                    results['Gini_Sale_Prices_Pvalue'] = gini['Regressions']['Gini_Sale_Prices']['slope']['pvalue']
                    results['Gini_Assessments_Pvalue'] = gini['Regressions']['Gini_Assessments']['slope']['pvalue']
                    results['Concentration_Index_Pvalue'] = gini['Regressions']['Concentration_Index']['slope']['pvalue']
                results['Reranking_Index'] = gini['Reranking_Index']
                results['Modified_Kakwani_Index'] = gini['Modified_Kakwani_Index']
                results['Modified_Kakwani_Index_Pvalue'] = float('nan')

            if gini_lorenz_plots:
                results['gini_lorenz_plot'] = gini['gini_lorenz_plot']
        return results
    except Exception as ex:
        raise RuntimeError('error in ratio study: rts_calculator', ex)
    
# Using a wrapper below to fix an issue with vscode's docstring (when typing) coming from another function.
def ratios_calculator(**kwargs):
    return rts_calculator(**kwargs)

# ---------------------------------------------------------------------------------------------------------------------

def _ratios_calculator_wrap(input_dict):
    """
    Call `ratios_calculator` from a single dictionary argument, for use with
    ``concurrent.futures`` executors that map a function over one iterable
    of arguments (as when parallelizing bootstrap resampling).

    Parameters
    ----------
    input_dict : dict
        Dictionary with the keys needed to run `ratios_calculator`:

        * market_value_array : ndarray -- array of sale amounts.
        * assessment_array : ndarray -- array of assessed / predicted
          values.
        * sale_count : int -- number of sales.
        * asr_array : ndarray -- array of assessment-to-sale ratios.
        * sale_error_array : ndarray, optional -- array of relative sale
          errors. Defaults to None if not present.
        * weights_array : ndarray, optional -- array of sample weights.
          Defaults to None if not present.
        * req_significance : float -- required significance level
          (p-value) for the PRB.
        * vei_conf_level : float, optional -- confidence level for VEI
          analysis. Defaults to 0.9 if not present, matching `ratios`.
        * vei_plot : bool, optional -- if True, generate the VEI plot.
          Defaults to False if not present.
        * vei_plot_ref : str, optional -- reference label for the VEI
          plot. Defaults to '' if not present.
        * vei_percentile_method : str -- percentile interpolation method
          used in VEI calculations.
        * vei_use_log_normal_delta : bool -- if True, use the log-normal
          delta method for VEI confidence intervals.
        * indicators : list of str -- indicators to compute.
        * gini_lorenz_plots : bool -- if True, generate Lorenz curve
          plots.
        * gini_plot_ref : str -- reference label for the Lorenz curve
          plot.

    Returns
    -------
    dict
        Dictionary of ratio study indicators, as returned by
        `ratios_calculator` (called here with `run_regression=True`,
        `skip_fsd=True`, `skip_count=True`, and `for_bootstrapping=True`).
    """
    try:
        result = ratios_calculator(market_value_array=input_dict['market_value_array']
                                   , assessment_array=input_dict['assessment_array']
                                   , sale_count = input_dict['sale_count']
                                   , asr_array=input_dict['asr_array']
                                   , sale_error_array=input_dict['sale_error_array'] if 'sale_error_array' in input_dict else None
                                   , weights_array=input_dict['weights_array'] if 'weights_array' in input_dict else None
                                   , req_significance=input_dict['req_significance']
                                   # 0.9, not 0.1: this is a confidence level,
                                   # not a significance level, and every other
                                   # default in the package spells it 0.9.
                                   , vei_conf_level=input_dict['vei_conf_level'] if 'vei_conf_level' in input_dict else 0.9
                                   , vei_plot=input_dict['vei_plot'] if 'vei_plot' in input_dict else False
                                   , vei_plot_ref=input_dict['vei_plot_ref'] if 'vei_plot_ref' in input_dict else ''
                                   , vei_percentile_method=input_dict['vei_percentile_method']
                                   , vei_use_log_normal_delta=input_dict['vei_use_log_normal_delta']
                                   , run_regression=True
                                   , skip_fsd=True
                                   , skip_count=True
                                   , for_bootstrapping=True
                                   , indicators=input_dict['indicators']
                                   , gini_lorenz_plots=input_dict['gini_lorenz_plots']
                                   , gini_plot_ref=input_dict['gini_plot_ref'])

        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: ratios_calculator_wrap:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def ratios_report(sales_df
        , assessment_header
        , market_value_header
        , mkt_value_proxy_header=None
        , prb_mvp_header=None
        , mvp_sale_price_share=DEFAULT_MVP_SALE_PRICE_SHARE
        , group_by=[]
        , export_directory= Path.cwd()
        , return_dict_results=False
        , weights_header=None
        , descriptive=True
        , indicators=None
        , model_name='model'
        , export_filename_date_ref=''
        , gini_args = {}
        , bo_args={}
        , confidence_intervals_aggs=None
        , ci_mode='distribution'
        , ci_conf_level=None
        , ci_indicators=None
        , ci_seed=None
        , bootstrap_args = {}
        , assessment_standards={}
        , export_ratios=''
        , export_in_compacted_file='zip'
        , pdf_args = {}
        , generate_log=False
        , log_transform=False
        , base_date=''
        , req_significance=0.05
        , vei_percentile_method='R7'
        , vei_compare_all_buckets=False
        , vei_conf_level=0.9
        , vem_conf_level=0.9
        , vem_ci_method='binom'
        , discrete_variables=[]
        , iqr_trim_multiple=0
        , iqr_log_transform=True
        , iqr_group_by=None
        , iqr_min_cod=0.06
        , iqr_filter_result='print'
        , print_ci_settings=True
        , return_dict=True
        , VE_bundle_comments=False
        , excel_sensitivity_tag=None
        , site_id = None
        ):

    """
    Run a ratio study on model results, with the option of generating
    reports in PDF, XLSX and CSV.

    Parameters
    ----------
    sales_df : DataFrame
        Sales dataframe containing assessed values, market values, and any
        additional columns needed for grouping or indicators.
    assessment_header : str
        Column name in `sales_df` for the assessed / predicted value.
    market_value_header : str
        Column name in `sales_df` for the market value / sale amount.
    mkt_value_proxy_header : str, optional
        Column to use as a proxy for market value (e.g. in VEI calculations)
        when `market_value_header` isn't the appropriate reference.
    prb_mvp_header : str, optional
        Column to use as the market-value proxy in the PRB regression,
        overriding `prb`'s default 50/50 blend of `market_value_header`
        and the ASR-adjusted assessment. Independent of
        `mkt_value_proxy_header`, which is for VEI.
    group_by : list of str, default []
        Grouping column names.
    export_directory : str or Path, default current working directory
        Location where the results will be saved.
    return_dict_results : bool, default False
        If True, the function returns a dictionary with all results
        instead of exported file paths.
    weights_header : str, optional
        Column name with sample weights.
    descriptive : bool, default True
        If True, ratios are evaluated against `assessment_standards` and
        reported with asterisks and comments where applicable.
    indicators : list of str, default ['Median', 'Mean', 'Weighted_Mean', 'CoD', 'PRD', 'PRB', 'VEI', 'VEM']
        Indicators to compute.
    model_name : str, default 'model'
        Model name. Used for naming exported files and adding a title to
        charts.
    export_filename_date_ref : str, default ''
        Date/time reference appended to exported file names. If not
        provided, the current date/time is used.
    gini_args : dict, optional
        Options for the Gini decomposition:

        * run_regression : bool, default True -- also compute Reranking
          Effect, Reranking Index, Kakwani Index, and associated p-values.
        * gini_decomposition : bool, default False -- run the Gini
          decomposition (also triggers export of a Gini decomposition
          Excel file).
        * gini_factors_variables : list of str, default [] -- independent
          model variables to decompose the Gini coefficient by.
        * gini_lorenz_plots : bool, default False -- generate Lorenz curve
          plots.
    bo_args : dict, optional
        Experimental; under testing and not yet documented. Leave unset.
    confidence_intervals_aggs : bool, str, list, or dict, default None
        Which bootstrap confidence intervals to compute. This is the
        shorthand for `bootstrap_args`, and it is applied last, so
        ``confidence_intervals_aggs=False`` switches every interval off
        whatever else was passed.

        * None -- keep the defaults (overall CIs on) or whatever
          `bootstrap_args` sets.
        * False / ``'none'`` / ``'off'`` -- no confidence intervals.
        * True / ``'overall'`` -- overall CIs only. This is the default.
        * ``'groups'`` -- per-group CIs only.
        * ``'both'`` / ``'all'`` -- overall and per-group CIs.
        * a list of indicator names -- overall CIs for those indicators,
          e.g. ``['Median', 'CoD']``.
        * a dict -- treated as `bootstrap_args`, short keys included.
    ci_mode : {'bootstrap', 'distribution'}, default 'distribution'
        Which family of confidence interval to build. `ci_conf_level`,
        `confidence_intervals_aggs` and `ci_indicators` mean the same thing in
        both; only the estimator behind each ``<indicator>_ci`` changes.

        * ``'bootstrap'`` -- resample the sample and read the interval off
          the resampling distribution, BCa by default. Works for every
          indicator in
          :data:`ratio_study.confint.CI_SUPPORTED_INDICATORS`, and needs
          ``min_count_threshold`` sales (20 by default) before it will
          produce one.
        * ``'distribution'`` -- use the closed-form formula the IAAO
          Exposure Draft (May 2026) Appendix D prescribes for each
          indicator: Student's t about the mean (D.1), the rank formula
          for the median (D.2), the ratio-estimator standard error for the
          weighted mean (D.3), the binomial tables for a median on 30
          sales or fewer (D.4), and Bonett-Seier for the CoD (D.6). Covers
          :data:`ratio_study.confint.DISTRIBUTION_CI_SUPPORTED_INDICATORS`
          -- ``Median``, ``Mean``, ``Weighted_Mean``, ``CoD``, ``PRB`` and
          ``PRD``. PRB and PRD are not in Appendix D: PRB keeps the
          interval its own slope regression always builds, at
          `ci_conf_level` rather than a separate formula, and PRD -- the
          ratio of ``Mean`` to ``Weighted_Mean`` -- gets a delta-method
          interval built from those two formulas
          (:func:`ratio_study.confint.prd_ci`). Nothing outside these six
          gets an interval in this mode. Carries no minimum-count gate:
          each formula is defined down to its own floor, and the binomial
          median tables reach 5 sales (90 %) or 6 (95 %).
          ``bootstrap_method``, ``bootstrap_resamplings``,
          ``bootstrap_concurrency`` and `ci_seed` are ignored.

        All three measures of valuation level -- ``Median``, ``Mean`` and
        ``Weighted_Mean`` -- are in `indicators` by default, so this mode
        reports the six intervals of the Standard's own demonstration
        report (Table 10.2) without being asked.
    ci_conf_level : float, default None
        Confidence level for the intervals, as a fraction (``0.9``) or a
        percentage (``90``). Defaults to 0.90, two-tailed, the level the
        Standard uses to decide compliance (s.11.1.2) and the level
        `vei_conf_level` and `vem_conf_level` default to. Note that
        ``ci_mode='distribution'`` has published tables for the median on
        small samples at 90 % and 95 % only; any other level there falls
        back to the exact binomial interval, with a message.
    ci_indicators : str or list of str, default None
        Indicators to build intervals for. Defaults to
        ``['Median', 'CoD', 'PRD', 'PRB']`` in bootstrap mode and to all
        six of
        :data:`ratio_study.confint.DISTRIBUTION_CI_SUPPORTED_INDICATORS`
        in distribution mode, intersected in both cases with
        `indicators`. Anything the mode cannot produce an interval for is
        dropped with a message. An indicator named here that the study
        does not compute is added to `indicators` in distribution mode --
        the closed forms are cheap -- and dropped with a message in
        bootstrap mode.
    ci_seed : int, default None
        Seed for the bootstrap resampler. Left unset, two identical runs
        publish slightly different intervals; set it when the numbers have
        to be reproducible. Unused when ``ci_mode='distribution'``, whose
        formulas are exact.
    bootstrap_args : dict, optional
        Full control over the bootstrap, for anything
        `confidence_intervals_aggs` / `ci_conf_level` / `ci_indicators` do not
        cover. Both the canonical keys and the short forms in
        ``ratio_study.confint._CI_ARG_ALIASES`` are accepted, so
        ``{'groups': True, 'level': 0.9}`` and
        ``{'bootstrap_cis_for_groups': True,
        'bootstrap_confint_level': 0.9}`` are equivalent. Unrecognised
        keys are reported rather than silently ignored.

        * bootstrap_cis_for_population : bool, default True -- calculate
          bootstrapped confidence intervals for the population of sales.
        * bootstrap_cis_for_groups : bool, default False -- calculate
          bootstrapped confidence intervals also for each grouping.
        * bootstrap_method : str, default 'BCa' -- 'BCa' or 'percentile'.
        * bootstrap_resamplings : int, default 1000 -- number of bootstrap
          resamplings.
        * bootstrap_resample_pct : float, default 1.0 -- share of the
          sample drawn in each resample, as a fraction (``0.2``) or a
          percentage (``20``). The classical bootstrap draws n of n; below
          1.0 this becomes an m-out-of-n bootstrap, which widens the
          intervals roughly as sqrt(n/m) and is the honest way to ask
          "how certain would I be on a smaller sample?". Each draw is
          floored at 20 sales (or n, if the sample is smaller), so a 20%
          draw from 50 sales takes 20, not 10.
        * bootstrap_confint_level : float, default 0.9 -- confidence
          level for confidence interval calculations.
        * bootstrap_indicators : list of str, default
          ``['Median', 'CoD', 'PRD', 'PRB']``
        * ci_only : bool, default True -- if False, also compute mean/
          median/standard deviation on bootstrap samples.
        * min_count_threshold : int, default 20 -- minimum sales needed for
          bootstrap CIs, applied to the whole sample and to each group
          independently.
        * bootstrap_seed : int, default None -- seed for the resampler.
        * bootstrap_concurrency -- 'processes', 'threads', or None.
    assessment_standards : dict, optional
        Acceptance thresholds keyed by indicator name. Used for comments
        when `descriptive` is True.
    export_ratios : str, default ''
        Dictates how generated ratios are exported: '' (no export),
        'xlsx', 'csv', or 'both'.
    export_in_compacted_file : str, default 'zip'
        If 'zip' or 'gzip' ('gz'/'tar.gz' are normalized to 'gzip'), all
        exported files are compacted into a single zip/gzip archive.
        Any other value disables compaction.
    pdf_args : dict, optional
        Options controlling the PDF report and its charts:

        * gen_pdf_report : bool, default True -- if True, a ratios and
          graphs PDF file is generated.
        * pdf_ratios_table_headers : list of str, default ['Median', 'CoD', 'PRB', 'VEI'] -- headers used in the PDF's summarized ratios tables.
        * pdf_overall_asr_graphs : bool, default True -- generate overall
          SAR/ASR graphs.
        * pdf_group_asr_graphs : bool, default False -- generate grouped
          SAR/ASR graphs.
        * pdf_mosaic_group_asr_graphs : bool, default False -- generate
          mosaic SAR/ASR graphs.
        * pdf_overall_VEI_graphs : bool, optional -- generate the overall
          VEI boxplot.
        * pdf_vertical_equity_panel : bool, default True -- generate the
          overall Vertical Equity Panel (PRB / PRD / VEI / VEM gauges over
          the VEM moving-median and VEI boxplot charts). Titled
          ``'Vertical Equity Panel — Overall'``. Added to
          ``'overall_graphs'`` and also returned under
          ``'vertical_equity_panel'``. See
          :func:`ratio_study.graphs.vertical_equity_panel`.
        * pdf_vertical_equity_panel_groups : bool, default False -- also
          generate one panel per group of every ``group_by`` level, titled
          ``'Vertical Equity Panel — <level>: <group>'`` so a reader
          flipping through the report always knows which stratum a dial
          belongs to. Appended to ``'group_graphs'``.
        * pdf_vertical_equity_panel_min_count : int, default 30 -- skip the
          per-group panel for groups with fewer sales than this, noting the
          skip in the error messages. VEI and VEM both need about 30 sales
          before their verdicts mean anything.
        * pdf_asr_predictability : bool, default False -- add the ML/SHAP
          diagnostics panel: how much of the ASR a gradient-boosted model
          can predict from property characteristics, and which
          characteristics carry it. Titled
          ``'ASR Predictability — Overall'``. Requires the ``ml`` extra
          (``pip install "ratio_study[ml]"``); if it is missing, the panel
          is skipped with a note rather than failing the study. Added to
          ``'overall_graphs'``, with the full result dict under
          ``'asr_predictability'``. See
          :func:`ratio_study.ml_diagnostics.asr_predictability`.
        * asr_predictability_variables : list of str, default [] --
          characteristics to test. Required by
          ``pdf_asr_predictability``; sale price is excluded by design.
        * asr_predictability_shap_sample_pct : float, default 0.20 -- share
          of the sales frame explained with SHAP, as a fraction, sampled
          across the sorted ratio distribution. The main runtime dial.
        * asr_predictability_group_by : str, optional -- column for the
          CMBI / CDR subgroup breakdown inside the diagnostics.
        * pdf_asr_graphs_x_vars_and_aliases : dict, default {'overall': [], 'groups': [], 'groups_mosaic': []} -- headers and aliases for ASR graphs.
        * pdf_VEI_boxplots_x_vars_and_aliases : dict, optional -- headers
          and aliases for VEI boxplots.
        * pdf_grouping_aliases : dict, default {} -- display aliases for
          grouping variables.
        * pdf_starting_comments, pdf_ending_comments : list of str,
          default [] -- comments prepended/appended to the PDF report.
        * facet_col_limit_per_page : int, default 3 -- max facet columns
          per PDF page.
        * facet_row_limit_per_page : int, default 4 -- max facet rows per
          PDF page.
        * classification_header : str, optional -- classification header
          shown in the PDF.
    generate_log : bool, default False
        If True, a log file is saved to `export_directory`.
    log_transform : bool, default False
        If the sales file has log prices instead of prices, set this to
        True so they're converted back to price scale.
    base_date : str, default ''
        Reference date shown in the PDF report.
    req_significance : float, default 0.05
        Required significance level (p-value) for statistical tests.
    vei_percentile_method : str, default 'R7'
        Percentile interpolation method used in VEI calculations.
    vei_conf_level : float, default 0.9
        Confidence level for VEI confidence intervals.
    vei_compare_all_buckets : bool, default False
        Have the VEI compare the two quantile groups with the widest gap
        in median ratio, rather than the first and last. A **diagnostic,
        not the Appendix E acceptance test** -- see
        :func:`ratio_study.vei_analysis`. Reported once per study when
        `print_ci_settings` is on.
    discrete_variables : list of str, default []
        Variables to treat as discrete rather than continuous.
    iqr_trim_multiple : float, default 0
        IQR multiple used to flag ASR outliers for trimming. 0 disables
        trimming.
    iqr_log_transform : bool, default True
        Compute the IQR trimming bounds on log(ASR) rather than the raw
        ratio. See :func:`ratio_study.ratio_study.ratios`.
    iqr_group_by : str or list, optional
        Column(s) to compute the IQR trim independently within. Same
        shape as `group_by`. See :func:`ratio_study.ratio_study.ratios`.
    iqr_min_cod : float, default 0.06
        Minimum CoD a sample (or group) must have for the IQR trim to
        run on it. See :func:`ratio_study.ratio_study.ratios`.
    iqr_filter_result : str or None, default 'print'
        What to do with the sales the IQR trim removed. Appendix C.6 of the
        Standard requires a report to state the basis for excluding outliers
        and identify what was removed, so the default is to say something
        rather than trim silently.

        * ``'print'`` -- print a summary: how many sales went, what share of
          the sample that is (Appendix C.5 caps it at 5-10%), and their total
          assessed and market value.
        * ``'save'`` -- also write the removed records to a ``.csv.gz`` in
          `export_directory`, then print the path and the summary.
        * a directory -- as ``'save'``, but written there.
        * a file path -- written exactly there, forced to ``.csv.gz``.
        * ``None`` / ``''`` -- silent. Use in loops and CI.

        The removed records are always available under
        ``'IQR_trimmed_sales'`` in the results dict regardless of this
        setting; the path written, if any, comes back under
        ``'IQR_trimmed_sales_export'``. No effect when ``iqr_trim_multiple``
        is 0.
    print_ci_settings : bool, default True
        Print the resolved confidence-interval settings before the study
        runs, marking any value that differs from the package default.
        Most of what governs an interval -- level, method, resamplings,
        resample size, the minimum count, the seed -- is a default nobody
        typed, so a published interval can rest on settings that were
        never reviewed. Set False in loops, or when the study produces no
        intervals.
    return_dict : bool, default True
        Passed through to `ratios` as `return_dict`.
    VE_bundle_comments : bool, default False
        If True, bundle vertical equity descriptive comments together.
    excel_sensitivity_tag : str, optional
        Sensitivity label tag written as custom document properties (e.g.
        MSIP labels) on exported Excel workbooks. Requires `site_id` to
        also be provided.
    site_id : str, optional
        Site ID associated with `excel_sensitivity_tag`. Requires
        `excel_sensitivity_tag` to also be provided.

    Returns
    -------
    DataFrame, dict or list of str
        * DataFrame with the table of all ratios -- the default, returned
          whether or not files were also written to disk.
        * dict with every ratio study object, if `return_dict_results` is
          True. Paths of everything written are under
          ``['exported_files']``: a single entry (the archive) when
          `export_in_compacted_file` is set, otherwise one entry per file.
          This is the way to get the export paths back.
        * list with a single path to a text file describing the error, if
          the study failed outright.

    Examples
    --------
    Overall 95 % BCa intervals for Median, CoD, PRD and PRB come out of
    the box; `descriptive` comments use them, so a breach of an assessment
    standard that the interval cannot confirm is no longer flagged:

    >>> ratios_report(sales, 'assessment', 'sale_price')     # doctest: +SKIP

    Turn them off when the run needs to be fast:

    >>> ratios_report(sales, 'assessment', 'sale_price',
    ...               confidence_intervals_aggs=False)            # doctest: +SKIP

    Per-group intervals too, at 90 %:

    >>> ratios_report(sales, 'assessment', 'sale_price', group_by=['nbhd'],
    ...               confidence_intervals_aggs='both',
    ...               ci_conf_level=90)                           # doctest: +SKIP
    """

    try:
        from ratio_study.reporting import _export_dataframes_as_excel_tabs, _pdf_breakdown_ratios_by_grouping, _pdf_output

    except Exception as ex:
        raise RuntimeError('Error in ratios_report:', ex)
    
    try:
        # set default values
        std_gini_args = {'run_regression':True, 'gini_decomposition':False, 'gini_factors_variables':[], 'gini_lorenz_plots':False}
        std_pdf_args = {'gen_pdf_report':True, 'pdf_ratios_table_headers':['Median', 'Mean', 'CoD', 'PRB', 'RRI', 'MKI'], 'pdf_overall_asr_graphs':True, 'pdf_group_asr_graphs':False, 'pdf_mosaic_group_asr_graphs':False
            , 'pdf_asr_graphs_x_vars_and_aliases': {'overall':[], 'groups':[], 'groups_mosaic':[]}, 'pdf_grouping_aliases':{}, 'pdf_starting_comments':[], 'pdf_ending_comments':[], 'facet_col_limit_per_page': 3, 'facet_row_limit_per_page': 4
            , 'pdf_vertical_equity_panel': True, 'pdf_vertical_equity_panel_groups': False
            , 'pdf_vertical_equity_panel_min_count': 30
            , 'pdf_asr_predictability': False, 'asr_predictability_variables': []
            , 'asr_predictability_shap_sample_pct': 0.20
            , 'asr_predictability_group_by': None}

        for key in std_gini_args.keys():
            if key not in gini_args: gini_args.update({key: std_gini_args[key]})
        for key in std_pdf_args.keys():
            if key not in pdf_args: pdf_args.update({key: std_pdf_args[key]})

        if generate_log:
            global gen_log
            gen_log = True
            logging.basicConfig(
                level=logging.WARNING
                , format='%(asctime)s %(levelname)s %(message)s'
                , datefmt='%Y-%m-%d %H:%M:%S'
                , filename=os.path.join(export_directory,'log_ratio_study.log'))

        # `ratios` defaults this too, but the header filter below reads it
        # first, so it has to be resolved here rather than left as None.
        if indicators is None:
            indicators = list(DEFAULT_INDICATORS)
        # Same for the indicators an explicitly requested distribution-mode
        # interval brings into the study: `ratios` adds them, but this filter
        # runs first and would drop the matching PDF column.
        _ci_added, _ = normalize_ci_args(
            bootstrap_args=bootstrap_args
            , ci_conf_level=ci_conf_level
            , ci_indicators=ci_indicators
            , ci_seed=ci_seed
            , ci_mode=ci_mode
            , study_indicators=indicators
            , has_groups=bool(group_by)
        )
        indicators = indicators + [
            x for x in _ci_added.get('ci_study_indicator_additions', [])
            if x not in indicators
        ]
        pdf_args['pdf_ratios_table_headers'] = [x for x in pdf_args['pdf_ratios_table_headers'] if x in indicators]

        # set date reference for exported files
        if not export_filename_date_ref:
            export_filename_date_ref = datetime.now().strftime('%d%b%Y_%Hh%M').lower()

        if export_in_compacted_file:
            if export_in_compacted_file in ['gz', 'tar.gz']:
                export_in_compacted_file = 'gzip'
            elif not export_in_compacted_file in ['zip', 'gzip']:
                export_in_compacted_file = ''

        # # replace invalid characters in model name
        # for r in ['_', ' ', '-']:
        #     model_name = model_name.replace(r, '')
        # if model_name[-3:] == '.py':
        #     model_name = model_name[:-3]

        dict_results = ratios(
            input_df=sales_df
            , assessment_header=assessment_header
            , market_value_header=market_value_header
            , mkt_value_proxy_header=mkt_value_proxy_header
            , prb_mvp_header=prb_mvp_header
            , mvp_sale_price_share=mvp_sale_price_share
            , weights_header=weights_header
            , export_directory=export_directory
            , model_name=model_name
            , indicators=indicators
            , group_by=group_by
            , log_transform=log_transform
            , gini_args=gini_args
            , bo_args=bo_args
            # The confidence-interval options are resolved once, inside
            # `ratios`, so both entry points behave identically.
            , confidence_intervals_aggs=confidence_intervals_aggs
            , ci_mode=ci_mode
            , ci_conf_level=ci_conf_level
            , ci_indicators=ci_indicators
            , ci_seed=ci_seed
            , bootstrap_args=bootstrap_args
            , descriptive=descriptive
            , vei_percentile_method=vei_percentile_method
            , vei_compare_all_buckets=vei_compare_all_buckets
            , vei_conf_level=vei_conf_level
            , vem_conf_level=vem_conf_level
            , vem_ci_method=vem_ci_method
            , assessment_standards=assessment_standards
            , graphs_args=pdf_args
            , req_significance=req_significance
            , discrete_variables=discrete_variables
            , iqr_trim_multiple=iqr_trim_multiple
            , iqr_log_transform=iqr_log_transform
            , iqr_group_by=iqr_group_by
            , iqr_min_cod=iqr_min_cod
            , iqr_filter_result=iqr_filter_result
            , print_ci_settings=print_ci_settings
            , return_dict=True
            , VE_bundle_comments=VE_bundle_comments

        )

        # replace invalid characters in model name
        model_name_rectified = model_name

        for r in ['_', ' ', '-']:
            model_name_rectified = model_name_rectified.replace(r, '')

        if model_name_rectified.endswith('.py'):
            model_name_rectified = model_name_rectified[:-3]
            
        if 'errors' in dict_results:
            if 'pdf_ending_comments' in pdf_args:
                error_messages = pdf_args['pdf_ending_comments'] + dict_results['errors'][1]
                error_messages_file = save_errors_to_text(error_messages, os.path.join(export_directory, 'errors_in_ratio_study.txt'))
            else:
                error_messages_file = dict_results['errors'][0]
            return [error_messages_file]

        if 'IQR_trimmed_sales' in dict_results and iqr_trim_multiple:
            if 'pdf_starting_comments' in pdf_args:
                pdf_args['pdf_starting_comments'].append('Note on outliers trimming: ' + str(int(len(dict_results['IQR_trimmed_sales']))) + ' out of '+ str(int(len(sales_df))) + ' sales were trimmed from the sales file as extreme outliers, due to Sales Ratio being outside of ' + str(iqr_trim_multiple) + ' times the IQR range.')

        # A data export that fails is a partial loss, not a total one: the PDF
        # is built further down from results already in hand, so each export is
        # isolated and its failure recorded rather than raised. Previously one
        # bad Excel write aborted the whole report and no PDF was produced.
        if export_ratios == 'csv' or export_ratios == 'both':
            csv_grouped_ratios_filepath = os.path.join(export_directory, '_'.join(['ratio_study', model_name_rectified, export_filename_date_ref]) + '.csv')
            try:
                dict_results['df_all_ratios_formatted'].to_csv(csv_grouped_ratios_filepath, index=None, encoding='utf-16')
                dict_results['exported_files'].append(csv_grouped_ratios_filepath)
            except Exception as ex:
                error_messages.append('error in exporting csv: ' + csv_grouped_ratios_filepath + ': ' + str(ex))
                if gen_log:
                    logging.warning('error in exporting csv: %s: %s', csv_grouped_ratios_filepath, ex)

        if export_ratios == 'xlsx' or export_ratios == 'both':
            excel_grouped_ratios_filepath = os.path.join(export_directory, '_'.join(['ratio_study', model_name_rectified, export_filename_date_ref]) + '.xlsx')
            try:
                grouped_ratios_export = _export_dataframes_as_excel_tabs(dict_of_dataframes={'Ratios': {'dataframe': dict_results['df_all_ratios_formatted']}}
                                                                         , filepath=''
                                                                         , excel_filename=excel_grouped_ratios_filepath
                                                                         , sensitivity_tag=excel_sensitivity_tag
                                                                         , site_id=site_id)

                dict_results['exported_files'].append(excel_grouped_ratios_filepath)
            except Exception as ex:
                error_messages.append('error in exporting xlsx: ' + excel_grouped_ratios_filepath + ': ' + str(ex))
                if gen_log:
                    logging.warning('error in exporting xlsx: %s: %s', excel_grouped_ratios_filepath, ex)
        # export gini decomposition:
        if gini_args['gini_decomposition'] or len(gini_args['gini_factors_variables']) > 0:
            export_gini_decomp_filepath = os.path.join(export_directory, '_'.join(['gini_decomposition', model_name_rectified, export_filename_date_ref]) + '.xlsx')
            try:
                gini_decomp_export = _export_dataframes_as_excel_tabs(dict_results['dict_gini_decomposition']
                                                                      , ''
                                                                      , export_gini_decomp_filepath
                                                                      , sensitivity_tag=excel_sensitivity_tag
                                                                      , site_id=site_id)

                dict_results['exported_files'].append(export_gini_decomp_filepath)
            except Exception as ex:
                error_messages.append('error in exporting Gini decomposition: ' + export_gini_decomp_filepath + ': ' + str(ex))
                if gen_log:
                    logging.warning('error in exporting Gini decomposition: %s: %s', export_gini_decomp_filepath, ex)

        # export Blinder Oaxaca decomposition:
        if bo_args:
            dict_of_dataframes_bo = {}
            if 'bo_formatted' in dict_results:
                dict_of_dataframes_bo['Blinder-Oaxaca'] = {
                    'dataframe': dict_results['bo_formatted']
                }
            if 'bo_multi' in dict_results:
                dict_of_dataframes_bo['Blinder-Oaxaca Summary'] = {
                    'dataframe': dict_results['bo_multi']['summary_table']
                }
            if dict_of_dataframes_bo:
                export_bo_decomp_filepath = os.path.join(export_directory, '_'.join(['Blinder_Oaxaca_Decomposition', model_name_rectified, export_filename_date_ref]) + '.xlsx')
                try:
                    bo_decomp_export = _export_dataframes_as_excel_tabs(dict_of_dataframes_bo
                                                                        , ''
                                                                        , export_bo_decomp_filepath
                                                                        , sensitivity_tag=excel_sensitivity_tag
                                                                        , site_id=site_id)

                    dict_results['exported_files'].append(export_bo_decomp_filepath)
                except Exception as ex:
                    error_messages.append('error in exporting Blinder-Oaxaca decomposition: ' + export_bo_decomp_filepath + ': ' + str(ex))
                    if gen_log:
                        logging.warning('error in exporting Blinder-Oaxaca decomposition: %s: %s', export_bo_decomp_filepath, ex)

        if pdf_args['gen_pdf_report']:
            export_pdf_filepath = os.path.join(export_directory, '_'.join(['ratio_study', model_name_rectified, export_filename_date_ref]) + '.pdf')
            # if os.path.exists(export_pdf_filepath):
            #     os.remove(export_pdf_filepath)
            # `df_all_ratios_formatted` already blanks anything not computed.
            # A printed table reads better with an explicit dash than with an
            # empty cell, so mark them here -- for the PDF only, leaving the
            # returned frame blank as callers expect.
            df_ratios = dict_results['df_all_ratios_formatted'].copy()
            df_ratios = df_ratios.map(
                lambda cell: '-' if (isinstance(cell, str) and not cell.strip())
                else ('-' if pd.isna(cell) else cell))
            # split ratios df into groupings
            try:
                ratio_tables_dfs = _pdf_breakdown_ratios_by_grouping(df_ratios, output_columns=pdf_args['pdf_ratios_table_headers'])
                _pdf_output(export_pdf_filepath=export_pdf_filepath
                           , ratio_tables_dfs=ratio_tables_dfs
                           , model_name=model_name
                           , base_date=base_date
                           , overall_graphs=dict_results['overall_graphs']
                           , group_graphs=dict_results['group_graphs']
                           , group_mosaic_graphs=dict_results['group_mosaic_graphs']
                           , dist_group_mosaic_graphs=dict_results['dist_plot_group_mosaic_graphs']
                           , pdf_grouping_aliases=pdf_args['pdf_grouping_aliases']
                           , gini_lorenz_plots=dict_results['gini_lorenz_plots'] if gini_args['gini_lorenz_plots'] else []
                           , ending_comments=pdf_args['pdf_ending_comments']
                           , starting_comments=pdf_args['pdf_starting_comments']
                           , classification_header = pdf_args['classification_header'] if 'classification_header' in pdf_args else None
                           )
                dict_results['exported_files'].append(export_pdf_filepath)
            except Exception as ex:
                raise RuntimeError('error in exporting pdf:' + export_pdf_filepath, ex)
        # Option to put all files in one compacted file
        if dict_results['exported_files']:
            if export_in_compacted_file:
                if export_in_compacted_file == 'gzip':
                    export_compacted_filepath = os.path.join(export_directory, '_'.join(['ratios', model_name_rectified, export_filename_date_ref]) + '.gz')
                    with tarfile.open(export_compacted_filepath, 'w:gz') as f:
                        for filepath in dict_results['exported_files']:
                            f.add(filepath, arcname=os.path.basename(filepath))
                            os.remove(filepath)
                    dict_results['exported_files'] = [export_compacted_filepath]
                # Option to put all files in one zip compacted file
                elif export_in_compacted_file == 'zip':
                    export_compacted_filepath = os.path.join(export_directory, '_'.join(['ratios', model_name_rectified, export_filename_date_ref]) + '.zip')
                    with zipfile.ZipFile(export_compacted_filepath, mode="w", compression=zipfile.ZIP_DEFLATED) as f:
                        for filepath in dict_results['exported_files']:
                            f.write(filepath, os.path.basename(filepath))
                            os.remove(filepath)
                    dict_results['exported_files'] = [export_compacted_filepath]
        # Two returns, not four. The branches that used to follow returned the
        # export paths, but `if return_dict_results == False` caught the
        # default first and they were unreachable -- so the documented
        # "returns the path to the archive" never happened and callers got a
        # DataFrame where they expected a filename. Rather than change what
        # the default call returns, the paths are published where they can be
        # relied on: `return_dict_results=True` -> ``['exported_files']``.
        if return_dict_results:
            return dict_results
        return dict_results['df_all_ratios_formatted']
    except Exception as x:
        return [save_errors_to_text(['error in ratio_study, please contact the maintainer', str(x)], os.path.join(export_directory, 'errors_in_ratio_study.txt'))]
    
# ---------------------------------------------------------------------------------------------------------------------

def _bootstrap_stacker(input_df
                      , assessment_header
                      , market_value_header
                      , weights_header
                      , bootstrap_resamplings
                      , req_significance
                      , indicators
                      , resample_pct=1.0
                      , vei_conf_level=0.9
                      , vei_percentile_method='R7'
                      , vei_use_log_normal_delta=True
                      , concurrency=None
                      , log_transform=False
                      , seed=None):
    """
    Resample sales and assessments with replacement, and calculate ratio
    study indicators for each resample.

    Parameters
    ----------
    input_df : DataFrame
        Pandas dataframe with sales, assessments, and any additional
        columns needed for the selected indicators.
    assessment_header : str
        Column name in `input_df` for the assessed / predicted value.
    market_value_header : str
        Column name in `input_df` for the market value / sale amount.
    weights_header : str, optional
        Column name in `input_df` for sample weights, assumed to have
        mean 1.
    bootstrap_resamplings : int
        Number of bootstrap resamplings.
    req_significance : float
        p-value threshold for the PRB.
    indicators : list of str
        Indicators to compute.
    vei_conf_level : float, default 0.9
        Confidence level for VEI analysis.
    vei_percentile_method : str, default 'R7'
        Percentile interpolation method used in VEI calculations.
    vei_use_log_normal_delta : bool, default True
        If True, use the log-normal delta method for VEI confidence
        intervals.
    concurrency : str or None, default None
        'processes', 'threads', or None. If None, a single-threaded
        process is used. In some cases, multiple processes or threads
        could be beneficial, especially when the number of resamples is
        high and/or a large number of groupings is being used in
        bootstrapping. Timed testing of each possibility is encouraged.
        Note that 'processes' can only be used in the main script.
    log_transform : bool, default False
        If True, `market_value_header`/`assessment_header` values are
        assumed to be in log scale and are exponentiated before
        resampling.
    seed : int, optional
        Seed for the resampler. None draws fresh entropy, so the
        resulting confidence intervals differ slightly between two
        otherwise identical runs; pass a seed to make them reproducible.

    Returns
    -------
    DataFrame
        Pandas dataframe of stacked bootstrapping results, one row per
        resample.
    """
    try:
        bootstrap_results = []
        n = len(input_df.index)
        # Draw size per resample. Below 100% the classical bootstrap becomes an
        # m-out-of-n bootstrap; the floor keeps each draw large enough for the
        # indicators to mean anything, and never exceeds the sample itself.
        draw = n if resample_pct >= 1 else min(n, max(BOOTSTRAP_MIN_RESAMPLE_SIZE,
                                                      int(round(n * resample_pct))))
        samples = []

        if log_transform:
            input_df[market_value_header] = np.exp(input_df[market_value_header])
            input_df[assessment_header] = np.exp(input_df[assessment_header])

        input_df.loc[:, 'bt_asr'] = input_df[assessment_header] / input_df[market_value_header]
        if any(item in indicators for item in ['MdAPE', 'MAPE', 'FSD']):
            input_df.loc[:, 'bt_sale_error'] = (input_df.loc[:, assessment_header] - input_df.loc[:, market_value_header]) / input_df.loc[:, market_value_header]
        
        rng = np.random.default_rng(seed)
        for i in range(bootstrap_resamplings):
            if concurrency == 'processes':
                # Lazy import
                from scipy.stats import randint
                try:
                    indices_b = randint.rvs(low=0, high=n, size=draw, random_state=rng)
                except:
                    if concurrency=='processes': concurrency='threads'
                    indices_b = list(rng.integers(low=0, high=n, size=draw))
            else:
                indices_b = list(rng.integers(low=0, high=n, size=draw))

            # sample_df = input_df.iloc[indices_b].reset_index(drop=True)
            sample_df = input_df.iloc[indices_b]
            input_dict = {'market_value_array': np.array(sample_df[market_value_header])
                        , 'assessment_array': np.array(sample_df[assessment_header])
                        , 'sale_count': draw
                        , 'asr_array': np.array(sample_df['bt_asr'])
                        , 'sale_error_array': np.array(sample_df['bt_sale_error']) if any(item in indicators for item in ['MdAPE', 'MAPE', 'FSD']) else None
                        , 'weights_array': np.array(sample_df[weights_header]) if weights_header else None
                        , 'req_significance': req_significance
                        , 'indicators': indicators
                        , 'vei_conf_level': vei_conf_level
                        , 'vei_percentile_method': vei_percentile_method
                        , 'vei_plot_ref': ''
                        , 'vei_use_log_normal_delta': vei_use_log_normal_delta
                        , 'gini_lorenz_plots': False
                        , 'gini_plot_ref': ''
                        , 'vei_plot': False}

            if concurrency is not None:
                samples.append(input_dict)
            else:
                bootstrap_results.append(_ratios_calculator_wrap(input_dict))

        if concurrency == 'processes' and __name__ == '__main__':
            # Lazy import
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor() as executor:
                bootstrap_results = list(executor.map(_ratios_calculator_wrap, samples, chunksize=20))

        elif concurrency == 'threads' or (concurrency == 'processes' and __name__ != '__main__'):
            # Lazy import
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                bootstrap_results = list(executor.map(_ratios_calculator_wrap, samples, chunksize=20))
        return pd.DataFrame(bootstrap_results)
    except Exception as ex:
        raise RuntimeError('error in ratio study: _bootstrap_stacker', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _bootstrap_analysis(input_df
                       , assessment_header
                       , market_value_header
                       , weights_header
                       , confint_level
                       , bootstrap_method
                       , bootstrap_resamplings
                       , population_ratios
                       , ci_only
                       , req_significance
                       , indicators
                       , vei_conf_level=0.9
                       , vei_use_log_normal_delta=True
                       , interpolation_method='median_unbiased'
                       , concurrency=None
                       , resample_pct=1.0
                       , seed=None):
    """
    Run bootstrapped ratio studies a pre-specified number of times, and
    from the resulting resamples calculate confidence intervals and,
    optionally, mean, median, standard deviation, kurtosis, and skewness.

    Parameters
    ----------
    input_df : DataFrame
        Pandas dataframe with sales, assessments, and any additional
        columns needed for the selected indicators.
    assessment_header : str
        Column name in `input_df` for the assessed / predicted value.
    market_value_header : str
        Column name in `input_df` for the market value / sale amount.
    weights_header : str, optional
        Column name in `input_df` for sample weights, assumed to have
        mean 1.
    confint_level : float
        Confidence level for confidence interval calculations
        (e.g. 0.95 for 95%).
    bootstrap_method : str
        Method used for the confidence intervals: 'BCa' (default in
        callers) or 'percentile'.
    bootstrap_resamplings : int
        Number of resamplings.
    population_ratios : dict
        Ratios for the entire population, used as the point estimate for
        BCa confidence intervals.
    ci_only : bool
        If False, mean, median, standard deviation, kurtosis and skewness
        are also calculated on the bootstrap samples, in addition to the
        confidence intervals.
    req_significance : float
        Required significance level (p-value) for the PRB
        (e.g. 0.05 for 5%).
    indicators : list of str
        Indicators to compute.
    vei_conf_level : float, default 0.9
        Confidence level for VEI analysis.
    vei_use_log_normal_delta : bool, default True
        Passed through to `_bootstrap_stacker` for each resample's VEI
        analysis.
    interpolation_method : str, default 'median_unbiased'
        Interpolation method used when computing percentiles for the
        confidence intervals.
    concurrency : str or None, default None
        'processes', 'threads', or None. Passed through to
        `_bootstrap_stacker` to control how resamples are computed. If
        resampling with the requested concurrency fails, it falls back
        to single-threaded execution.
    seed : int, optional
        Seed for the resampler, passed through to `_bootstrap_stacker`.
        None draws fresh entropy, so two identical runs publish slightly
        different intervals.

    Returns
    -------
    dict
        Dictionary keyed by indicator name (matching the columns produced
        by `_bootstrap_stacker`), with `<indicator>_ci` holding the
        confidence interval and, when `ci_only` is False,
        `<indicator>_btst_md`, `_btst_mn`, `_btst_std`, `_btst_krt`, and
        `_btst_skw` holding the bootstrap median, mean, standard
        deviation, kurtosis and skewness.
    """
    try:
        # Lazy import
        from scipy.stats import skew, kurtosis
        result = {}
        if concurrency is not None:
            try:
                bootstrap_results = _bootstrap_stacker(input_df=input_df.copy()
                                                  , assessment_header=assessment_header
                                                  , market_value_header=market_value_header                                                  
                                                  , weights_header=weights_header
                                                  , bootstrap_resamplings=bootstrap_resamplings
                                                    , resample_pct=resample_pct
                                                  , req_significance=req_significance
                                                  , vei_conf_level=vei_conf_level
                                                  , vei_use_log_normal_delta=vei_use_log_normal_delta
                                                  , indicators=indicators
                                                  , concurrency=concurrency
                                                  , seed=seed)

            except: # if there are incompatibilities with multithreading or multiprocessing
                concurrency=None
        if concurrency is None:
            bootstrap_results = _bootstrap_stacker(input_df=input_df.copy()
                                                  , assessment_header=assessment_header
                                                  , market_value_header=market_value_header
                                                  , weights_header=weights_header
                                                  , bootstrap_resamplings=bootstrap_resamplings
                                                    , resample_pct=resample_pct
                                                  , req_significance=req_significance
                                                  , vei_conf_level=vei_conf_level
                                                  , vei_use_log_normal_delta=vei_use_log_normal_delta
                                                  , indicators=indicators
                                                  , concurrency=None
                                                  , seed=seed)

        for col in bootstrap_results.columns:
            series = bootstrap_results[col].copy()
            series.sort_values(inplace=True)
            series.reset_index(inplace=True, drop=True)
            if ci_only == False and 'Pvalue' not in col:
                series_median = series.median()
                series_mean = series.mean()
                series_std = series.std()
                series_kurtosis = kurtosis(series, axis=0, bias=True)
                series_skewness = skew(series, axis=0, bias=True)

                if abs(series_median) > 1000:
                    rounding = 0
                elif abs(series_median) > 10:
                    rounding = 2
                elif abs(series_median) > 0.1:
                    rounding = 5
                else:
                    rounding = 7
                result[col + '_btst_md'] = round(series_median, rounding)
                result[col + '_btst_mn'] = round(series_mean, rounding)
                result[col + '_btst_std'] = round(series_std, rounding)
                result[col + '_btst_krt'] = round(series_kurtosis, 6)
                result[col + '_btst_skw'] = round(series_skewness, 6)

            if bootstrap_method == 'percentile':
                # Lazy import
                from ratio_study.confint import _percentile_ci
                result[col + '_ci'] = _percentile_ci(series, confint_level, interpolation_method=interpolation_method)

            elif bootstrap_method == 'BCa':
                if col not in population_ratios:
                    if gen_log:
                        logging.warning('Could not find column ' + col + ' in population ratios, as a result BCa confidence intervals for' + col + 'could not be calculated.')
                else:
                    # An insignificant PRB arrives as a marked string such as
                    # '-0.05896*'; the sign has to survive the parse or BCa
                    # bias-corrects against a point estimate of the wrong sign.
                    population_estimate = float(re.findall(r'-?\d+\.?\d*', population_ratios[col])[0]) if isinstance(population_ratios[col], str) else population_ratios[col]
                    # Lazy import
                    from ratio_study.confint import _bca_ci
                    result[col + '_ci'] = _bca_ci(series, population_estimate, confint_level, interpolation_method=interpolation_method)
        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: bootstrap_analysis', ex)


