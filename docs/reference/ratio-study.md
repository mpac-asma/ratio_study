# Core study

The two entry points most studies need: `ratios` and `ratios_report`.

*Module:* `ratio_study.ratio_study`

## Contents

- [`ratios`](#ratios) — Compute a ratio study (assessed-to-market-value ratios) with optional
- [`ratios_report`](#ratios-report) — Run a ratio study on model results, with the option of generating
- [`rts_calculator`](#rts-calculator) — Calculate ratio study indicators (Median, CoD, PRD, PRB, VEI, etc.)

---

## `ratios`

```python
ratios(input_df, assessment_header, market_value_header, mkt_value_proxy_header=None,
       prb_mvp_header=None, mvp_sale_price_share=0.5, group_by=None, log_transform=False,
       weights_header=None, export_directory=Path.cwd(), descriptive=False,
       dict_results=None, model_name='model', indicators=None, gini_args=None,
       bo_args=None, confidence_intervals_aggs=None, ci_mode='distribution',
       ci_conf_level=None, ci_indicators=None, ci_seed=None, bootstrap_args=None,
       print_ci_settings=True, vei_percentile_method='R7', vei_compare_all_buckets=False,
       vei_conf_level=0.9, vei_plot=False, vei_plot_ref='',
       vei_use_log_normal_delta=True, ve_plot_ref='', vem_acceptable_inequity_level=0.1,
       vem_conf_level=0.9, vem_ci_method='binom', vem_plot=False,
       vem_title='Vertical Equity Moving Median', round_values=True,
       req_significance=0.05, assessment_standards=None, graphs_args=None,
       discrete_variables=None, iqr_trim_multiple=0, iqr_log_transform=True,
       iqr_group_by=None, iqr_min_cod=0.06, iqr_filter_result='print', return_dict=False,
       VE_bundle_comments=False)
```

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

---

## `ratios_report`

```python
ratios_report(sales_df, assessment_header, market_value_header,
              mkt_value_proxy_header=None, prb_mvp_header=None, mvp_sale_price_share=0.5,
              group_by=[], export_directory=Path.cwd(), return_dict_results=False,
              weights_header=None, descriptive=True, indicators=None, model_name='model',
              export_filename_date_ref='', gini_args={}, bo_args={},
              confidence_intervals_aggs=None, ci_mode='distribution', ci_conf_level=None,
              ci_indicators=None, ci_seed=None, bootstrap_args={},
              assessment_standards={}, export_ratios='', export_in_compacted_file='zip',
              pdf_args={}, generate_log=False, log_transform=False, base_date='',
              req_significance=0.05, vei_percentile_method='R7',
              vei_compare_all_buckets=False, vei_conf_level=0.9, vem_conf_level=0.9,
              vem_ci_method='binom', discrete_variables=[], iqr_trim_multiple=0,
              iqr_log_transform=True, iqr_group_by=None, iqr_min_cod=0.06,
              iqr_filter_result='print', print_ci_settings=True, return_dict=True,
              VE_bundle_comments=False, excel_sensitivity_tag=None, site_id=None)
```

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

---

## `rts_calculator`

```python
rts_calculator(input_df=None, sale_count=None, assessment_header=None,
               market_value_header=None, mkt_value_proxy_header=None,
               prb_mvp_header=None, mvp_sale_price_share=0.5, market_value_array=None,
               assessment_array=None, weights_header=None, weights_array=None,
               asr_array=None, sale_error_array=None,
               indicators=['Median', 'Mean', 'Weighted_Mean', 'CoD', 'PRD', 'PRB', 'VEI', 'VEM'],
               req_significance=0.05, vei_conf_level=0.9, vei_percentile_method='R7',
               vei_compare_all_buckets=False, vei_plot=False, vei_plot_ref='',
               vei_use_log_normal_delta=True, vem_acceptable_inequity_level=0.05,
               vem_conf_level=0.9, vem_ci_method='binom', vem_plot=False,
               vem_title='Vertical Equity Moving Median', ve_plot_ref='',
               ci_conf_level=None, gini_lorenz_plots=False, gini_plot_ref='',
               export_directory=Path.cwd(), run_regression=True, skip_fsd=False,
               skip_count=False, for_bootstrapping=False, prb_non_significant_marker='*',
               log_transform=False)
```

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
