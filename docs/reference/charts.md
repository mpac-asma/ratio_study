# Charts

Every chart the package draws, and the Vertical Equity Panel.

*Module:* `ratio_study.graphs`

## Contents

- [`asr_boxplot`](#asr-boxplot) — One ratio box per category, for horizontal equity across strata.
- [`asr_facet`](#asr-facet) — A small-multiples grid of ratio scatterplots, one panel per stratum.
- [`asr_histogram`](#asr-histogram) — Histogram of the ratio distribution, with the median and the band marked.
- [`asr_scatter`](#asr-scatter) — Scatter the ratio against any variable, with a lowess trend.
- [`lorenz_plot`](#lorenz-plot) — Lorenz and concentration curves, straight from the sales frame.
- [`vei_boxplot`](#vei-boxplot) — Boxplot of the Vertical Equity Indicator (VEI) by value quantile group.
- [`vem_plot`](#vem-plot) — The VEM moving-median chart, straight from the sales frame.
- [`vertical_equity_panel`](#vertical-equity-panel) — Build the Vertical Equity Panel: four indicator gauges over two charts.

---

## `asr_boxplot`

```python
asr_boxplot(df, assessment_header=None, market_value_header=None, by=None, *,
            asr_col=None, min_count=5, max_groups=25, order_by='group', bands=(0.9, 1.1),
            show_values=False, value_fmt='{:.3f}', title=None, x_label=None,
            figsize=(11.0, 6.0), return_object=False, graph_filepath='', show=None)
```

One ratio box per category, for horizontal equity across strata.

This is the visual form of a grouped ratio table: boxes sitting at
different heights are differing levels of valuation between strata,
which the Standard asks to be within 5 percent of the overall level.
Boxes of different heights are differing uniformity -- the CoD story.

Parameters
----------
df : DataFrame
    Sales frame.
assessment_header, market_value_header : str, optional
    Value columns, unless `asr_col` is given.
by : str
    Categorical column to split on. Required.
asr_col : str, optional
    Existing ratio column.
min_count : int, default 5
    Drop categories with fewer sales than this; a box built on three
    sales invites a conclusion the sample cannot support.
max_groups : int, default 25
    Keep only the largest categories, so a high-cardinality column
    does not produce an unreadable chart. The number dropped is noted.
order_by : {'group', 'median', 'count'}, default 'group'
    Box order: by category value, by median ratio (most useful for
    spotting a level gradient), or by sales count.
bands : tuple of (float, float) or None, default (0.90, 1.10)
    Acceptable level-of-valuation band.
title : str, optional
    Chart title.
x_label : str, optional
    X axis label; defaults to a tidied `by`.
figsize : tuple, default (11.0, 6.0)
    Figure size in inches.
return_object, graph_filepath, show
    Standard three-way output; see `_finish_chart`.

Returns
-------
matplotlib.figure.Figure, str or list
    Per the standard three-way output.

---

## `asr_facet`

```python
asr_facet(df, assessment_header=None, market_value_header=None, x=None, *, col=None,
          row=None, asr_col=None, col_wrap=3, max_facets=12, tile_min_records=30,
          log_x=False, lowess=True, zoom=True, bands=None, share_y=True, height=3.1,
          aspect=1.25, title=None, x_label=None, return_object=False, graph_filepath='',
          show=None)
```

A small-multiples grid of ratio scatterplots, one panel per stratum.

An overall scatter can look flat while individual neighbourhoods each
slope hard in opposite directions. Faceting is how that shows up: the
same axes repeated across strata, so panels can be compared by eye
against a shared reference line.

Parameters
----------
df : DataFrame
    Sales frame.
assessment_header, market_value_header : str, optional
    Value columns, unless `asr_col` is given.
x : str, optional
    Variable on the x axis of every panel. Defaults to
    `market_value_header`.
col : str
    Column faceted across panels. Required.
row : str, optional
    Second facet dimension, laid out down the grid. Ignored by
    `col_wrap` when given.
asr_col : str, optional
    Existing ratio column.
col_wrap : int, default 3
    Panels per row when `row` is not used.
max_facets : int, default 12
    Cap on panels, keeping the largest strata. The number dropped is
    reported in the subtitle rather than silently discarded.
tile_min_records : int, default 30
    Smallest sample a panel may be drawn from. Panels below it are
    dropped and named in a printed note, so a handful of sales does
    not get a panel of its own and read as a trend. ``0`` draws every
    panel, however thin.
log_x : bool, default False
    Log-scale the x axis.
lowess : bool, default True
    Overlay a lowess trend in each panel.
zoom : bool or str, default True
    Trim the axes to the bulk of the data instead of its full range.
    ``True`` measures each panel separately and centres it on its own
    median price and median ratio, so a cheap stratum is not squeezed
    into the left inch of an axis stretched by an expensive one.
    ``'overall'`` is the older single window measured across the whole
    frame and shared by every panel. ``False`` leaves the axes alone.
bands : tuple of (float, float) or None, default None
    Acceptable level-of-valuation band, drawn in every panel. Off by
    default here, unlike the full-size charts: panels this small get
    crowded by two more lines. Pass ``(0.90, 1.10)`` for it. With
    per-panel zoom a band edge far from a panel's own median can fall
    outside that panel.
share_y : bool, default True
    Give every panel the same y axis. Under ``zoom='overall'`` or
    ``zoom=False`` that is one shared axis, so levels can be compared
    by eye. Under per-panel zoom the panels are centred separately, so
    this shares the *span* -- the axis is the same number of ratio
    points tall everywhere, which keeps spread and slope comparable
    while each panel stays readable. Turn it off to let each panel
    size its own y axis.
height, aspect : float
    Panel height in inches and width/height ratio, as in seaborn.
title : str, optional
    Figure title.
x_label : str, optional
    Shared x axis label.
return_object, graph_filepath, show
    Standard three-way output; see `_finish_chart`.

Returns
-------
matplotlib.figure.Figure, str or list
    Per the standard three-way output.

---

## `asr_histogram`

```python
asr_histogram(df, assessment_header=None, market_value_header=None, *, asr_col=None,
              bins=60, clip=(0.0, 2.0), bands=(0.9, 1.1), title=None,
              x_label='Assessment-to-sale ratio', figsize=(9.5, 5.5),
              return_object=False, graph_filepath='', show=None)
```

Histogram of the ratio distribution, with the median and the band marked.

The shape matters as much as the summary statistics: the Standard asks
the analyst to examine the distribution before trusting any measure
that assumes normality, and a bimodal or heavily skewed histogram is
the fastest way to see that two populations have been mixed, or that
outliers are driving the mean.

Parameters
----------
df : DataFrame
    Sales frame.
assessment_header, market_value_header : str, optional
    Value columns, unless `asr_col` is given.
asr_col : str, optional
    Existing ratio column.
bins : int, default 60
    Histogram bins.
clip : tuple of (float, float) or None, default (0.0, 2.0)
    Restrict the drawn range so a handful of extreme ratios cannot
    compress the bulk of the distribution into one bar. The count of
    excluded sales is noted on the chart. None draws everything.
bands : tuple of (float, float) or None, default (0.90, 1.10)
    Acceptable level-of-valuation band.
title : str, optional
    Chart title.
x_label : str, default 'Assessment-to-sale ratio'
    X axis label.
figsize : tuple, default (9.5, 5.5)
    Figure size in inches.
return_object, graph_filepath, show
    Standard three-way output; see `_finish_chart`.

Returns
-------
matplotlib.figure.Figure, str or list
    Per the standard three-way output.

---

## `asr_scatter`

```python
asr_scatter(df, assessment_header=None, market_value_header=None, x=None, *,
            asr_col=None, x_label=None, y_label='Assessment-to-sale ratio', log_x=False,
            lowess=True, zoom=True, bands=None, point_size=8, alpha=0.35, marginals=True,
            marginal_bins=120, title=None, figsize=(9.5, 6.5), return_object=False,
            graph_filepath='', show=None)
```

Scatter the ratio against any variable, with a lowess trend.

The workhorse diagnostic: a flat cloud is a model behaving, and a
visible slope is the model's ratio moving with whatever is on the x
axis. Plotting against the sale price is the graphical companion to
PRB and the VEI; plotting against living area, age or quality shows
whether a characteristic the model already uses is still carrying
bias.

Parameters
----------
df : DataFrame
    Sales frame (pandas, polars, pyarrow or narwhals).
assessment_header, market_value_header : str, optional
    Assessed value and (time-adjusted) sale price. Optional only if
    `asr_col` names an existing ratio column.
x : str, optional
    Variable for the x axis. Defaults to `market_value_header`, the
    vertical-equity view.
asr_col : str, optional
    Use this existing ratio column instead of deriving one.
x_label, y_label : str, optional
    Axis labels. `x_label` defaults to a tidied `x`.
log_x : bool, default False
    Log-scale the x axis. Worth turning on for prices, whose
    distribution is long-tailed enough to squash most of the sample
    into the left margin.
lowess : bool, default True
    Overlay a lowess trend line. This is the part to read: it follows
    curvature that a straight-line PRB regression cannot report.
bands : tuple of (float, float) or None, default (0.90, 1.10)
    Acceptable level-of-valuation band. None to omit.
point_size : float, default 8
    Marker size.
alpha : float, default 0.35
    Marker opacity, low by default so a dense cloud stays readable.
marginals : bool, default True
    Add marginal histograms for `x` and the ratio.
title : str, optional
    Chart title. A descriptive default is built when omitted.
figsize : tuple, default (9.5, 6.5)
    Figure size in inches.
return_object, graph_filepath, show
    Standard three-way output; see `_finish_chart`.

Returns
-------
matplotlib.figure.Figure, str or list
    Per the standard three-way output.

Examples
--------
>>> asr_scatter(sales, 'assessment', 'sale_price', log_x=True)   # doctest: +SKIP
>>> asr_scatter(sales, 'assessment', 'sale_price', x='living_area')   # doctest: +SKIP

---

## `lorenz_plot`

```python
lorenz_plot(df, assessment_header=None, market_value_header=None, *, weights_header=None,
            title=None, graph_filepath='', show=None)
```

Lorenz and concentration curves, straight from the sales frame.

Equivalent to running :func:`ratio_study.gini_stats` and passing its
Gini and concentration figures into
:func:`ratio_study.gini_lorenz_plot`, without the two-step.

Parameters
----------
df : DataFrame
    Sales frame.
assessment_header, market_value_header : str
    Assessed value and (time-adjusted) sale price columns.
weights_header : str, optional
    Observation weights.
title : str, optional
    Chart title.
graph_filepath : str, default ''
    Save a PNG here and return the path.
show : bool, optional
    Force display on or off.

Returns
-------
matplotlib.figure.Figure or str
    The figure, or the saved path when `graph_filepath` is given.

---

## `vei_boxplot`

```python
vei_boxplot(asr, mkt_value_proxy=None, vei_result=None, short_labels=True,
            graph_filepath='', title='', return_object=False, *, assessment_header=None,
            market_value_header=None, conf_level=0.9, acceptable_range=10,
            percentile_method='R7')
```

Boxplot of the Vertical Equity Indicator (VEI) by value quantile group.

The graphical form of Appendix E of the Standard on Ratio Studies: one
box per quantile group of the market value proxy, so the median ratio
of the lowest group can be read against the highest -- which is what
the VEI measures.

Two ways to call it:

* **Frame first** (preferred) -- pass the sales frame and the two value
  columns, and the VEI analysis is run internally::

      vei_boxplot(sales, assessment_header='av', market_value_header='sp')

* **Arrays** -- pass the ratio, the proxy, and a `vei_analysis` result
  you already have. This is the original signature, kept for the report
  pipeline and existing callers.

Parameters
----------
asr : DataFrame or array-like
    Either the sales frame (with `assessment_header` /
    `market_value_header` given), or an array of ratios.
mkt_value_proxy : array-like, optional
    Market value proxy, same length as `asr`. Only for the array form.
vei_result : dict, optional
    Result of ``vei_analysis``. Only for the array form; computed
    internally in the frame form.
assessment_header, market_value_header : str, optional
    Value columns, for the frame form.
conf_level : float, default 0.9
    Confidence level for the internally computed VEI. Frame form only.
acceptable_range : float, default 10
    VEI acceptable range in percent, per the Standard. Frame form only.
percentile_method : {'R7', 'R6'}, default 'R7'
    Quantile-ranking method (Appendix E.5). Frame form only.
short_labels : bool, default True
    If True, abbreviate the median proxy value tick labels (e.g.
    ``'$1.23M'``, ``'$45K'``). If False, use full comma-formatted
    dollar amounts. Either way, each tick carries the group's CoD on
    a second line.
graph_filepath : str, default ''
    Filepath location to save the plot. If empty, the plot is not
    saved to disk.
title : str, default ''
    Plot title. If empty, defaults to 'Vertical Equity Analysis'.
return_object : bool, default False
    If True, return ``[title, image_obj]`` instead of saving/showing
    the plot.

Returns
-------
list or str or matplotlib.figure.Figure
    * If `return_object` is True, a list ``[title, image_obj]`` where
      `image_obj` is a ReportLab ``Image``.
    * If `graph_filepath` is provided, the filepath (str) where the
      plot was saved.
    * Otherwise, the current matplotlib ``Figure``, after displaying it.

---

## `vem_plot`

```python
vem_plot(df, assessment_header=None, market_value_header=None, *, asr_col=None,
         quantiles=None, window_size=None, conf_level=0.9, ci_method='binom',
         acceptable_inequity_level=0.1, x_scale='log',
         title='Vertical Equity Moving Median', plot_ref=None, graph_filepath=None)
```

The VEM moving-median chart, straight from the sales frame.

Equivalent to calling :func:`ratio_study.vem` with ``generate_plot=True``
and pulling ``result['plot']`` out, but without needing an ``'asr'``
column in the frame first.

Where the VEI compares the top and bottom quantile groups, this tracks
the local median ratio across the whole value range with a confidence
band, so an inequity confined to the middle of the range -- invisible
to a two-group comparison and to a straight-line PRB -- shows up.

Parameters
----------
df : DataFrame
    Sales frame.
assessment_header, market_value_header : str, optional
    Value columns, unless `asr_col` is given.
asr_col : str, optional
    Existing ratio column.
quantiles : int, optional
    Number of quantile groups. Takes precedence over `window_size`.
window_size : int, optional
    Rolling-window size used when `quantiles` is None.
conf_level : float, default 0.90
    Level for the band around the local median. None omits the band --
    and with it the verdict, which is tested against the band.
ci_method : {'binom', 'R6', 'R7'}, default 'binom'
    Median-interval family for the band.
acceptable_inequity_level : float, default 0.10
    Distance from the overall median that counts as inequity.
x_scale : {'log', 'linear'}, default 'log'
    Value-axis scale.
title : str, default 'Vertical Equity Moving Median'
    Chart title.
plot_ref : str, optional
    Reference label appended to the title.
graph_filepath : str, optional
    Save here instead of returning the figure.

Returns
-------
plotly.graph_objects.Figure or str
    The interactive figure, or the saved path when `graph_filepath`
    is given. Needs the ``interactive`` extra
    (``pip install "ratio_study[interactive]"``).

---

## `vertical_equity_panel`

```python
vertical_equity_panel(df, assessment_header=None, market_value_header=None,
                      mkt_value_proxy_header=None, prb_mvp_header=None,
                      mvp_sale_price_share=0.5, weights_header=None, asr=None,
                      mkt_value_proxy=None, prb_value=None, prb_not_significant=None,
                      prd_value=None, vei_result=None, vem_result=None, cod_value=None,
                      assessment_standards=None, req_significance=0.05,
                      vei_conf_level=0.9, vei_acceptable_range=10,
                      vei_percentile_method='R7', vei_use_log_normal_delta=True,
                      vem_acceptable_inequity_level=0.1, vem_use_mvp=False,
                      vem_window_size=None, vem_conf_level=0.9, vem_zoom_y=True,
                      vem_smart_zoom=True, vem_asr_col='asr', include_vem=True,
                      gauge_outer_multiple=3.0, x_scale='log',
                      title='Vertical Equity Panel', subtitle='', figsize=(8.5, 11.0),
                      return_object=False, graph_filepath='', show=None)
```

Build the Vertical Equity Panel: four indicator gauges over two charts.

The figure is laid out in letter-page proportions (portrait), so it
drops into a report page as is: a row of gauges on top, then the VEI
boxplot, then the VEM moving-median chart, each spanning the full
width.

The top row holds one gauge per vertical-equity indicator -- PRB, PRD,
VEI and VEM (the last one only when `include_vem` is True). Every
dial has the same fixed geometry, so they are directly comparable at
a glance:

* a 45 degree **green** wedge on top -- the result is within the
  acceptable band;
* a 135 degree **red** wedge on the right -- **progressivity**
  (higher-value properties assessed at higher ratios);
* a 135 degree **red** wedge on the left -- **regressivity**;
* a 45 degree **grey** wedge at the bottom -- an **inconclusive**
  result (not statistically significant, or the confidence interval
  still overlaps the acceptable band).

Because the wedges are fixed, the value-to-angle scale is piecewise
linear: neutrality always points straight up, each acceptable-band
edge lands exactly on a green wedge edge (which is what lets PRD's
asymmetric ``(0.98, 1.03)`` band share the dial with PRB's symmetric
``(-0.10, 0.10)``), and ``gauge_outer_multiple`` times the band edge
reaches the outer end of the red wedge. PRD is reflected, since a PRD
below 1 indicates progressivity, so "right means progressive" holds on
all four gauges.

The needle always shows where the point estimate falls, so it agrees
with the number printed inside the dial. The one thing that overrides
it is inconclusiveness: a PRB that is not significant at
`req_significance`, or a VEI/VEM whose own conclusion is inconclusive,
points at the grey wedge instead. Where an indicator's significance
test reaches a verdict its point estimate does not support, that
verdict is reported on the line beneath the gauge (e.g. ``'VEI test:
Acceptable'`` under a regressive needle) rather than silently moving
the needle.

Below the gauges, two charts are stacked: the VEI boxplot of ratios
by market-value percentile group (from
:func:`ratio_study.indicators.vei_analysis`), then the VEM moving
median of the ASR (from :func:`ratio_study.indicators.vem`). VEM
computes its indices on the median-centred ASR, but the chart is
shifted back into ASR units, so its reference line sits at the median
ASR (typically near 1) rather than at 0, on the same scale as the VEI
boxplot above it.

Any indicator not supplied is computed from `df`. Supplying the
already-computed results (as :func:`ratio_study.ratios` does) avoids
recomputation and guarantees the panel matches the ratios table.

Parameters
----------
df : DataFrame
    pandas/polars/pyarrow/narwhals DataFrame with the sales data.
assessment_header : str, optional
    Column name for the assessed / predicted value. Required unless
    both `asr` and all four indicator results are supplied.
market_value_header : str, optional
    Column name for the market value / time-adjusted sale amount.
    Required under the same condition as `assessment_header`.
mkt_value_proxy_header : str, optional
    Column to use as the market value proxy for the VEI grouping. When
    None, the IAAO proxy ``0.5 * market value + 0.5 * (assessment /
    median ASR)`` is used, matching :func:`ratio_study.ratios`.
prb_mvp_header : str, optional
    Column to use as the market-value proxy in the PRB regression,
    overriding the blend `prb` builds itself. Only used when
    `prb_value` is not supplied, since PRB is recomputed here only in
    that case; independent of `mkt_value_proxy_header`, which is for VEI.
mvp_sale_price_share : float, default 0.5
    Share of the sale price in the market value proxy -- see
    :func:`ratio_study.market_value_proxy`. Applies to the VEI grouping
    proxy and to the PRB regression's proxy, for whichever of the two
    is not supplied as a column or an array.
weights_header : str, optional
    Column with sample weights, rescaled internally to mean 1.
asr : array-like, optional
    Precomputed assessment/sales ratios. Derived from
    `assessment_header` / `market_value_header` when omitted.
mkt_value_proxy : array-like, optional
    Precomputed market value proxy, taking precedence over
    `mkt_value_proxy_header`.
prb_value : float or str, optional
    Precomputed PRB, optionally carrying a non-significance marker
    such as ``'0.043*'``.
prb_not_significant : bool, optional
    True when the PRB slope is not significant at `req_significance`,
    i.e. the ``prb_sig_flag`` returned by
    :func:`ratio_study.indicators.prb`.
prd_value : float, optional
    Precomputed PRD.
vei_result : dict, optional
    Precomputed result from
    :func:`ratio_study.indicators.vei_analysis`. Either the metrics
    dictionary or the ``{'vei_metrics': ..., 'group_stats': ...}``
    wrapper is accepted; it must retain ``'pg_stats'`` for the VEI
    boxplot to be drawn.
vem_result : dict, optional
    Precomputed result from :func:`ratio_study.indicators.vem`,
    including its ``'dataframe'`` for the moving-median chart.
cod_value : float, optional
    Precomputed coefficient of dispersion as a proportion (e.g.
    0.083), reported as ``'COD: 8.3%'`` under the VEI gauge. Computed
    from the ratios when omitted.
assessment_standards : dict, optional
    Acceptable bands keyed by indicator. Defaults to
    ``{'PRB': (-0.05, 0.05), 'PRD': (0.98, 1.03)}``.
req_significance : float, default 0.05
    Required significance level for the PRB slope test.
vei_conf_level : float, default 0.9
    Confidence level for the VEI median confidence intervals.
vei_acceptable_range : float, default 10
    Acceptable |VEI| in percent.
vei_percentile_method : str, default 'R7'
    Percentile interpolation method used to form the VEI groups.
vei_use_log_normal_delta : bool, default True
    Passed through to ``vei_analysis``.
vem_acceptable_inequity_level : float, default 0.1
    Acceptable inequity level for ``vem``, as a proportion. Also
    sets the VEM gauge's acceptable band.
vem_use_mvp : bool, default False
    If True, use the market value proxy (as used in the VEI, so
    honouring `mvp_sale_price_share`) for sorting prices, otherwise use
    sale price (market_value_header)
vem_window_size : int, optional
    Rolling-window size for ``vem``. Defaults to the same rule
    used by :func:`ratio_study.ratios`: 1000 rows when the sample
    exceeds 10,000, otherwise a tenth of the sample.
vem_conf_level : float, default 0.9
    Confidence level for the moving-median CI band.
vem_zoom_y : bool, list, or tuple, default True
    Y-axis range for the moving-median chart. True zooms to the
    median ASR +/- 0.3; a ``(low, high)`` pair sets a custom range,
    in ASR units; False leaves it auto-ranged. Only used when
    `vem_smart_zoom` is False, unless it is an explicit pair, which
    always wins.
vem_smart_zoom : bool, default True
    Frame the moving-median chart on the bulk of the scatter -- the
    x-axis on the 1st-99th percentile of the market value proxy, the
    y-axis on 95% of the ASR values inside that window -- instead of
    on the full data range. See :func:`_ve_smart_limits`.
vem_asr_col : str, default 'asr'
    Name of the ASR column in the ``vem``
    working frame. Only needs changing when a `vem_result` computed
    with a non-default ``asr_col`` is passed in.
include_vem : bool, default True
    Whether to report the VEM verdict. False drops the VEM gauge (the
    remaining three are re-centred) and strips the moving-median
    chart of everything that reports a verdict: no
    acceptable-inequity lines, and one plain blue scatter instead of
    the green/red split by local median. The moving median and its CI
    band stay, so the chart still shows how the level of assessment
    moves across the value range.
gauge_outer_multiple : float, default 3.0
    Full scale of each red wedge, as a multiple of the band edge.
x_scale : str, default 'log'
    X-axis scale for the moving-median chart, 'linear' or 'log'.
title : str, default 'Vertical Equity Panel'
    Panel title.
subtitle : str, default ''
    Optional second title line, e.g. a model name or date reference.
figsize : tuple, default (8.5, 11.0)
    Figure size in inches; the default is a letter page in portrait.
    The saved image is not tightly cropped, so it keeps whatever
    proportions are given here.
return_object : bool, default False
    If True, return ``[title, image_obj]`` where `image_obj` is a
    ReportLab ``Image``, ready to append to the report's graph lists.
graph_filepath : str, default ''
    If given (and `return_object` is False), save the panel to this
    path ('.png' appended when missing) and return the path.
show : bool, optional
    Whether to call ``plt.show()``. Defaults to True only when the
    matplotlib backend is not an inline/notebook one, so that a
    notebook does not render the panel twice.

Returns
-------
list or str or matplotlib.figure.Figure
    * ``[title, image_obj]`` when `return_object` is True.
    * The filepath (str) when `graph_filepath` is given.
    * Otherwise the ``matplotlib.figure.Figure``.

Examples
--------
Inline in a notebook:

>>> from ratio_study.graphs import vertical_equity_panel
>>> fig = vertical_equity_panel(df, 'assessment', 'sale_price')

In the PDF report, via :func:`ratio_study.ratios_report`:

>>> ratios_report(df, 'assessment', 'sale_price',
...               indicators=['Median', 'CoD', 'PRD', 'PRB', 'VEI', 'VEM'],
...               pdf_args={'pdf_vertical_equity_panel': True})
