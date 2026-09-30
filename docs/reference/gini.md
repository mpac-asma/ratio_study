# Gini family

Quintos’ Gini measures and decompositions (Appendix F).

*Module:* `ratio_study.gini`

## Contents

- [`gini_decomp`](#gini-decomp) — Calculate the subgroup effects of Gini-based inequality indices via decomposition.
- [`gini_factors`](#gini-factors) — Calculate Gini factor decomposition estimates based on Quintos (2021).
- [`gini_lorenz_plot`](#gini-lorenz-plot) — Generate Lorenz curves for sale prices and assessments.
- [`gini_stats`](#gini-stats) — Calculate Gini equality indicators for sale prices and assessments.

---

## `gini_decomp`

```python
gini_decomp(sale_prices, assessed_values, group_by, grouping_label, population_ratios,
            export_directory, dict_results=None, unbiased=False, decomp_chart='',
            weights=None)
```

Calculate the subgroup effects of Gini-based inequality indices via decomposition.

Parameters
----------
sale_prices : array-like
    Numeric sale prices (numpy array, pandas series, or list).
assessed_values : array-like
    Numeric assessed values (numpy array, pandas series, or list).
group_by : array-like
    Group classifications, one per observation. Can be numeric or
    string; values are coerced to string.
grouping_label : str
    Description of the grouping variable, used in chart titles.
population_ratios : dict
    Population-level Gini ratios, as returned by ``gini_stats``. Index
    0 holds the overall statistics, index 1 holds the statistics used
    for the residual calculation.
export_directory : str
    Directory where generated chart images are saved.
dict_results : dict, optional
    Accumulator dictionary tracking exported file paths under the
    ``'exported_files'`` key. Created internally as
    ``{'exported_files': []}`` when None.
unbiased : bool, default False
    Use True if sample size < 30, to apply the small-sample correction.
decomp_chart : str, default ''
    Type of chart to generate for the decomposition:

    * 'bar' -- single bar chart with within/between/residual effects.
    * 'pie_with_bar' -- pie chart with 'within' values decomposed in a bar chart on the right hand side.
    * '' -- no chart is generated.
weights : array-like, optional
    Observation weights, assumed to have mean = 1.

Returns
-------
tuple of (dict, dict)
    `group_results` -- the Gini decomposition results (within,
    between, residual, and optionally chart objects); and
    `dict_results` -- the accumulator dict with any exported chart
    file paths appended.

---

## `gini_factors`

```python
gini_factors(data, depvar_header, indvar_headers, unbiased=False)
```

Calculate Gini factor decomposition estimates based on Quintos (2021).

Credit to Michael Boateng.

Parameters
----------
data : DataFrame
    pandas, polars, pyarrow, or narwhals DataFrame containing the
    dependent and independent variables.
depvar_header : str
    Column name for the dependent variable (time-adjusted sale price).
indvar_headers : list of str
    Column names for all independent variables used in the model.
unbiased : bool, default False
    Whether to apply the small-sample bias correction when computing
    the concentration index used for Factor MKI.

Returns
-------
pandas.DataFrame
    One row per independent variable, sorted descending by
    ``'% of Total MKI'``, with columns ``'Variable'``, ``'Mean'``,
    ``'Elasticities'``, ``'CI'``, ``'Coefficient'``,
    ``'Contribution'``, ``'Factor MKI'``, and ``'% of Total MKI'``.
    Returned as a plain pandas DataFrame regardless of the input
    DataFrame type (gating constraint of
    ``statsmodels.formula.api.ols``).

---

## `gini_lorenz_plot`

```python
gini_lorenz_plot(market_value, assessment, gini_sale_prices, concentration_index, title,
                 filepath='')
```

Generate Lorenz curves for sale prices and assessments.

Parameters
----------
market_value : array-like
    Sale prices (numpy array, pandas/polars/narwhals Series, or list).
assessment : array-like
    Assessed values (numpy array, pandas/polars/narwhals Series, or list).
gini_sale_prices : float
    Gini coefficient of sale prices, shown in the legend label.
concentration_index : float
    Concentration index, shown in the legend label.
title : str
    Plot title.
filepath : str, default ''
    Path to save the chart as a PNG. When empty, the figure object
    is returned instead of being saved.

Returns
-------
str or matplotlib.figure.Figure
    The saved file path (with a ``.png`` extension appended if
    missing) when `filepath` is given; otherwise the
    ``matplotlib.figure.Figure`` object.

---

## `gini_stats`

```python
gini_stats(market_value, assessment, weights=None, unbiased=False,
           between_groups_rank=False, run_regression=True, plot_ref='',
           gini_lorenz_plots=False, export_directory='')
```

Calculate Gini equality indicators for sale prices and assessments.

Parameters
----------
market_value : array-like
    Numeric sale prices (numpy array, pandas series, or list).
assessment : array-like
    Numeric assessed values (numpy array, pandas series, or list).
weights : array-like, optional
    Observation weights, assumed to have mean = 1.
unbiased : bool, default False
    Use True if sample size < 30, to apply the small-sample
    correction.
between_groups_rank : array-like or False, default False
    For between-groups decomposition, provide a specific fractional
    rank array to use instead of the default within-sample ranking.
run_regression : bool, default True
    Enables calculation of the Reranking Effect, Reranking Index,
    Kakwani Index, and Modified Kakwani Index (plus their
    supporting regressions and p-values) in addition to the basic
    Gini coefficients.
plot_ref : str, default ''
    Text used in the plot title. Only used when `gini_lorenz_plots`
    is True.
gini_lorenz_plots : bool, default False
    If True, generate a Lorenz curve chart via `gini_lorenz_plot`.
export_directory : str, default ''
    Directory where the Lorenz curve chart is saved, when
    `gini_lorenz_plots` is True.

Returns
-------
dict
    Dictionary of Gini indicators, including
    ``'Gini_Sale_Prices'``, ``'Gini_Assessments'``,
    ``'Concentration_Index'``, and ``'Redistributive_Effect'``.
    When `run_regression` is True, also includes ``'Regressions'``
    (per-indicator regression details), ``'Kakwani_Index'``,
    ``'Modified_Kakwani_Index'``, ``'Reranking_Effect'``, and
    ``'Reranking_Index'``. When `gini_lorenz_plots` is True, also
    includes ``'gini_lorenz_plot'`` with the saved chart's file
    path.
