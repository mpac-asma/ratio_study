# Blinder–Oaxaca

Decomposing a ratio gap between two groups.

*Module:* `ratio_study.blinder_oaxaca`

## Contents

- [`bo_coefficient_plot`](#bo-coefficient-plot) — Draw a horizontal dot-and-whisker plot of per-variable contributions
- [`bo_decompose`](#bo-decompose) — Decompose the mean log-ASR gap between two groups using the
- [`bo_decomposition_plot`](#bo-decomposition-plot) — Draw a grouped bar chart of explained vs unexplained contributions
- [`bo_detail_plot`](#bo-detail-plot) — Draw a single-component per-variable dot-and-whisker plot
- [`bo_format_table`](#bo-format-table) — Build a publication-ready summary table from a single
- [`bo_multi_decomposition`](#bo-multi-decomposition) — Run the Blinder-Oaxaca decomposition across multiple strata and/or

---

## `bo_coefficient_plot`

```python
bo_coefficient_plot(decomp: 'dict', *, components: 'Optional[List[str]]' = None,
                    exclude_const: 'bool' = True,
                    title: 'str' = 'Blinder-Oaxaca Coefficient Plot',
                    figsize: 'Tuple[float, float]' = (8, 6),
                    filepath: 'Optional[str]' = None, return_object: 'bool' = True) -> 'Optional[plt.Figure]'
```

Draw a horizontal dot-and-whisker plot of per-variable contributions
(replicates Figure 2 / Figure 4 in Petit & Tedds 2025).

Parameters
----------
decomp : dict
    Result dictionary returned by ``bo_decompose``.
components : list of str, optional
    Which components to plot, e.g. ``['Q', 'U']`` or
    ``['E', 'C', 'I']``. Defaults to the twofold ``['Q', 'U']`` if
    available in ``decomp['detail']``, else ``['E', 'C', 'I']``.
exclude_const : bool, default True
    If True, drop the ``'const'`` row from the detail table before
    plotting.
title : str, default 'Blinder-Oaxaca Coefficient Plot'
    Figure title.
figsize : tuple of float, default (8, 6)
    Figure size in inches, passed to ``matplotlib.pyplot.subplots``.
filepath : str, optional
    If given, save the figure to this path at 150 DPI.
return_object : bool, default True
    If True, return the ``matplotlib.figure.Figure`` object;
    otherwise close the figure and return None.

Returns
-------
matplotlib.figure.Figure or None
    The figure object if ``return_object`` is True, otherwise None.

---

## `bo_decompose`

```python
bo_decompose(df, asr_col: 'str', group_col: 'str', characteristics: 'List[str]', *,
             log_asr: 'bool' = True, reference: 'Union[str, float]' = 'pooled',
             threshold: 'Optional[float]' = None,
             group_high_label: 'Optional[str]' = None,
             group_low_label: 'Optional[str]' = None,
             dummy_sets: 'Optional[Dict[str, List[str]]]' = None,
             detail_groups: 'Optional[Dict[str, List[str]]]' = None,
             alpha: 'float' = 0.05, cov_type: 'str' = 'HC1', threefold: 'bool' = True,
             twofold: 'bool' = True) -> 'dict'
```

Decompose the mean log-ASR gap between two groups using the
Blinder-Oaxaca method.

Parameters
----------
df : pandas / polars / pyarrow / narwhals DataFrame
    Must contain ``asr_col``, ``group_col`` and all
    ``characteristics``. The frame is converted to a pandas
    DataFrame internally because ``statsmodels.api.OLS`` (used
    here) and the per-row numpy conversions below are most
    efficient on pandas objects.
asr_col : str
    Column holding the ASR (or log-ASR if ``log_asr=False``).
group_col : str
    Binary, categorical, or continuous grouping variable.
characteristics : list of str
    Column names of property / neighbourhood characteristics.
log_asr : bool, default True
    If True, the module takes ``ln(asr_col)`` internally. Set False
    when the column is already in log form.
reference : str or float, default 'pooled'
    Reference coefficient vector. ``'pooled'`` adds a group
    indicator to the pooled regression per Jann (2008). Also
    accepts ``'group_0'``, ``'group_1'``, ``'reimers'``,
    ``'cotton'``, ``'sloczynski'``, or a float lambda in [0, 1].
threshold : float, optional
    For non-binary ``group_col``: values greater than ``threshold``
    are assigned to group 1. If None and the column is non-binary,
    the median is used for continuous variables; for categorical
    variables, the user must supply a threshold or ensure the
    column is already binary.
group_high_label : str, optional
    Human-readable name for group 1 (e.g. ``'Vulnerable'``).
group_low_label : str, optional
    Human-readable name for group 0 (e.g. ``'Non-Vulnerable'``).
dummy_sets : dict, optional
    Mapping of ``{set_name: [col_name_1, col_name_2, ...]}``. Each
    listed set of columns is treated as dummies from a single
    categorical variable and subjected to deviation-contrast
    normalisation (Yun 2005) so that detailed results are
    base-category invariant.
detail_groups : dict, optional
    Mapping of ``{group_label: [col_name, ...]}``. Aggregates
    individual variable contributions into named groups for
    plotting/tables (e.g. ``{'Property': ['age', 'area'],
    'Neighbourhood': ['green']}``).
alpha : float, default 0.05
    Significance level for confidence intervals.
cov_type : str, default 'HC1'
    Covariance estimator passed to statsmodels.
threefold : bool, default True
    Compute the three-fold (E + C + I) decomposition.
twofold : bool, default True
    Compute the two-fold (Q + U) decomposition.

Returns
-------
dict
    Dictionary with keys:

    * overall : dict -- aggregate components and standard errors.
    * detail : DataFrame -- one row per variable with per-variable
      explained / unexplained / endowments / coefficients /
      interaction contributions.
    * group_stats : dict -- means and counts per group.
    * models : dict -- fitted OLS result dicts, keyed by
      ``'group_1'`` and ``'group_0'``.
    * reference : str -- label for the reference choice.
    * alpha : float -- alpha used.
    * var_names : list of str -- design-matrix column names,
      including ``'const'``.
    * detail_groups : DataFrame or None -- aggregated-group
      decomposition, if ``detail_groups`` was requested.

---

## `bo_decomposition_plot`

```python
bo_decomposition_plot(multi_result: 'dict', *,
                      component_pair: 'Tuple[str, str]' = ('Q', 'U'),
                      title: 'str' = 'Blinder-Oaxaca Decomposition by Stratum',
                      figsize: 'Tuple[float, float]' = (10, 6),
                      filepath: 'Optional[str]' = None, return_object: 'bool' = True) -> 'Optional[plt.Figure]'
```

Draw a grouped bar chart of explained vs unexplained contributions
across strata and reference models (replicates Figure 3 in Petit &
Tedds 2025).

Parameters
----------
multi_result : dict
    Result dictionary returned by ``bo_multi_decomposition``,
    containing a ``'summary_table'`` DataFrame.
component_pair : tuple of str, default ('Q', 'U')
    Pair of component columns from
    ``multi_result['summary_table']`` to plot side by side for each
    stratum/reference (e.g. ``('E', 'C')``).
title : str, default 'Blinder-Oaxaca Decomposition by Stratum'
    Figure title.
figsize : tuple of float, default (10, 6)
    Figure size in inches, passed to ``matplotlib.pyplot.subplots``.
filepath : str, optional
    If given, save the figure to this path at 150 DPI.
return_object : bool, default True
    If True, return the ``matplotlib.figure.Figure`` object;
    otherwise close the figure and return None.

Returns
-------
matplotlib.figure.Figure or None
    The figure object if ``return_object`` is True and the summary
    table is non-empty and contains ``component_pair``; otherwise
    None.

---

## `bo_detail_plot`

```python
bo_detail_plot(decomp: 'dict', *, component: 'str' = 'Q', exclude_const: 'bool' = True,
               title: 'Optional[str]' = None, figsize: 'Tuple[float, float]' = (7, 5),
               filepath: 'Optional[str]' = None, return_object: 'bool' = True) -> 'Optional[plt.Figure]'
```

Draw a single-component per-variable dot-and-whisker plot
(replicates individual panels in Figures 2/4).

Parameters
----------
decomp : dict
    Result dictionary returned by ``bo_decompose``.
component : str, default 'Q'
    Which component column from ``decomp['detail']`` to plot (e.g.
    ``'Q'``, ``'U'``, ``'E'``, ``'C'``, ``'I'``).
exclude_const : bool, default True
    If True, drop the ``'const'`` row from the detail table before
    plotting.
title : str, optional
    Figure title. Defaults to ``'Detail: <component label>'`` when
    None.
figsize : tuple of float, default (7, 5)
    Figure size in inches, passed to ``matplotlib.pyplot.subplots``.
filepath : str, optional
    If given, save the figure to this path at 150 DPI.
return_object : bool, default True
    If True, return the ``matplotlib.figure.Figure`` object;
    otherwise close the figure and return None.

Returns
-------
matplotlib.figure.Figure or None
    The figure object if ``return_object`` is True and
    ``component`` is found in ``decomp['detail']``; otherwise None.

---

## `bo_format_table`

```python
bo_format_table(decomp: 'dict', *, proportions: 'bool' = True, stars: 'bool' = True,
                decimals: 'int' = 4) -> 'pd.DataFrame'
```

Build a publication-ready summary table from a single
``bo_decompose`` result.

Parameters
----------
decomp : dict
    Result dictionary returned by ``bo_decompose``.
proportions : bool, default True
    If True, include a ``% of R`` column showing each component as
    a share of the raw gap.
stars : bool, default True
    If True, append significance stars (``*`` p<.05, ``**`` p<.01,
    ``***`` p<.001) in a ``Sig`` column.
decimals : int, default 4
    Number of decimal places for display formatting.

Returns
-------
DataFrame
    One row per component (raw gap plus each requested
    decomposition component), with columns ``Component``,
    ``Estimate``, ``SE``, ``p-value``, ``% of R`` (if
    ``proportions``), and ``Sig``.

---

## `bo_multi_decomposition`

```python
bo_multi_decomposition(df, asr_col: 'str', group_col: 'str',
                       characteristics: 'List[str]', *,
                       stratify_by: 'Optional[Union[str, List[str]]]' = None,
                       references: 'Optional[List[Union[str, float]]]' = None,
                       log_asr: 'bool' = True, threshold: 'Optional[float]' = None,
                       group_high_label: 'Optional[str]' = None,
                       group_low_label: 'Optional[str]' = None,
                       dummy_sets: 'Optional[Dict[str, List[str]]]' = None,
                       detail_groups: 'Optional[Dict[str, List[str]]]' = None,
                       alpha: 'float' = 0.05, cov_type: 'str' = 'HC1') -> 'dict'
```

Run the Blinder-Oaxaca decomposition across multiple strata and/or
multiple reference models, returning a structure suitable for
Table 3 / Figure 3 style outputs.

Parameters
----------
df : pandas / polars / pyarrow / narwhals DataFrame
    Must contain ``asr_col``, ``group_col`` and all
    ``characteristics``. Converted to a pandas DataFrame internally.
asr_col : str
    Column holding the ASR (or log-ASR if ``log_asr=False``).
group_col : str
    Binary, categorical, or continuous grouping variable.
characteristics : list of str
    Column names of property / neighbourhood characteristics.
stratify_by : str or list of str, optional
    Column(s) defining strata (e.g. ``['city', 'roll_year']``). If
    None, the whole frame is treated as a single stratum labelled
    ``'Overall'``.
references : list, optional
    Reference choices to run for each stratum, each passed to
    ``bo_decompose`` as ``reference``. Defaults to
    ``['group_0', 'group_1', 'sloczynski']`` following the paper's
    gain / loss / gap models.
log_asr : bool, default True
    If True, ``bo_decompose`` takes ``ln(asr_col)`` internally. Set
    False when the column is already in log form.
threshold : float, optional
    For non-binary ``group_col``: values greater than ``threshold``
    are assigned to group 1. Passed through to ``bo_decompose``.
group_high_label : str, optional
    Human-readable name for group 1.
group_low_label : str, optional
    Human-readable name for group 0.
dummy_sets : dict, optional
    Mapping of ``{set_name: [col_name_1, col_name_2, ...]}``
    identifying dummy-coded categorical variables for
    deviation-contrast normalisation. Passed through to
    ``bo_decompose``.
detail_groups : dict, optional
    Mapping of ``{group_label: [col_name, ...]}`` used to aggregate
    per-variable contributions. Passed through to ``bo_decompose``.
alpha : float, default 0.05
    Significance level for confidence intervals.
cov_type : str, default 'HC1'
    Covariance estimator passed to statsmodels.

Returns
-------
dict
    Dictionary with keys:

    * results : dict -- nested dict ``{stratum: {reference:
      decomp}}`` where each ``decomp`` is a ``bo_decompose`` result.
    * summary_table : DataFrame -- long-form table with one row per
      stratum/reference combination, for Table 3 style output.
    * baseline_table : DataFrame -- Table 2 style raw group-size
      table per stratum.
