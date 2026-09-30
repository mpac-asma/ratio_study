# Reporting helpers

Comparison tables and export helpers.

*Module:* `ratio_study.reporting`

## Contents

- [`compare_ratios`](#compare-ratios) — Calculate the difference and percentage change for each indicator between two ratios dataframes.

---

## `compare_ratios`

```python
compare_ratios(r1, r2, suffixes=('_r1', '_r2'), difference=True, pct_chg=True,
               merging_cols=['Level', 'Group'], indicators=None)
```

Calculate the difference and percentage change for each indicator between two ratios dataframes.

Useful for comparing ratios before/after a change in a model,
sales filtering, etc.

Parameters
----------
r1 : DataFrame or dict
    First ratios dataframe (pandas, polars, pyarrow or narwhals),
    or a dict with the ratios dataframe under key
    ``'df_all_ratios'``.
r2 : DataFrame or dict
    Second ratios dataframe (pandas, polars, pyarrow or narwhals),
    or a dict with the ratios dataframe under key
    ``'df_all_ratios'``.
suffixes : tuple, default ('_r1', '_r2')
    Suffixes applied to ``r1`` and ``r2`` columns, respectively,
    when merging. Suggested to start with ``'_'``, since it is not
    added automatically if omitted.
difference : bool, default True
    If True, calculate the difference (``r2`` - ``r1``) for each
    indicator.
pct_chg : bool, default True
    If True, calculate the percentage change from ``r1`` to ``r2``
    for each indicator.
merging_cols : list, default ['Level', 'Group']
    Columns to merge ``r1`` and ``r2`` on. Only columns present in
    both dataframes are used.
indicators : list, optional, default None
    Column names present in both ``r1`` and ``r2`` corresponding
    to indicators to compare. If None, all columns common to both
    dataframes that are not in ``merging_cols`` are used.

Returns
-------
pandas.DataFrame
    Dataframe with ``merging_cols`` plus, for each indicator, its
    ``r1`` and ``r2`` columns side-by-side (suffixed per
    ``suffixes``), and the difference and/or percentage-change
    columns when ``difference=True`` and/or ``pct_chg=True``,
    respectively.
