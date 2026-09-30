# ML diagnostics

ASR predictability and the SHAP explanation family.

*Module:* `ratio_study.ml_diagnostics`

## Contents

- [`asr_predictability`](#asr-predictability) — Measure how much of the ASR is predictable from characteristics, and by what.
- [`asr_predictability_panel`](#asr-predictability-panel) — Composite: how much, then by what.
- [`asr_target`](#asr-target) — Assessment-to-sale ratio, ``assessment / market_value``.
- [`cdr`](#cdr) — Conditional Dispersion Ratio -- ``Var(y | group) / Var(y)``.
- [`cmbi`](#cmbi) — Conditional Mean Bias Index -- ``|E[y | group] - E[y]|`` per group.
- [`dcorr`](#dcorr) — Distance correlation: zero if and only if ``x`` and ``y`` are independent.
- [`log_residual`](#log-residual) — ``log(market_value) - log(assessment)``.
- [`r2_correlation`](#r2-correlation) — Squared Pearson correlation between actual and predicted.
- [`shap_beeswarm_plot`](#shap-beeswarm-plot) — SHAP value per property, per characteristic, coloured by feature value.
- [`shap_dependence_plot`](#shap-dependence-plot) — One characteristic's SHAP value against its own value.
- [`shap_importance_plot`](#shap-importance-plot) — Mean |SHAP| per characteristic -- "by what", ranked.
- [`shap_waterfall_plot`](#shap-waterfall-plot) — Decompose one property's predicted ratio into its contributions.

---

## `asr_predictability`

```python
asr_predictability(df, market_value_header, assessment_header, independent_variables,
                   target='asr', group_by=None, n_splits=3, random_state=0,
                   model_params=None, compute_shap=True, shap_sample_pct=0.2,
                   shap_max_rows=None, compute_dcorr=True, dcorr_max_rows=2000,
                   min_group_n=30, thresholds=None, verbose=False)
```

Measure how much of the ASR is predictable from characteristics, and by what.

Fits ``ASR ~ f(X)`` -- Bidanset & Wehrli's ``ASR^ = F(X)`` -- and reports
out-of-fold predictability plus a SHAP decomposition.

Sale price is deliberately **not** offered as a feature. Because
``ASR = AV / SP``, a model given ``SP`` need only reconstruct ``AV`` from
``X``, which any competent mass appraisal model makes easy; the resulting
R-squared measures how well ``X`` mimics the assessment model rather than
anything about the ratio. Price-related bias is PRD/PRB/VEM's job.

Parameters
----------
df : DataFrame
    pandas, polars, pyarrow or narwhals frame.
market_value_header, assessment_header : str
    Time-adjusted sale price and assessed value columns.
independent_variables : list of str
    Characteristics to test. The only modelling input required.
target : {'asr', 'log_residual'}, default 'asr'
    ``'asr'`` follows Bidanset & Wehrli. ``'log_residual'`` uses
    ``log(SP) - log(AV)``, which is symmetric in over- and
    under-assessment; see `log_residual`.
group_by : str, optional
    Column for the CMBI / CDR subgroup breakdown.
n_splits : int, default 3
    Cross-fitting folds, following the source presentations.
random_state : int, default 0
    Seeds folds, subsampling and the booster, so results reproduce.
model_params : dict, optional
    Booster overrides. Unnecessary for normal use.
compute_shap, compute_dcorr : bool, default True
    Switches for the two expensive stages.
shap_sample_pct : float, default 0.20
    Share of the sales frame to explain with SHAP, as a fraction. The
    rows are a stratified sample across the sorted target, so they span
    the whole ratio distribution. TreeSHAP cost is linear in rows, so
    this is the main runtime dial; raise it for a small frame, lower it
    for a large one. 1.0 explains every row.
shap_max_rows : int, optional
    Absolute ceiling applied after ``shap_sample_pct``, for when 20% of
    a very large frame is still more than you want to wait for.
dcorr_max_rows : int, default 2000
    Cap for the O(n^2) distance correlation. Reductions are reported in
    ``model_info``.
min_group_n : int, default 30
    Subgroups below this are marked unreliable rather than dropped.
thresholds : dict, optional
    Overrides for `DEFAULT_THRESHOLDS`.
verbose : bool, default False
    Print stage timings.

Returns
-------
dict
    ``predictability`` (float), ``r2`` (float), ``summary``,
    ``shap_importance``, ``shap_values``, ``shap_base``, ``shap_data``,
    ``dcorr``, ``cmbi_cdr``, ``y``, ``oof_prediction``, ``model_info``,
    ``attribution``.

---

## `asr_predictability_panel`

```python
asr_predictability_panel(diag, top_n=12, n_dependence=4, title=None, figsize=(13.5, 12),
                         return_object=False, graph_filepath='', show=None)
```

Composite: how much, then by what.

A headline predictability figure, the importance ranking, the beeswarm,
and dependence plots for the leading characteristics.

---

## `asr_target`

```python
asr_target(market_value, assessment)
```

Assessment-to-sale ratio, ``assessment / market_value``.

---

## `cdr`

```python
cdr(y, groups, min_n=30)
```

Conditional Dispersion Ratio -- ``Var(y | group) / Var(y)``.

Values far from 1.0 mark subgroups treated much more, or much less,
consistently than the roll as a whole. ``min_n`` behaves as in `cmbi`.

---

## `cmbi`

```python
cmbi(y, groups, min_n=30)
```

Conditional Mean Bias Index -- ``|E[y | group] - E[y]|`` per group.

Which subgroups sit systematically above or below the overall level.

Parameters
----------
min_n : int, default 30
    Groups smaller than this are marked ``reliable=False``. A stratum of
    two sales produces a large index from noise alone, so flagging by
    size keeps those out of the headline.

---

## `dcorr`

```python
dcorr(x, y, max_rows=2000, random_state=0)
```

Distance correlation: zero if and only if ``x`` and ``y`` are independent.

Model-free, and sensitive to nonlinear and non-monotone dependence that
Pearson correlation misses -- which makes it the cheap first pass for
deciding which characteristics deserve a model at all.

O(n^2) in memory, so samples above ``max_rows`` are subsampled; callers
report the reduction rather than hiding it.

References
----------
Szekely, Rizzo & Bakirov (2007), *Annals of Statistics* 35(6), 2769-2794.

---

## `log_residual`

```python
log_residual(market_value, assessment)
```

``log(market_value) - log(assessment)``.

The alternative target. Symmetric where the raw ratio is not (twice-over
is ``+0.693``, half-under ``-0.693``, against ``+100%`` and ``-50%``), so
an optimiser fitting it is not pulled toward one direction. Available via
``target='log_residual'`` for callers who prefer Wehrli's formulation;
the default follows Bidanset & Wehrli's ``ASR^ = F(X)``.

---

## `r2_correlation`

```python
r2_correlation(actual, predicted)
```

Squared Pearson correlation between actual and predicted.

Used instead of the variance-decomposition R-squared, which goes
negative once a model predicts worse than the mean -- routine for
out-of-fold ratio predictions, and meaningless as a "share explained".

---

## `shap_beeswarm_plot`

```python
shap_beeswarm_plot(diag, top_n=14, title=None, figsize=(10, 7), max_points=3000,
                   return_object=False, graph_filepath='', show=None)
```

SHAP value per property, per characteristic, coloured by feature value.

The densest single view of the decomposition, following the beeswarm in
Bidanset & Wehrli (2025). Each dot is one property. Horizontal position
is that property's SHAP value; colour is the feature's own value, low to
high. So a band running blue-left to red-right means high values of the
characteristic push the ratio up, and a band with colour mixed across
the range means the effect is not monotone.

---

## `shap_dependence_plot`

```python
shap_dependence_plot(diag, feature, title=None, figsize=(8.5, 5.5), show_histogram=True,
                     return_object=False, graph_filepath='', show=None)
```

One characteristic's SHAP value against its own value.

Follows the dependence panels in Bidanset & Wehrli (2025), including
the grey histogram of the feature's distribution, which is what keeps a
dramatic-looking swing in a sparse tail from being over-read. The red
line is a binned median, robust to the outliers a mean would chase.

---

## `shap_importance_plot`

```python
shap_importance_plot(diag, top_n=15, title=None, figsize=(9, 6), return_object=False,
                     graph_filepath='', show=None)
```

Mean |SHAP| per characteristic -- "by what", ranked.

Follows the SHAP feature-importance bar in Bidanset & Wehrli (2025),
labelled with the magnitude in target units so the bars can be read as
"this characteristic moves the ratio by about this much".

---

## `shap_waterfall_plot`

```python
shap_waterfall_plot(diag, row=0, top_n=14, title=None, figsize=(9.5, 7),
                    return_object=False, graph_filepath='', show=None)
```

Decompose one property's predicted ratio into its contributions.

Follows the waterfall in Bidanset & Wehrli (2025): starts from the base
value ``E[f(X)]`` -- the average predicted ratio -- and adds each
characteristic's contribution to reach ``f(x)`` for this property. This
is the chart that answers an individual appeal: *why does the model
think this property's ratio differs from the norm?*

Parameters
----------
row : int, default 0
    Positional index into ``diag['shap_data']``, not the original frame.
