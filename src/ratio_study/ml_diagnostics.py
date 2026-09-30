"""How much of the ASR is predictable, and by what?

Based on Paul Bidanset and Joe Wehrli's presentation at the GIS/Valuation
Technologies Conference 2025 ("Enhancing Ratio Studies with AI: Identifying
Sources of Inequity with SHAP Values"), and on Joe Wehrli's presentation at
the IAAO/IPTI Mass Appraisal Valuation Symposium 2026 ("Automating Equity
Analysis with Machine Learning and AI").

The question, in Bidanset & Wehrli's form ``ASR^ = F(X)``:

* **How much** of the assessment-to-sale ratio can a model predict from
  property characteristics? If the ASR were pure noise with respect to ``X``,
  nothing about a property would tell you how it was assessed relative to its
  sale price. To the extent a model *can* predict it, the ratio carries
  structure tied to characteristics.
* **By what**? SHAP decomposes that predictability into per-characteristic,
  per-property contributions, in ASR units.

Predictability is ``Corr(actual, predicted) ** 2`` on **out-of-fold**
predictions. In-sample R-squared for a boosted tree is ~1 regardless of the
data, so an in-sample number would report every roll as fully predictable.

.. warning::
   These are **diagnostics, not acceptance tests**. Both presentations
   describe the work as in progress, with open questions about which
   statistical tests over SHAP populations are reliable. Nothing here emits
   a pass/fail grade, and ``DEFAULT_THRESHOLDS`` holds indicative starting
   points, not IAAO standards.

.. warning::
   SHAP values explain the **model of the ratio**, not the assessment model.
   A large SHAP value for ``living_area`` means living area predicts the
   *ratio*, not that it drives assessed value.

Requires the ``ml`` extra::

    pip install "ratio_study[ml]"
"""
import time
import warnings
from io import BytesIO

import numpy as np
import pandas as pd

from . import nw_utils, _optional

__all__ = [
    'asr_target',
    'log_residual',
    'r2_correlation',
    'cmbi',
    'cdr',
    'dcorr',
    'asr_predictability',
    'shap_importance_plot',
    'shap_beeswarm_plot',
    'shap_dependence_plot',
    'shap_waterfall_plot',
    'asr_predictability_panel',
    'DEFAULT_THRESHOLDS',
    'ATTRIBUTION',
]

ATTRIBUTION = (
    "Method: Bidanset & Wehrli, GIS/Valuation Technologies 2025; "
    "Wehrli, IAAO/IPTI MAVS 2026"
)

DEFAULT_THRESHOLDS = {
    'r2': 0.10,      # above this, the ratio is meaningfully predictable
    'dcorr': 0.20,   # above this, a variable relates to the ratio
    'cmbi': 0.05,    # |mean centred ratio| within a subgroup
}

# Categoricals with more levels than this are frequency-encoded.
_MAX_CARDINALITY = 50

# Blue -> red, matching the SHAP convention for low -> high feature value.
_LOW, _HIGH = '#008bfb', '#ff0051'


# ==========================================================================
# Targets
# ==========================================================================

def asr_target(market_value, assessment):
    """Assessment-to-sale ratio, ``assessment / market_value``."""
    mv = nw_utils.to_numpy_1d(market_value).astype(float)
    av = nw_utils.to_numpy_1d(assessment).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        asr = av / mv
    asr[~np.isfinite(asr) | (mv <= 0) | (av <= 0)] = np.nan
    return asr


def log_residual(market_value, assessment):
    """``log(market_value) - log(assessment)``.

    The alternative target. Symmetric where the raw ratio is not (twice-over
    is ``+0.693``, half-under ``-0.693``, against ``+100%`` and ``-50%``), so
    an optimiser fitting it is not pulled toward one direction. Available via
    ``target='log_residual'`` for callers who prefer Wehrli's formulation;
    the default follows Bidanset & Wehrli's ``ASR^ = F(X)``.
    """
    mv = nw_utils.to_numpy_1d(market_value).astype(float)
    av = nw_utils.to_numpy_1d(assessment).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(mv) - np.log(av)
    r[~np.isfinite(r)] = np.nan
    return r


# ==========================================================================
# Metrics -- pure numpy, no ML dependency
# ==========================================================================

def r2_correlation(actual, predicted):
    """Squared Pearson correlation between actual and predicted.

    Used instead of the variance-decomposition R-squared, which goes
    negative once a model predicts worse than the mean -- routine for
    out-of-fold ratio predictions, and meaningless as a "share explained".
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(p[ok]) == 0:
        return 0.0
    return float(np.corrcoef(a[ok], p[ok])[0, 1] ** 2)


def cmbi(y, groups, min_n=30):
    """Conditional Mean Bias Index -- ``|E[y | group] - E[y]|`` per group.

    Which subgroups sit systematically above or below the overall level.

    Parameters
    ----------
    min_n : int, default 30
        Groups smaller than this are marked ``reliable=False``. A stratum of
        two sales produces a large index from noise alone, so flagging by
        size keeps those out of the headline.
    """
    y = np.asarray(y, dtype=float)
    g = pd.Series(np.asarray(groups)).astype(str).values
    ok = np.isfinite(y)
    overall = np.mean(y[ok])
    frame = pd.DataFrame({'group': g[ok], 'y': y[ok]})
    out = (frame.groupby('group')['y']
                .agg(n='size', group_mean='mean')
                .reset_index())
    out['cmbi'] = (out['group_mean'] - overall).abs()
    out['reliable'] = out['n'] >= min_n
    return (out.sort_values(['reliable', 'cmbi'], ascending=[False, False])
               .reset_index(drop=True))


def cdr(y, groups, min_n=30):
    """Conditional Dispersion Ratio -- ``Var(y | group) / Var(y)``.

    Values far from 1.0 mark subgroups treated much more, or much less,
    consistently than the roll as a whole. ``min_n`` behaves as in `cmbi`.
    """
    y = np.asarray(y, dtype=float)
    g = pd.Series(np.asarray(groups)).astype(str).values
    ok = np.isfinite(y)
    overall = np.var(y[ok])
    frame = pd.DataFrame({'group': g[ok], 'y': y[ok]})
    out = (frame.groupby('group')['y']
                .agg(n='size', variance='var')
                .reset_index())
    out['cdr'] = out['variance'] / overall if overall > 0 else np.nan
    out['reliable'] = out['n'] >= min_n
    return (out.sort_values(['reliable', 'cdr'], ascending=[False, False])
               .reset_index(drop=True))


def dcorr(x, y, max_rows=2000, random_state=0):
    """Distance correlation: zero if and only if ``x`` and ``y`` are independent.

    Model-free, and sensitive to nonlinear and non-monotone dependence that
    Pearson correlation misses -- which makes it the cheap first pass for
    deciding which characteristics deserve a model at all.

    O(n^2) in memory, so samples above ``max_rows`` are subsampled; callers
    report the reduction rather than hiding it.

    References
    ----------
    Szekely, Rizzo & Bakirov (2007), *Annals of Statistics* 35(6), 2769-2794.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10:
        return np.nan
    if n > max_rows:
        idx = np.random.default_rng(random_state).choice(n, max_rows, replace=False)
        x, y = x[idx], y[idx]

    def _centred(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()

    A, B = _centred(x), _centred(y)
    denom = np.sqrt((A * A).mean() * (B * B).mean())
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max((A * B).mean(), 0.0) / denom))


# ==========================================================================
# Model layer
# ==========================================================================

def _prepare_features(df, variables):
    """Coerce the caller's variable list into a model-ready frame.

    Nothing for the caller to configure: numerics pass through,
    low-cardinality categoricals become native pandas categories (which
    XGBoost consumes directly), and high-cardinality ones are frequency
    encoded with a warning -- one-hot would explode the feature space and
    ordinal codes would invent an order that isn't there.
    """
    X = pd.DataFrame(index=df.index)
    encoding = {}
    for col in variables:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            X[col] = s.astype(float)
            encoding[col] = 'numeric'
        else:
            levels = s.nunique(dropna=True)
            if levels <= _MAX_CARDINALITY:
                X[col] = s.astype('category')
                encoding[col] = f'category({levels})'
            else:
                X[col] = s.map(s.value_counts(normalize=True)).astype(float)
                encoding[col] = f'frequency({levels})'
                warnings.warn(
                    f"'{col}' has {levels} levels (> {_MAX_CARDINALITY}); "
                    "frequency-encoded. Frequency encoding can manufacture "
                    "apparent structure -- read its SHAP values with care.",
                    stacklevel=3,
                )
    return X, encoding


def _make_model(n, random_state, params=None):
    """Gradient-boosted regressor whose capacity comes from early stopping.

    Deliberately no hyperparameter search. For a diagnostic we need a model
    that finds learnable structure without fitting noise, and early stopping
    on a held-out slice does that directly. A tuning sweep would buy
    marginal detection power for 10-100x the runtime, and would put back the
    decisions this function exists to hide.
    """
    xgb = _optional.require('xgboost', 'ml', 'ASR predictability diagnostics')
    defaults = dict(
        n_estimators=400, learning_rate=0.05, max_depth=0,
        max_leaves=int(min(31, max(4, n // 50))),
        min_child_weight=max(5, n // 200), reg_lambda=1.0,
        subsample=0.9, colsample_bytree=0.9,
        tree_method='hist', grow_policy='lossguide',
        enable_categorical=True, early_stopping_rounds=25,
        random_state=random_state, n_jobs=0, verbosity=0,
    )
    if params:
        defaults.update(params)
    return xgb.XGBRegressor(**defaults)


def _cross_fit(X, y, n_splits, random_state, params=None):
    """Out-of-fold predictions, early stopping inside each fold.

    The early-stopping validation slice is carved from the *training* fold,
    never the held-out fold, so the out-of-fold predictions stay genuinely
    out of sample.
    """
    sk = _optional.require('sklearn.model_selection', 'ml',
                           'ASR predictability diagnostics')
    oof = np.full(len(y), np.nan)
    kf = sk.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        inner, val = sk.train_test_split(train_idx, test_size=0.15,
                                         random_state=random_state)
        model = _make_model(len(inner), random_state, params)
        model.fit(X.iloc[inner], y[inner],
                  eval_set=[(X.iloc[val], y[val])], verbose=False)
        oof[test_idx] = model.predict(X.iloc[test_idx])
    return oof


def _shap_values(X, y, random_state, params=None, sample_pct=0.20,
                 max_rows=None):
    """Exact TreeSHAP contributions from one full-data fit.

    XGBoost's ``pred_contribs`` is the same TreeSHAP algorithm the ``shap``
    package calls for tree models, in C++ -- so no extra dependency.

    Note the asymmetry with the predictability score: that comes from
    cross-fitted predictions, these from a single model fit to all rows.
    SHAP wants one stable model to explain.
    """
    xgb = _optional.require('xgboost', 'ml', 'ASR predictability diagnostics')
    sk = _optional.require('sklearn.model_selection', 'ml',
                           'ASR predictability diagnostics')
    train, val = sk.train_test_split(np.arange(len(y)), test_size=0.15,
                                     random_state=random_state)
    model = _make_model(len(train), random_state, params)
    model.fit(X.iloc[train], y[train],
              eval_set=[(X.iloc[val], y[val])], verbose=False)

    # Stratified subsample: take every k-th row of the target-sorted index,
    # so the explained rows span the whole ratio distribution instead of
    # landing wherever a random draw happens to fall.
    n_take = int(round(len(X) * sample_pct))
    if max_rows:
        n_take = min(n_take, int(max_rows))
    n_take = int(np.clip(n_take, min(200, len(X)), len(X)))
    if n_take < len(X):
        order = np.argsort(np.asarray(y))
        idx = np.sort(order[np.linspace(0, len(order) - 1, n_take).astype(int)])
    else:
        idx = np.arange(len(X))
    X_explain = X.iloc[idx]

    contribs = model.get_booster().predict(
        xgb.DMatrix(X_explain, enable_categorical=True), pred_contribs=True)
    return contribs[:, :-1], contribs[:, -1], X_explain, idx


# ==========================================================================
# Entry point
# ==========================================================================

def asr_predictability(df,
                       market_value_header,
                       assessment_header,
                       independent_variables,
                       target='asr',
                       group_by=None,
                       n_splits=3,
                       random_state=0,
                       model_params=None,
                       compute_shap=True,
                       shap_sample_pct=0.20,
                       shap_max_rows=None,
                       compute_dcorr=True,
                       dcorr_max_rows=2000,
                       min_group_n=30,
                       thresholds=None,
                       verbose=False):
    """Measure how much of the ASR is predictable from characteristics, and by what.

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
    """
    t0 = time.time()
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    pdf = nw_utils.require_pandas(df, reason='ml_diagnostics uses xgboost')

    needed = [market_value_header, assessment_header] + list(independent_variables)
    if group_by:
        needed.append(group_by)
    missing = [c for c in needed if c not in pdf.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}")
    if target not in ('asr', 'log_residual'):
        raise ValueError("target must be 'asr' or 'log_residual'")

    build = asr_target if target == 'asr' else log_residual
    y_all = build(pdf[market_value_header], pdf[assessment_header])
    keep = np.isfinite(y_all)
    dropped = int((~keep).sum())
    work = pdf.loc[keep].reset_index(drop=True)
    y = y_all[keep]
    if len(work) < 100:
        raise ValueError(
            f"only {len(work)} usable rows after dropping non-positive or "
            "missing values; these diagnostics need at least 100."
        )

    X, encoding = _prepare_features(work, independent_variables)

    # --- how much -------------------------------------------------------
    oof = _cross_fit(X, np.asarray(y, dtype=float), n_splits, random_state,
                     model_params)
    r2 = r2_correlation(y, oof)
    if verbose:
        print(f"  ASR ~ f(X): R2={r2:.4f}  ({time.time() - t0:.1f}s)")

    unit = 'ASR' if target == 'asr' else 'log-residual'
    summary = pd.DataFrame([
        {'Metric': 'Predictable share of ASR (R²)',
         'Value': f'{r2:.3f}',
         'Reading': ('the ratio carries structure tied to characteristics'
                     if r2 > thresholds['r2']
                     else 'the ratio looks close to noise w.r.t. these characteristics')},
        {'Metric': 'Unexplained share (1 - R²)',
         'Value': f'{1 - r2:.3f}',
         'Reading': 'variation no characteristic in X accounts for'},
        {'Metric': 'Cross-fitting',
         'Value': f'{n_splits}-fold out-of-fold',
         'Reading': 'in-sample R² would be ~1 for any boosted tree'},
        {'Metric': 'Target',
         'Value': f'{target} ({unit} units)',
         'Reading': 'sale price excluded as a feature by design'},
    ])

    # --- by what: SHAP --------------------------------------------------
    shap_importance = shap_vals = shap_data = None
    shap_base = None
    if compute_shap:
        shap_vals, base, shap_data, _ = _shap_values(
            X, np.asarray(y, dtype=float), random_state, model_params,
            shap_sample_pct, shap_max_rows)
        shap_base = float(np.mean(base))
        mean_abs = np.abs(shap_vals).mean(axis=0)
        total = mean_abs.sum()
        shap_importance = (pd.DataFrame({
            'feature': list(X.columns),
            'mean_abs_shap': mean_abs,
            'share': mean_abs / total if total else np.nan,
            'mean_shap': shap_vals.mean(axis=0),
        }).sort_values('mean_abs_shap', ascending=False)
          .reset_index(drop=True))
        shap_importance['rank'] = shap_importance.index + 1
        if verbose:
            print(f"  SHAP done ({time.time() - t0:.1f}s)")

    # --- by what: model-free screen -------------------------------------
    dcorr_table = None
    if compute_dcorr:
        recs = []
        for col in X.columns:
            v = (X[col].cat.codes.values if str(X[col].dtype) == 'category'
                 else pd.to_numeric(X[col], errors='coerce').values)
            d = dcorr(y, v, max_rows=dcorr_max_rows, random_state=random_state)
            recs.append({'variable': col, 'dcorr': d,
                         'flag': 'investigate' if (np.isfinite(d) and
                                                   d > thresholds['dcorr']) else 'ok'})
        dcorr_table = (pd.DataFrame(recs).sort_values('dcorr', ascending=False)
                         .reset_index(drop=True))

    # --- subgroups ------------------------------------------------------
    cmbi_cdr = None
    if group_by:
        a = cmbi(y, work[group_by], min_n=min_group_n)
        b = cdr(y, work[group_by], min_n=min_group_n)[['group', 'variance', 'cdr']]
        cmbi_cdr = a.merge(b, on='group', how='left')
        cmbi_cdr['flag'] = np.where(
            ~cmbi_cdr['reliable'], f'n < {min_group_n}',
            np.where(cmbi_cdr['cmbi'] > thresholds['cmbi'], 'investigate', 'ok'))

    elapsed = time.time() - t0
    info = {
        'engine': 'xgboost (native TreeSHAP via pred_contribs)',
        'target': target,
        'unit': unit,
        'n_rows_input': int(len(pdf)),
        'n_rows_used': int(len(work)),
        'n_rows_dropped': dropped,
        'n_features': int(X.shape[1]),
        'encoding': encoding,
        'n_splits': n_splits,
        'shap_sample_pct': shap_sample_pct,
        'shap_rows_explained': int(len(shap_data)) if shap_data is not None else 0,
        'shap_subsampled': bool(shap_data is not None and len(shap_data) < len(work)),
        'dcorr_subsampled': bool(compute_dcorr and len(work) > dcorr_max_rows),
        'shap_base_value': shap_base,
        'min_group_n': min_group_n,
        'random_state': random_state,
        'elapsed_seconds': round(elapsed, 2),
        'thresholds': thresholds,
    }
    if verbose:
        print(f"  total {elapsed:.1f}s")

    return {
        'predictability': r2,
        'r2': r2,
        'summary': summary,
        'shap_importance': shap_importance,
        'shap_values': shap_vals,
        'shap_base': shap_base,
        'shap_data': shap_data,
        'dcorr': dcorr_table,
        'cmbi_cdr': cmbi_cdr,
        'y': y,
        'oof_prediction': oof,
        'model_info': info,
        'attribution': ATTRIBUTION,
    }


# ==========================================================================
# Chart plumbing
# ==========================================================================

def _finish(fig, title, return_object, graph_filepath, show):
    """The package's standard three-way chart return."""
    import matplotlib.pyplot as plt
    from .graphs import _get_image, _showing_inline

    if return_object:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return [title, _get_image(buf)]
    if graph_filepath:
        if not graph_filepath.lower().endswith('.png'):
            graph_filepath += '.png'
        fig.savefig(graph_filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return graph_filepath
    if (not _showing_inline()) if show is None else show:
        plt.show()
    return fig


def _attribute(fig):
    fig.text(0.5, -0.02, ATTRIBUTION, ha='center', va='top', fontsize=7,
             color='#666666', style='italic')


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('rs_shap', [_LOW, '#9b5de5', _HIGH])


def _feature_numeric(col):
    """Feature values as floats, with categories mapped to their codes."""
    if str(col.dtype) == 'category':
        return col.cat.codes.values.astype(float)
    return pd.to_numeric(col, errors='coerce').values.astype(float)


def _require_shap(diag):
    if diag.get('shap_values') is None:
        raise ValueError("run asr_predictability with compute_shap=True")


def _unit(diag):
    return diag['model_info']['unit']


# ==========================================================================
# Charts
# ==========================================================================

def shap_importance_plot(diag, top_n=15, title=None, figsize=(9, 6),
                         return_object=False, graph_filepath='', show=None):
    """Mean |SHAP| per characteristic -- "by what", ranked.

    Follows the SHAP feature-importance bar in Bidanset & Wehrli (2025),
    labelled with the magnitude in target units so the bars can be read as
    "this characteristic moves the ratio by about this much".
    """
    import matplotlib.pyplot as plt
    _require_shap(diag)
    imp = diag['shap_importance'].head(top_n).iloc[::-1]
    title = title or 'SHAP Feature Importance'
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(imp['feature'], imp['mean_abs_shap'], color=_HIGH, alpha=0.9)
    span = imp['mean_abs_shap'].max()
    for y_, (v, sh) in enumerate(zip(imp['mean_abs_shap'], imp['share'])):
        ax.text(v + span * 0.01, y_, f'{v:.4f}  ({sh:.0%})',
                va='center', fontsize=8, color='#333333')
    ax.set_xlim(0, span * 1.25)
    ax.set_xlabel(f'mean(|SHAP value|)   [{_unit(diag)} units]')
    ax.set_title(title, fontsize=13, weight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _attribute(fig)
    return _finish(fig, title, return_object, graph_filepath, show)


def shap_beeswarm_plot(diag, top_n=14, title=None, figsize=(10, 7),
                       max_points=3000, return_object=False,
                       graph_filepath='', show=None):
    """SHAP value per property, per characteristic, coloured by feature value.

    The densest single view of the decomposition, following the beeswarm in
    Bidanset & Wehrli (2025). Each dot is one property. Horizontal position
    is that property's SHAP value; colour is the feature's own value, low to
    high. So a band running blue-left to red-right means high values of the
    characteristic push the ratio up, and a band with colour mixed across
    the range means the effect is not monotone.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    _require_shap(diag)

    vals, data = diag['shap_values'], diag['shap_data']
    order = diag['shap_importance']['feature'].head(top_n).tolist()[::-1]
    cols = list(data.columns)
    rng = np.random.default_rng(diag['model_info']['random_state'])

    if len(data) > max_points:
        pick = rng.choice(len(data), max_points, replace=False)
    else:
        pick = np.arange(len(data))

    title = title or 'SHAP value (impact on predicted ratio)'
    fig, ax = plt.subplots(figsize=figsize)
    cmap = _cmap()

    for row, feat in enumerate(order):
        j = cols.index(feat)
        sv = vals[pick, j]
        fv = _feature_numeric(data[feat])[pick]

        # Colour by the feature's own percentile, so a single outlier cannot
        # flatten the whole colour range.
        ok = np.isfinite(fv)
        c = np.full(len(fv), np.nan)
        if ok.sum() > 1:
            lo, hi = np.nanpercentile(fv[ok], [1, 99])
            c[ok] = np.clip((fv[ok] - lo) / (hi - lo), 0, 1) if hi > lo else 0.5

        # Vertical jitter proportional to local density, which is what gives
        # the beeswarm its shape and stops overplotting hiding the mass.
        jitter = np.zeros(len(sv))
        finite = np.isfinite(sv)
        if finite.sum() > 1:
            counts, edges = np.histogram(sv[finite], bins=60)
            slot = np.clip(np.digitize(sv, edges) - 1, 0, len(counts) - 1)
            dens = counts[slot] / max(counts.max(), 1)
            jitter = rng.uniform(-1, 1, len(sv)) * dens * 0.38

        good = finite & np.isfinite(c)
        ax.scatter(sv[good], row + jitter[good], c=c[good], cmap=cmap,
                   vmin=0, vmax=1, s=7, alpha=0.7, edgecolors='none',
                   rasterized=True)
        ax.scatter(sv[finite & ~np.isfinite(c)],
                   row + jitter[finite & ~np.isfinite(c)],
                   color='#bbbbbb', s=7, alpha=0.5, edgecolors='none')

    ax.axvline(0, color='#999999', lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(f'SHAP value (impact on model output)   [{_unit(diag)} units]')
    ax.set_title(title, fontsize=13, weight='bold')
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)

    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                        ax=ax, pad=0.015, aspect=30)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Feature value', fontsize=9)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    _attribute(fig)
    return _finish(fig, title, return_object, graph_filepath, show)


def shap_dependence_plot(diag, feature, title=None, figsize=(8.5, 5.5),
                         show_histogram=True, return_object=False,
                         graph_filepath='', show=None):
    """One characteristic's SHAP value against its own value.

    Follows the dependence panels in Bidanset & Wehrli (2025), including
    the grey histogram of the feature's distribution, which is what keeps a
    dramatic-looking swing in a sparse tail from being over-read. The red
    line is a binned median, robust to the outliers a mean would chase.
    """
    import matplotlib.pyplot as plt
    _require_shap(diag)
    vals, data = diag['shap_values'], diag['shap_data']
    if feature not in list(data.columns):
        raise ValueError(f"{feature!r} not among: {list(data.columns)}")

    j = list(data.columns).index(feature)
    col = data[feature]
    categorical = str(col.dtype) == 'category'
    x = _feature_numeric(col)
    y = vals[:, j]
    title = title or f'SHAP value for {feature}'

    fig, ax = plt.subplots(figsize=figsize)

    if show_histogram:
        # Distribution underlay on a twin axis, drawn first and pushed
        # behind, so it gives context without competing for attention.
        hax = ax.twinx()
        finite = np.isfinite(x)
        hax.hist(x[finite], bins=min(40, max(8, len(np.unique(x[finite])))),
                 color='#d9d9d9', alpha=0.85, zorder=0)
        hax.set_yticks([])
        hax.set_ylabel('')
        for sp in hax.spines.values():
            sp.set_visible(False)
        ax.set_zorder(hax.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.axhline(0, color='#999999', lw=1, ls='--')
    ax.scatter(x, y, s=8, alpha=0.4, color=_LOW, edgecolors='none',
               rasterized=True)

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 30:
        edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 13)))
        if len(edges) > 2:
            cx, cy = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = ok & (x >= lo) & (x <= hi)
                if m.sum() >= 10:
                    cx.append(np.median(x[m]))
                    cy.append(np.median(y[m]))
            if len(cx) > 2:
                ax.plot(cx, cy, color=_HIGH, lw=2, label='binned median')
                ax.legend(frameon=False, fontsize=9, loc='best')

    if categorical:
        cats = list(col.cat.categories)
        step = max(1, len(cats) // 12)
        ax.set_xticks(range(0, len(cats), step))
        ax.set_xticklabels([str(c) for c in cats[::step]], rotation=45,
                           ha='right', fontsize=8)
    ax.set_xlabel(feature)
    ax.set_ylabel(f'SHAP value for {feature}  [{_unit(diag)}]')
    ax.set_title(title, fontsize=12, weight='bold')
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _attribute(fig)
    return _finish(fig, title, return_object, graph_filepath, show)


def shap_waterfall_plot(diag, row=0, top_n=14, title=None, figsize=(9.5, 7),
                        return_object=False, graph_filepath='', show=None):
    """Decompose one property's predicted ratio into its contributions.

    Follows the waterfall in Bidanset & Wehrli (2025): starts from the base
    value ``E[f(X)]`` -- the average predicted ratio -- and adds each
    characteristic's contribution to reach ``f(x)`` for this property. This
    is the chart that answers an individual appeal: *why does the model
    think this property's ratio differs from the norm?*

    Parameters
    ----------
    row : int, default 0
        Positional index into ``diag['shap_data']``, not the original frame.
    """
    import matplotlib.pyplot as plt
    _require_shap(diag)
    vals, data = diag['shap_values'], diag['shap_data']
    if not 0 <= row < len(data):
        raise ValueError(f"row must be in [0, {len(data)}), got {row}")

    sv = vals[row]
    base = diag['shap_base']
    cols = list(data.columns)
    order = np.argsort(np.abs(sv))[::-1][:top_n]
    order = order[np.argsort(np.abs(sv[order]))]      # smallest first, bottom up

    labels, contribs = [], []
    for j in order:
        raw = data.iloc[row, j]
        shown = f'{raw:g}' if isinstance(raw, (int, float, np.number)) else str(raw)
        labels.append(f'{shown} = {cols[j]}')
        contribs.append(sv[j])
    contribs = np.array(contribs)

    # Cumulative left edges, running from the base value upward.
    omitted = sv.sum() - contribs.sum()
    start = base + omitted
    lefts, run = [], start
    for c in contribs:
        lefts.append(run)
        run += c

    title = title or 'SHAP contributions for one property'
    fig, ax = plt.subplots(figsize=figsize)

    # Reserve room for the value labels before drawing them, otherwise a bar
    # reaching the edge pushes its label into the y-axis tick labels.
    edges = np.array(lefts + [lefts[-1] + contribs[-1]] + [base])
    lo, hi = edges.min(), edges.max()
    pad = max((hi - lo), np.abs(contribs).max()) * 0.22
    ax.set_xlim(lo - pad, hi + pad)
    gap = (hi - lo + 2 * pad) * 0.012

    for i, (l, c) in enumerate(zip(lefts, contribs)):
        ax.barh(i, c, left=l, color=_HIGH if c > 0 else _LOW,
                height=0.62, alpha=0.92)
        ax.text(l + c + (gap if c > 0 else -gap), i,
                f'{c:+.4f}', va='center',
                ha='left' if c > 0 else 'right', fontsize=8, color='#333333')

    ax.axvline(base, color='#999999', lw=1, ls='--')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f'predicted {_unit(diag)}')
    ax.set_title(f"{title}\nE[f(X)] = {base:.4f}   ->   f(x) = {run:.4f}",
                 fontsize=12, weight='bold')
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _attribute(fig)
    return _finish(fig, title, return_object, graph_filepath, show)


def asr_predictability_panel(diag, top_n=12, n_dependence=4, title=None,
                             figsize=(13.5, 12), return_object=False,
                             graph_filepath='', show=None):
    """Composite: how much, then by what.

    A headline predictability figure, the importance ranking, the beeswarm,
    and dependence plots for the leading characteristics.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    _require_shap(diag)

    vals, data = diag['shap_values'], diag['shap_data']
    imp = diag['shap_importance']
    r2 = diag['r2']
    unit = _unit(diag)
    title = title or 'ASR Predictability Diagnostics'

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_dependence, height_ratios=[1.0, 1.5, 1.0],
                          hspace=0.55, wspace=0.28)
    cmap = _cmap()

    # Row 1: importance.
    ax = fig.add_subplot(gs[0, :])
    d = imp.head(top_n).iloc[::-1]
    ax.barh(d['feature'], d['mean_abs_shap'], color=_HIGH, alpha=0.9)
    span = d['mean_abs_shap'].max()
    for y_, (v, sh) in enumerate(zip(d['mean_abs_shap'], d['share'])):
        ax.text(v + span * 0.01, y_, f'{sh:.0%}', va='center', fontsize=7.5,
                color='#333333')
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel(f'mean(|SHAP value|)  [{unit}]', fontsize=9)
    ax.set_title('By what — SHAP feature importance', fontsize=11, weight='bold')
    ax.tick_params(labelsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    # Row 2: beeswarm.
    ax = fig.add_subplot(gs[1, :])
    order = imp['feature'].head(top_n).tolist()[::-1]
    cols = list(data.columns)
    rng = np.random.default_rng(diag['model_info']['random_state'])
    pick = (rng.choice(len(data), 2500, replace=False)
            if len(data) > 2500 else np.arange(len(data)))
    for r_, feat in enumerate(order):
        j = cols.index(feat)
        sv, fv = vals[pick, j], _feature_numeric(data[feat])[pick]
        ok = np.isfinite(fv)
        c = np.full(len(fv), 0.5)
        if ok.sum() > 1:
            lo, hi = np.nanpercentile(fv[ok], [1, 99])
            if hi > lo:
                c[ok] = np.clip((fv[ok] - lo) / (hi - lo), 0, 1)
        finite = np.isfinite(sv)
        jitter = np.zeros(len(sv))
        if finite.sum() > 1:
            counts, edges = np.histogram(sv[finite], bins=50)
            slot = np.clip(np.digitize(sv, edges) - 1, 0, len(counts) - 1)
            jitter = (rng.uniform(-1, 1, len(sv))
                      * (counts[slot] / max(counts.max(), 1)) * 0.36)
        ax.scatter(sv[finite], r_ + jitter[finite], c=c[finite], cmap=cmap,
                   vmin=0, vmax=1, s=5, alpha=0.65, edgecolors='none',
                   rasterized=True)
    ax.axvline(0, color='#999999', lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(f'SHAP value (impact on model output)  [{unit}]', fontsize=9)
    ax.set_title('Direction and spread — one dot per property',
                 fontsize=11, weight='bold')
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                        ax=ax, pad=0.012, aspect=28)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Feature value', fontsize=8)
    cbar.outline.set_visible(False)

    # Row 3: dependence for the leaders.
    for k, feat in enumerate(imp['feature'].head(n_dependence)):
        ax = fig.add_subplot(gs[2, k])
        j = cols.index(feat)
        x, y = _feature_numeric(data[feat]), vals[:, j]
        finite = np.isfinite(x)
        hax = ax.twinx()
        hax.hist(x[finite], bins=30, color='#e0e0e0', alpha=0.9, zorder=0)
        hax.set_yticks([])
        for sp in hax.spines.values():
            sp.set_visible(False)
        ax.set_zorder(hax.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.axhline(0, color='#999999', lw=0.8, ls='--')
        ax.scatter(x, y, s=4, alpha=0.35, color=_LOW, edgecolors='none',
                   rasterized=True)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_ylabel(f'SHAP [{unit}]', fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"{title}\nHow much — {r2:.1%} of ASR variation is predictable from "
        f"{diag['model_info']['n_features']} characteristics "
        f"({diag['model_info']['n_splits']}-fold out-of-fold R²)",
        fontsize=13, weight='bold', y=0.985)
    _attribute(fig)
    return _finish(fig, title, return_object, graph_filepath, show)
