"""
Ratio Studies Package – Blinder-Oaxaca Decomposition Module
===========================================================

Implements the Blinder-Oaxaca decomposition of ASR (assessment-to-sale
ratio) gaps for horizontal / group equity analysis in real property
assessment.

Based on:
    Petit, G. & Tedds, L. (2025).  Structural Bias in Real Property
        Assessment: A Blinder-Oaxaca Decomposition.  SSRN 5284398.
    Jann, B. (2008).  The Blinder-Oaxaca Decomposition for Linear
        Regression Models.  The Stata Journal 8(4).

References
----------
Yun, M.-S. (2005) A Simple Solution to the Identification Problem in
    Detailed Wage Decompositions.  Economic Inquiry 43 (4).

Notes
-----
*   The dependent variable is **log(ASR)** throughout, following the
    paper's convention.  When the caller supplies raw ASR the module
    takes the natural log internally.
*   The ``group_col`` variable may be binary, categorical or continuous.
    For non-binary inputs the module binarises at a user-supplied
    threshold (or the median for continuous).
*   All standard errors are computed by the analytical delta method
    that accounts for stochastic regressors (Jann 2008 §4.3).

"""

from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
import matplotlib.pyplot as plt

from . import nw_utils
from .helpers import _use_agg_if_headless

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MIN_GROUP_SIZE = 20
_ALPHA_DEFAULT = 0.05
_REFERENCE_CHOICES = frozenset(
    {"group_0", "group_1", "pooled", "reimers", "cotton", "sloczynski"}
)

logger = logging.getLogger(__name__)


# ===================================================================== #
#                        CORE DECOMPOSITION                              #
# ===================================================================== #

def _ols_group(
    y: np.ndarray,
    X: np.ndarray,
    *,
    cov_type: str = "HC1",
) -> dict:
    """
    Run OLS for a single group and return coefficients and covariance matrix.

    Parameters
    ----------
    y : ndarray
        Dependent variable (log-ASR or ASR) for the group's observations.
    X : ndarray
        Design matrix of shape (n, k) for the group's observations,
        including any constant column.
    cov_type : str, default 'HC1'
        Covariance estimator passed to ``statsmodels.api.OLS.fit``.

    Returns
    -------
    dict
        Dictionary with keys:

        * params : ndarray -- estimated OLS coefficients.
        * vcov : ndarray -- estimated coefficient covariance matrix.
        * nobs : int -- number of observations used.
        * resid : ndarray -- OLS residuals.
        * model : RegressionResultsWrapper -- fitted statsmodels object.
    """
    model = sm.OLS(y, X).fit(cov_type=cov_type)
    return {
        "params": model.params,
        "vcov":   np.array(model.cov_params()),
        "nobs":   int(model.nobs),
        "resid":  model.resid,
        "model":  model,
    }


def _deviation_contrast(
    X: np.ndarray,
    dummy_sets: Dict[str, List[int]],
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """
    Apply the Yun (2005) deviation-contrast transform to each set of
    dummies so that the detailed decomposition is invariant to the
    choice of omitted base category.

    Parameters
    ----------
    X : ndarray
        Design matrix of shape (n, k); the first column may be a
        constant.
    dummy_sets : dict
        Mapping of ``{set_name: [col_indices]}`` giving the column
        indices in ``X`` belonging to each dummy-coded categorical
        variable.

    Returns
    -------
    tuple
        Two-element tuple:

        * X_dc : ndarray -- transformed design matrix, same shape as
          ``X``.
        * dummy_sets_dc : dict -- the (unchanged) column-index mapping.
    """
    X_dc = X.copy()
    for _name, cols in dummy_sets.items():
        if len(cols) < 2:
            continue
        block = X_dc[:, cols]
        means = block.mean(axis=0)
        X_dc[:, cols] = block - means
    return X_dc, dummy_sets


def _reference_beta(
    res0: dict,
    res1: dict,
    reference: Union[str, float],
    X_full: np.ndarray,
    y_full: np.ndarray,
    group_vec: np.ndarray,
    cov_type: str = "HC1",
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Return the reference coefficient vector, its covariance, and the
    effective lambda weight for the chosen reference.

    Parameters
    ----------
    res0 : dict
        Result dict from ``_ols_group`` for group 0 (keys ``params``,
        ``vcov``, ``nobs``, ``resid``, ``model``).
    res1 : dict
        Result dict from ``_ols_group`` for group 1.
    reference : str or float
        Reference choice. One of ``'group_0'``, ``'group_1'``,
        ``'pooled'``, ``'reimers'``, ``'cotton'``, ``'sloczynski'``, or a
        float lambda in [0, 1] giving the weight on beta_1 directly.
    X_full : ndarray
        Full design matrix (both groups stacked), used to fit the pooled
        regression when ``reference='pooled'``.
    y_full : ndarray
        Full dependent-variable vector (both groups stacked), used for
        the pooled regression.
    group_vec : ndarray
        Binary group indicator (0/1) aligned with ``X_full`` /
        ``y_full``, used as an additional regressor in the pooled
        regression.
    cov_type : str, default 'HC1'
        Covariance estimator passed to statsmodels when fitting the
        pooled regression.

    Returns
    -------
    tuple
        Three-element tuple:

        * beta_star : ndarray -- reference coefficient vector.
        * vcov_star : ndarray -- covariance matrix of ``beta_star``.
        * effective_lambda : float or None -- weight on beta_1 in the
          convex combination beta_star = lambda * beta_1 +
          (1 - lambda) * beta_0. ``None`` for the pooled estimator,
          which does not follow the convex-combination form.
    """
    b0, V0, n0 = res0["params"], res0["vcov"], res0["nobs"]
    b1, V1, n1 = res1["params"], res1["vcov"], res1["nobs"]

    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        lam = float(reference)
        beta_star = lam * b1 + (1 - lam) * b0
        vcov_star = lam**2 * V1 + (1 - lam)**2 * V0
        return beta_star, vcov_star, lam

    if reference == "group_0":
        return b0.copy(), V0.copy(), 0.0

    if reference == "group_1":
        return b1.copy(), V1.copy(), 1.0

    if reference == "reimers":
        return _reference_beta(res0, res1, 0.5, X_full, y_full, group_vec, cov_type)

    if reference == "cotton":
        lam = n1 / (n0 + n1)
        return _reference_beta(res0, res1, lam, X_full, y_full, group_vec, cov_type)

    if reference == "sloczynski":
        p = n1 / (n0 + n1)
        lam = 1 - p
        return _reference_beta(res0, res1, lam, X_full, y_full, group_vec, cov_type)

    if reference == "pooled":
        X_p = sm.add_constant(
            np.column_stack([group_vec, X_full[:, 1:]]),
            has_constant="skip",
        )
        mod_p = sm.OLS(y_full, X_p).fit(cov_type=cov_type)
        # drop the group dummy (column index 1) from pooled betas
        keep = [0] + list(range(2, X_p.shape[1]))
        beta_star = mod_p.params[keep]
        vcov_star = np.array(mod_p.cov_params())[np.ix_(keep, keep)]
        return beta_star, vcov_star, None  # pooled has no simple λ

    raise ValueError(
        f"Unknown reference '{reference}'. "
        f"Choose from {sorted(_REFERENCE_CHOICES)} or a float λ."
    )


# --------------------------------------------------------------------- #
#  Main decomposition engine                                             #
# --------------------------------------------------------------------- #

def bo_decompose(
    df,  # pandas / polars / pyarrow / narwhals DataFrame
    asr_col: str,
    group_col: str,
    characteristics: List[str],
    *,
    log_asr: bool = True,
    reference: Union[str, float] = "pooled",
    threshold: Optional[float] = None,
    group_high_label: Optional[str] = None,
    group_low_label: Optional[str] = None,
    dummy_sets: Optional[Dict[str, List[str]]] = None,
    detail_groups: Optional[Dict[str, List[str]]] = None,
    alpha: float = _ALPHA_DEFAULT,
    cov_type: str = "HC1",
    threefold: bool = True,
    twofold: bool = True,
) -> dict:
    """
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
    """
    # ---------------------------------------------------------------- #
    # 0.  Copy & prepare                                                #
    # ---------------------------------------------------------------- #
    # Convert at the boundary to a pandas DataFrame; the per-column
    # operations below rely on pandas-specific dtypes/accessors.
    if not isinstance(df, pd.DataFrame):
        data = nw_utils.require_pandas(
            df, reason="bo_decompose() uses statsmodels.api.OLS and pandas slicing"
        )
    else:
        data = df.copy()
    needed = [asr_col, group_col] + list(characteristics)
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    data.dropna(subset=needed, inplace=True)

    # ---------------------------------------------------------------- #
    # 1.  Binarise the group variable if necessary                       #
    # ---------------------------------------------------------------- #
    g = data[group_col]
    nuniq = g.nunique()
    if nuniq < 2:
        raise ValueError(f"group_col '{group_col}' has < 2 unique values.")

    if nuniq == 2:
        levels = sorted(g.unique())
        data["_bo_group"] = (g == levels[1]).astype(int)
        if group_high_label is None:
            group_high_label = str(levels[1])
        if group_low_label is None:
            group_low_label = str(levels[0])
    elif np.issubdtype(g.dtype, np.number) and nuniq > 2:
        thr = threshold if threshold is not None else float(g.median())
        data["_bo_group"] = (g > thr).astype(int)
        if group_high_label is None:
            group_high_label = f"{group_col} > {thr}"
        if group_low_label is None:
            group_low_label = f"{group_col} <= {thr}"
    else:
        if threshold is None:
            raise ValueError(
                f"group_col '{group_col}' is categorical with {nuniq} "
                "levels. Provide a `threshold` value to binarise."
            )
        data["_bo_group"] = (g == threshold).astype(int)
        if group_high_label is None:
            group_high_label = str(threshold)
        if group_low_label is None:
            group_low_label = f"Not {threshold}"

    # ---------------------------------------------------------------- #
    # 2.  Dependent variable                                             #
    # ---------------------------------------------------------------- #
    if log_asr:
        data["_bo_y"] = np.log(data[asr_col].astype(float))
    else:
        data["_bo_y"] = data[asr_col].astype(float)

    # ---------------------------------------------------------------- #
    # 3.  Design matrix                                                  #
    # ---------------------------------------------------------------- #
    X_raw = data[list(characteristics)].values.astype(float)
    var_names = ["const"] + list(characteristics)
    X_full = sm.add_constant(X_raw, has_constant="add")

    # Deviation-contrast transform for dummy sets
    dc_col_map: Dict[str, List[int]] = {}
    if dummy_sets:
        for sname, scols in dummy_sets.items():
            dc_col_map[sname] = [
                var_names.index(c) for c in scols if c in var_names
            ]
        X_full, dc_col_map = _deviation_contrast(X_full, dc_col_map)

    y_full = data["_bo_y"].values
    grp    = data["_bo_group"].values

    mask1 = grp == 1
    mask0 = grp == 0

    if mask1.sum() < _MIN_GROUP_SIZE or mask0.sum() < _MIN_GROUP_SIZE:
        warnings.warn(
            f"One group has < {_MIN_GROUP_SIZE} observations; results "
            "may be unreliable."
        )

    # ---------------------------------------------------------------- #
    # 4.  Group regressions                                              #
    # ---------------------------------------------------------------- #
    res1 = _ols_group(y_full[mask1], X_full[mask1], cov_type=cov_type)
    res0 = _ols_group(y_full[mask0], X_full[mask0], cov_type=cov_type)

    b1, V1, n1 = res1["params"], res1["vcov"], res1["nobs"]
    b0, V0, n0 = res0["params"], res0["vcov"], res0["nobs"]

    X1_bar = X_full[mask1].mean(axis=0)
    X0_bar = X_full[mask0].mean(axis=0)
    dX     = X1_bar - X0_bar

    y1_bar = y_full[mask1].mean()
    y0_bar = y_full[mask0].mean()
    R      = y1_bar - y0_bar        # raw gap (group 1 − group 0)

    k = len(var_names)

    # Variance of group means  Var(X_bar) = Sigma / n
    # where Sigma = sample covariance of X within group
    SigmaX1 = np.cov(X_full[mask1].T) if n1 > 1 else np.zeros((k, k))
    SigmaX0 = np.cov(X_full[mask0].T) if n0 > 1 else np.zeros((k, k))

    # ---------------------------------------------------------------- #
    # 5.  Reference coefficients                                         #
    # ---------------------------------------------------------------- #
    beta_star, Vstar, _lam = _reference_beta(
        res0, res1, reference, X_full, y_full, grp, cov_type
    )

    # ---------------------------------------------------------------- #
    # 6.  Twofold decomposition (Jann 2008, eq 3–5)                      #
    # ---------------------------------------------------------------- #
    overall: Dict[str, Any] = {
        "R": R,
        "y1_bar": y1_bar,
        "y0_bar": y0_bar,
        "n1": n1,
        "n0": n0,
        "reference": str(reference),
        "group_high_label": group_high_label,
        "group_low_label": group_low_label,
    }

    detail_rows = []

    if twofold:
        Q_vec = dX * beta_star          # explained (per variable)
        U_vec_1 = X1_bar * (b1 - beta_star)
        U_vec_0 = X0_bar * (beta_star - b0)
        U_vec   = U_vec_1 + U_vec_0     # unexplained (per variable)

        Q_total = Q_vec.sum()
        U_total = U_vec.sum()

        # --- Delta-method SE for Q_total ---
        # Var(Q) = β*' Var(dX) β* + dX' Var(β*) dX
        VardX = SigmaX1 / max(n1, 1) + SigmaX0 / max(n0, 1)
        VarQ = beta_star @ VardX @ beta_star + dX @ Vstar @ dX
        seQ  = np.sqrt(max(VarQ, 0))

        # --- Delta-method SE for U_total ---
        # For λ-based references:
        #   U = W'(β₁ - β₀)  where W = (1-λ)X̄₁ + λX̄₀
        #   V(U) = W' [V(β₁)+V(β₀)] W
        #        + (β₁-β₀)' [(1-λ)²V(X̄₁) + λ²V(X̄₀)] (β₁-β₀)
        # For pooled: use sum-of-parts approximation.
        db = b1 - b0
        VardX1 = SigmaX1 / max(n1, 1)
        VardX0 = SigmaX0 / max(n0, 1)
        Vdb = V1 + V0  # Var(β₁ − β₀) assuming group independence

        if _lam is not None:
            W = (1 - _lam) * X1_bar + _lam * X0_bar
            VarU = (W @ Vdb @ W
                    + db @ ((1 - _lam)**2 * VardX1
                            + _lam**2 * VardX0) @ db)
        else:
            # pooled: U = X̄₁(β₁ − β*) + X̄₀(β* − β₀)
            d1s = b1 - beta_star
            ds0 = beta_star - b0
            VarU = (X1_bar @ (V1 + Vstar) @ X1_bar
                    + d1s @ VardX1 @ d1s
                    + X0_bar @ (Vstar + V0) @ X0_bar
                    + ds0 @ VardX0 @ ds0)
        seU = np.sqrt(max(float(VarU), 1e-30))

        # Per-variable SEs (diagonal approximation)
        diagVardX  = np.diag(VardX)
        diagVstar  = np.diag(Vstar)
        diagV1     = np.diag(V1)
        diagV0     = np.diag(V0)
        diagVdb    = np.diag(Vdb)
        diagVardX1 = np.diag(VardX1)
        diagVardX0 = np.diag(VardX0)

        seQ_vec = np.sqrt(beta_star**2 * diagVardX + dX**2 * diagVstar)

        # Per-variable SE for U
        if _lam is not None:
            W_vec = (1 - _lam) * X1_bar + _lam * X0_bar
            seU_vec = np.sqrt(
                W_vec**2 * diagVdb
                + db**2 * ((1 - _lam)**2 * diagVardX1
                           + _lam**2 * diagVardX0)
            )
        else:
            d1s = b1 - beta_star
            ds0 = beta_star - b0
            seU_vec = np.sqrt(
                X1_bar**2 * (diagV1 + diagVstar)
                + d1s**2 * diagVardX1
                + X0_bar**2 * (diagVstar + diagV0)
                + ds0**2 * diagVardX0
            )

        overall["Q"] = Q_total
        overall["U"] = U_total
        overall["se_Q"] = seQ
        overall["se_U"] = seU

        for j, vname in enumerate(var_names):
            detail_rows.append({
                "variable": vname,
                "Q":    Q_vec[j],
                "U":    U_vec[j],
                "se_Q": seQ_vec[j],
                "se_U": seU_vec[j],
            })

    # ---------------------------------------------------------------- #
    # 7.  Threefold decomposition (Jann 2008, eq 1)                      #
    # ---------------------------------------------------------------- #
    if threefold:
        E_vec = dX * b0               # endowments  (evaluated at β₀)
        C_vec = X0_bar * (b1 - b0)    # coefficients (evaluated at X̄₀)
        I_vec = dX * (b1 - b0)        # interaction

        E_total = E_vec.sum()
        C_total = C_vec.sum()
        I_total = I_vec.sum()

        # SEs (diagonal delta method)
        seE_vec = np.sqrt(b0**2 * np.diag(VardX) + dX**2 * np.diag(V0))
        db = b1 - b0
        Vdb = V1 + V0   # Var(b1 - b0) assuming independence
        seC_vec = np.sqrt(X0_bar**2 * np.diag(Vdb))
        seI_vec = np.sqrt(dX**2 * np.diag(Vdb) + db**2 * np.diag(VardX))

        seE = np.sqrt(max(b0 @ VardX @ b0 + dX @ V0 @ dX, 0))
        seC = np.sqrt(max(X0_bar @ Vdb @ X0_bar, 0))
        seI = np.sqrt(max(dX @ Vdb @ dX + db @ VardX @ db, 0))

        overall["E"] = E_total
        overall["C"] = C_total
        overall["I"] = I_total
        overall["se_E"] = seE
        overall["se_C"] = seC
        overall["se_I"] = seI

        for j, vname in enumerate(var_names):
            row_match = [r for r in detail_rows if r["variable"] == vname]
            if row_match:
                row_match[0].update({
                    "E": E_vec[j], "C": C_vec[j], "I": I_vec[j],
                    "se_E": seE_vec[j], "se_C": seC_vec[j], "se_I": seI_vec[j],
                })
            else:
                detail_rows.append({
                    "variable": vname,
                    "E": E_vec[j], "C": C_vec[j], "I": I_vec[j],
                    "se_E": seE_vec[j], "se_C": seC_vec[j], "se_I": seI_vec[j],
                })

    # ---------------------------------------------------------------- #
    # 8.  Build detail DataFrame                                         #
    # ---------------------------------------------------------------- #
    detail_df = pd.DataFrame(detail_rows)

    # confidence intervals
    z = sp_stats.norm.ppf(1 - alpha / 2)
    for comp in ["Q", "U", "E", "C", "I"]:
        if comp in detail_df.columns:
            se_col = f"se_{comp}"
            detail_df[f"{comp}_lo"] = detail_df[comp] - z * detail_df[se_col]
            detail_df[f"{comp}_hi"] = detail_df[comp] + z * detail_df[se_col]
            detail_df[f"{comp}_pvalue"] = 2 * (
                1 - sp_stats.norm.cdf(np.abs(detail_df[comp] / detail_df[se_col].replace(0, np.nan)))
            )

    # Aggregate to detail_groups if provided
    detail_groups_df = None
    if detail_groups:
        agg_rows = []
        for gname, gcols in detail_groups.items():
            mask_cols = detail_df["variable"].isin(gcols)
            sub = detail_df.loc[mask_cols]
            row = {"group": gname}
            for comp in ["Q", "U", "E", "C", "I"]:
                if comp in sub.columns:
                    row[comp] = sub[comp].sum()
                    se_col = f"se_{comp}"
                    row[se_col] = np.sqrt((sub[se_col] ** 2).sum())
                    row[f"{comp}_lo"] = row[comp] - z * row[se_col]
                    row[f"{comp}_hi"] = row[comp] + z * row[se_col]
            agg_rows.append(row)
        detail_groups_df = pd.DataFrame(agg_rows)

    # Overall CIs
    for comp in ["Q", "U", "E", "C", "I"]:
        if comp in overall:
            se_key = f"se_{comp}"
            overall[f"{comp}_lo"] = overall[comp] - z * overall[se_key]
            overall[f"{comp}_hi"] = overall[comp] + z * overall[se_key]
            overall[f"{comp}_pvalue"] = 2 * (
                1 - sp_stats.norm.cdf(abs(overall[comp]) / max(overall[se_key], 1e-30))
            )

    # ---------------------------------------------------------------- #
    # 9.  Pack results                                                   #
    # ---------------------------------------------------------------- #
    group_stats = {
        "group_1": {
            "label": group_high_label, "n": n1,
            "mean_log_asr": y1_bar,
            "mean_asr": float(np.exp(y_full[mask1]).mean()),
        },
        "group_0": {
            "label": group_low_label, "n": n0,
            "mean_log_asr": y0_bar,
            "mean_asr": float(np.exp(y_full[mask0]).mean()),
        },
    }

    return {
        "overall":        overall,
        "detail":         detail_df,
        "group_stats":    group_stats,
        "models":         {"group_1": res1, "group_0": res0},
        "reference":      str(reference),
        "alpha":          alpha,
        "var_names":      var_names,
        "detail_groups":  detail_groups_df,
    }


# ===================================================================== #
#                  MULTI-STRATA / MULTI-MODEL RUNNER                     #
# ===================================================================== #

def bo_multi_decomposition(
    df,  # pandas / polars / pyarrow / narwhals DataFrame
    asr_col: str,
    group_col: str,
    characteristics: List[str],
    *,
    stratify_by: Optional[Union[str, List[str]]] = None,
    references: Optional[List[Union[str, float]]] = None,
    log_asr: bool = True,
    threshold: Optional[float] = None,
    group_high_label: Optional[str] = None,
    group_low_label: Optional[str] = None,
    dummy_sets: Optional[Dict[str, List[str]]] = None,
    detail_groups: Optional[Dict[str, List[str]]] = None,
    alpha: float = _ALPHA_DEFAULT,
    cov_type: str = "HC1",
) -> dict:
    """
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
    """
    if references is None:
        references = ["group_0", "group_1", "sloczynski"]

    # ``bo_decompose`` requires a pandas DataFrame; convert at the boundary
    # so callers can pass any supported frame.
    df = nw_utils.require_pandas(df, reason="bo_multi_decomposition()")

    if stratify_by is None:
        strata_keys = ["__all__",]
        strata_frames = {"__all__": df}
    else:
        if isinstance(stratify_by, str):
            stratify_by = [stratify_by]
        grouped = df.groupby(stratify_by, observed=True)
        strata_keys = list(grouped.groups.keys())
        strata_frames = {k: grouped.get_group(k) for k in strata_keys}

    all_results: Dict[Any, Dict[str, dict]] = {}
    summary_rows = []
    baseline_rows = []

    for sk in strata_keys:
        sdf = strata_frames[sk]
        sk_label = str(sk) if sk != "__all__" else "Overall"
        all_results[sk] = {}

        for ref in references:
            try:
                dec = bo_decompose(
                    sdf, asr_col, group_col, characteristics,
                    log_asr=log_asr,
                    reference=ref,
                    threshold=threshold,
                    group_high_label=group_high_label,
                    group_low_label=group_low_label,
                    dummy_sets=dummy_sets,
                    detail_groups=detail_groups,
                    alpha=alpha,
                    cov_type=cov_type,
                )
                all_results[sk][str(ref)] = dec
                ov = dec["overall"]
                row = {
                    "stratum":   sk_label,
                    "reference": str(ref),
                    "n1":        ov["n1"],
                    "n0":        ov["n0"],
                    "y1_bar":    ov["y1_bar"],
                    "y0_bar":    ov["y0_bar"],
                    "R":         ov["R"],
                }
                for comp in ["Q", "U", "E", "C", "I"]:
                    if comp in ov:
                        row[comp]         = ov[comp]
                        row[f"se_{comp}"] = ov[f"se_{comp}"]
                        row[f"{comp}_pct_of_R"] = (
                            ov[comp] / ov["R"] * 100 if abs(ov["R"]) > 1e-12 else np.nan
                        )
                summary_rows.append(row)
            except Exception as exc:
                logger.warning("B-O decomposition failed for stratum=%s, ref=%s: %s",
                               sk_label, ref, exc)

        # baseline row (no characteristics)
        baseline_rows.append({
            "stratum":         sk_label,
            "n1":              int(sdf[sdf.get("_bo_group", sdf[group_col]) == 1].shape[0])
                               if "_bo_group" in sdf.columns
                               else int((sdf[group_col] == sdf[group_col].unique()[-1]).sum()),
            "n0":              int(sdf.shape[0]) - int(sdf[sdf.get("_bo_group", sdf[group_col]) == 1].shape[0])
                               if "_bo_group" in sdf.columns
                               else int((sdf[group_col] != sdf[group_col].unique()[-1]).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    return {
        "results":        all_results,
        "summary_table":  summary_df,
        "baseline_table": baseline_df,
    }


# ===================================================================== #
#                           FORMATTING                                   #
# ===================================================================== #

def bo_format_table(
    decomp: dict,
    *,
    proportions: bool = True,
    stars: bool = True,
    decimals: int = 4,
) -> pd.DataFrame:
    """
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
    """
    ov = decomp["overall"]
    R  = ov["R"]
    rows = []

    def _star(p):
        if not stars or np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    rows.append({
        "Component": "Raw gap (R)",
        "Estimate": R,
        "SE": np.nan,
        "p-value": np.nan,
        "% of R": 100.0,
        "Sig": "",
    })

    for comp, label in [
        ("Q", "Explained (Q)"),
        ("U", "Unexplained (U)"),
        ("E", "Endowments (E)"),
        ("C", "Coefficients (C)"),
        ("I", "Interaction (I)"),
    ]:
        if comp in ov:
            pv = ov.get(f"{comp}_pvalue", np.nan)
            rows.append({
                "Component": label,
                "Estimate": ov[comp],
                "SE": ov[f"se_{comp}"],
                "p-value": pv,
                "% of R": ov[comp] / R * 100 if abs(R) > 1e-12 else np.nan,
                "Sig": _star(pv),
            })

    out = pd.DataFrame(rows)
    if not proportions:
        out.drop(columns=["% of R"], inplace=True)
    return out


# ===================================================================== #
#                              PLOTS                                     #
# ===================================================================== #

def bo_coefficient_plot(
    decomp: dict,
    *,
    components: Optional[List[str]] = None,
    exclude_const: bool = True,
    title: str = "Blinder-Oaxaca Coefficient Plot",
    figsize: Tuple[float, float] = (8, 6),
    filepath: Optional[str] = None,
    return_object: bool = True,
) -> Optional[plt.Figure]:
    """
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
    """
    _use_agg_if_headless()
    det = decomp["detail"].copy()
    if exclude_const:
        det = det[det["variable"] != "const"].copy()

    if components is None:
        if "Q" in det.columns:
            components = ["Q", "U"]
        else:
            components = ["E", "C", "I"]

    comp_labels = {
        "Q": "Explained",
        "U": "Unexplained",
        "E": "Endowments",
        "C": "Coefficients",
        "I": "Interaction",
    }
    colors = {
        "Q": "#2166ac",
        "U": "#b2182b",
        "E": "#4393c3",
        "C": "#d6604d",
        "I": "#878787",
    }

    n_comps = len(components)
    fig, axes = plt.subplots(
        1, n_comps, figsize=figsize, sharey=True,
        constrained_layout=True,
    )
    if n_comps == 1:
        axes = [axes]

    y_pos = np.arange(len(det))
    var_labels = det["variable"].values

    for ax, comp in zip(axes, components):
        vals = det[comp].values
        lo   = det[f"{comp}_lo"].values
        hi   = det[f"{comp}_hi"].values
        xerr = np.array([vals - lo, hi - vals])

        ax.errorbar(
            vals, y_pos, xerr=xerr, fmt="o",
            color=colors.get(comp, "black"),
            ecolor=colors.get(comp, "black"),
            elinewidth=1.5, capsize=3, markersize=5,
        )
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_labels)
        ax.set_xlabel(comp_labels.get(comp, comp))
        ax.invert_yaxis()

    fig.suptitle(title)
    if filepath:
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
    if return_object:
        return fig
    plt.close(fig)
    return None


def bo_decomposition_plot(
    multi_result: dict,
    *,
    component_pair: Tuple[str, str] = ("Q", "U"),
    title: str = "Blinder-Oaxaca Decomposition by Stratum",
    figsize: Tuple[float, float] = (10, 6),
    filepath: Optional[str] = None,
    return_object: bool = True,
) -> Optional[plt.Figure]:
    """
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
    """
    _use_agg_if_headless()
    st = multi_result["summary_table"]
    if st.empty:
        warnings.warn("Summary table is empty; cannot plot.")
        return None

    comp_a, comp_b = component_pair
    if comp_a not in st.columns or comp_b not in st.columns:
        warnings.warn(f"Components {component_pair} not found in summary table.")
        return None

    ref_labels = {
        "group_0":     "Gain (β₀)",
        "group_1":     "Loss (β₁)",
        "sloczynski":  "Gap (λ=1-p)",
        "pooled":      "Pooled",
        "reimers":     "Reimers",
        "cotton":      "Cotton",
    }
    comp_labels = {"Q": "Explained", "U": "Unexplained",
                   "E": "Endowments", "C": "Coefficients", "I": "Interaction"}
    comp_colors = {"Q": "#2166ac", "U": "#b2182b",
                   "E": "#4393c3", "C": "#d6604d", "I": "#878787"}

    strata  = st["stratum"].unique()
    refs    = st["reference"].unique()
    n_strat = len(strata)
    n_ref   = len(refs)

    x = np.arange(n_strat)
    width = 0.8 / (n_ref * 2)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for j, ref in enumerate(refs):
        sub = st[st["reference"] == ref]
        lbl = ref_labels.get(ref, ref)
        for ci, comp in enumerate([comp_a, comp_b]):
            offset = (j * 2 + ci - n_ref) * width + width / 2
            vals = []
            for s in strata:
                row = sub[sub["stratum"] == s]
                vals.append(row[comp].values[0] if len(row) else 0)
            colour = comp_colors.get(comp, "gray")
            hatch  = "" if ci == 0 else "///"
            bar_label = f"{lbl} – {comp_labels.get(comp, comp)}"
            ax.bar(x + offset, vals, width, label=bar_label,
                   color=colour, alpha=0.6 + 0.3 * (1 - ci), hatch=hatch,
                   edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=30, ha="right")
    ax.set_ylabel("Contribution to log-ASR gap")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_title(title)

    if filepath:
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
    if return_object:
        return fig
    plt.close(fig)
    return None


def bo_detail_plot(
    decomp: dict,
    *,
    component: str = "Q",
    exclude_const: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
    filepath: Optional[str] = None,
    return_object: bool = True,
) -> Optional[plt.Figure]:
    """
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
    """
    _use_agg_if_headless()
    comp_labels = {"Q": "Explained", "U": "Unexplained",
                   "E": "Endowments", "C": "Coefficients", "I": "Interaction"}
    det = decomp["detail"].copy()
    if exclude_const:
        det = det[det["variable"] != "const"]

    if component not in det.columns:
        warnings.warn(f"Component '{component}' not in detail.")
        return None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    y = np.arange(len(det))
    vals = det[component].values
    lo   = det[f"{component}_lo"].values
    hi   = det[f"{component}_hi"].values
    xerr = np.array([vals - lo, hi - vals])

    ax.errorbar(vals, y, xerr=xerr, fmt="o", color="#2166ac",
                ecolor="#2166ac", elinewidth=1.5, capsize=3, markersize=5)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(det["variable"].values)
    ax.invert_yaxis()
    ax.set_xlabel(f"{comp_labels.get(component, component)} contribution")
    if title is None:
        title = f"Detail: {comp_labels.get(component, component)}"
    ax.set_title(title)

    if filepath:
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
    if return_object:
        return fig
    plt.close(fig)
    return None
