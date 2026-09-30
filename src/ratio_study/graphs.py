import logging
import re
import uuid
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import warnings
import textwrap
from io import BytesIO
import seaborn as sns
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image
from typing import Optional, Union
from ratio_study.confint import compute_median_ci
from . import nw_utils, _optional



def _format_proxy_labels(values, short=True):
    """
    Format market value proxy amounts as axis tick labels.

    Parameters
    ----------
    values : iterable of float
        Dollar amounts to format.
    short : bool, default True
        If True, abbreviate the amounts (e.g. ``'$1.23M'``, ``'$45K'``).
        If False, use full comma-formatted dollar amounts.

    Returns
    -------
    list of str
        One formatted label per entry in `values`.
    """
    if not short:
        return [f"${x:,.0f}" for x in values]

    labels = []
    for x in values:
        if x >= 1e6:
            labels.append(f"${x/1e6:.2f}M")
        elif x >= 1e3:
            labels.append(f"${int(round(x/1e3))}K")
        else:
            labels.append(f"${int(round(x))}")
    return labels


def _group_cod_pct(group):
    """
    Return the CoD of one VEI percentile group, in percent.

    Uses the ``'cod'`` entry ``vei_analysis`` stores per group (a
    fraction), and falls back to computing it from the group's ASR vector
    so results produced before that entry existed still label correctly.

    Parameters
    ----------
    group : dict
        One ``pg_stats`` entry.

    Returns
    -------
    float or None
        The group's CoD in percent, or None when it cannot be determined.
    """
    value = group.get('cod')
    if value is None:
        vector = group.get('asr_vector')
        if vector is None:
            return None
        ratios = np.asarray(vector, dtype=float)
        median = group.get('median_ratio')
        median = float(np.median(ratios)) if median is None else float(median)
        if ratios.size == 0 or median == 0:
            return None
        value = float(np.mean(np.abs(ratios - median) / median))
    return float(value) * 100.0


def _format_proxy_cod_labels(pg_stats, short=True):
    """
    Build the VEI boxplot x-axis tick labels: median proxy over group CoD.

    Each label carries the group's median market value proxy on the first
    line and its CoD on the second (e.g. ``'$290K\\n15.0%'``), so the
    dispersion behind every box is read off the axis. Groups whose CoD is
    unavailable keep the proxy amount alone.

    Parameters
    ----------
    pg_stats : sequence of dict
        Per-group statistics from ``vei_analysis``, each with
        ``'med_proxy'`` and ideally ``'cod'``.
    short : bool, default True
        Abbreviate the proxy amounts (see :func:`_format_proxy_labels`).

    Returns
    -------
    list of str
        One two-line label per entry in `pg_stats`.
    """
    proxy_labels = _format_proxy_labels([group['med_proxy'] for group in pg_stats],
                                       short=short)
    labels = []
    for proxy_label, group in zip(proxy_labels, pg_stats):
        cod_pct = _group_cod_pct(group)
        labels.append(proxy_label if cod_pct is None
                      else f"{proxy_label}\n{cod_pct:.1f}%")
    return labels


_VEI_X_LABEL = 'Median Market Value Proxy and CoD per group'

# ---------------------------------------------------------------------------------------------------------------------

# Create VEI boxplot
def vei_boxplot(asr, mkt_value_proxy=None, vei_result=None, short_labels=True, graph_filepath='', title='', return_object=False,
                *, assessment_header=None, market_value_header=None, conf_level=0.9,
                acceptable_range=10, percentile_method='R7'):
    """
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
    """
    # Frame form: derive the ratio, the proxy and the VEI analysis from the
    # sales frame, so the caller does not have to run vei_analysis first.
    if assessment_header or market_value_header or vei_result is None:
        from ratio_study.indicators import vei_analysis as _vei_analysis
        frame, ratio_col = _asr_frame(asr, assessment_header, market_value_header)
        proxy = (nw_utils.to_numpy_1d(mkt_value_proxy) if mkt_value_proxy is not None
                 else frame[market_value_header].to_numpy())
        asr = frame[ratio_col].to_numpy()
        mkt_value_proxy = proxy
        if vei_result is None:
            vei_result = _vei_analysis(asr, proxy, conf_level=conf_level,
                                       acceptable_range=acceptable_range,
                                       percentile_method=percentile_method)

    asr_arr = nw_utils.to_numpy_1d(asr)
    mv_arr = nw_utils.to_numpy_1d(mkt_value_proxy)
    sample_median = vei_result["Median"] if 'Median' in vei_result else float(np.median(asr_arr))

    order = np.argsort(mv_arr)
    sorted_asr = asr_arr[order]
    pg_stats_key = 'pg_stats' if 'pg_stats' in vei_result else 'VEI_pg_stats'
    m = len(vei_result[pg_stats_key])
    group_sizes = [x["sales"] for x in vei_result[pg_stats_key]]

    # Assign group labels
    labels = []
    for i, size in enumerate(group_sizes):
        labels.extend([i + 1] * size)
    labels = np.array(labels)

    formatted_labels = _format_proxy_cod_labels(vei_result[pg_stats_key],
                                                short=short_labels)

    pctile_type = (
        "Halves" if m == 2 else
        "Quartiles" if m == 4 else
        "Deciles" if m == 10 else
        "Groups"
    )

    plot_df = pd.DataFrame({
        "Ratio": sorted_asr,
        "Group": pd.Categorical(labels, categories=range(1, m+1), ordered=True)
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x="Group", y="Ratio", data=plot_df, color="#015da8")
    plt.axhline(y=sample_median, linestyle="--", color="red")

    if title == '':
        title = 'Vertical Equity Analysis'
    # 'Acceptable' is the passing verdict; this read 'Acceptable', which
    # `vei_analysis` never returns, so the short form never fired.
    if vei_result['conclusion'] == 'Acceptable':
        title = f"{title}: VEI is {vei_result['conclusion']}."
    else:
        title = f"{title}: VEI = {str(round(vei_result['vei'],2))}%, {vei_result['conclusion']}."

    if not return_object:
        plt.title(f"{title}")

    plt.xlabel(_VEI_X_LABEL)
    plt.ylabel("ASR")
    
    ax.set_xticks(range(m))
    ax.set_xticklabels(formatted_labels, rotation=45, ha="right")

    # Appendix E.5: R6 and R7 can assign boundary sales to different groups and
    # produce different VEIs, so the chart records which one drew these boxes.
    # Bottom right, small and grey -- provenance, not a finding.
    _method = str(vei_result.get('percentile_method') or percentile_method).upper()
    _note = f'{m} quantile groups, {_method} ranking'
    if vei_result.get('compare_all_buckets'):
        _note += (f" | widest-gap pair: {vei_result.get('group_compared_low')}"
                  f"-{vei_result.get('group_compared_high')}")
    ax.text(0.992, 0.02, _note, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#666666', style='italic',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))

    plt.tight_layout()
    if return_object:
        imgdata = BytesIO()
        plt.gcf().savefig(imgdata, format='png', bbox_inches='tight')
        plt.close()
        imgdata.seek(0)
        image_obj = _get_image(imgdata, width=7 * inch)
        title_obj = f"{title}"

        return [title_obj, image_obj]
    
    elif graph_filepath:
        if graph_filepath[-4:] != '.png':
            graph_filepath = graph_filepath + '.png'
        plt.savefig(graph_filepath)
        plt.close()
        return graph_filepath
    
    else:
        plt.show()
        return plt.gcf()


# ---------------------------------------------------------------------------------------------------------------------

def _reg_graph(df, col_x, col_y=None, y_label='', x_label='', x_ticks=None, title='', x_is_discrete=False, histogram=False):
    """
    Make a scatterplot, boxplot, or histogram from a DataFrame according to
    the given options. Modified version of James Gilbert's function.

    Parameters
    ----------
    df : DataFrame
        pandas/polars/pyarrow/narwhals DataFrame with data.
    col_x : str
        Column header name for the x axis.
    col_y : str, optional
        Column header name for the y axis. Not required when `histogram`
        is True.
    y_label : str, default ''
        Y axis label for the graph.
    x_label : str, default ''
        X axis label for the graph.
    x_ticks : str, optional
        Column header name used to derive custom x-axis tick labels for
        the scatterplot (binned by `col_x`).
    title : str, default ''
        Graph title.
    x_is_discrete : bool, default False
        If True, a boxplot is generated; otherwise a scatterplot (or
        histogram, if `histogram` is True) is generated.
    histogram : bool, default False
        If True (and `x_is_discrete` is False), a histogram of `col_x` is
        generated instead of a scatterplot.

    Returns
    -------
    reportlab.platypus.Image
        ReportLab Image object embedding the rendered seaborn plot as PNG.
    """
    try:
        # seaborn requires a real pandas DataFrame. Coerce at the boundary.
        df = nw_utils.require_pandas(df, reason="_reg_graph() uses seaborn")
        sns.set_theme(style="darkgrid", color_codes=True, )
        df = df.copy()
        if not x_is_discrete:
            df = df[df[col_x] != 0]
        if not df.empty and len(df) > 1:
            custom_fig_size = (12, 10)
            if not x_is_discrete and not histogram:  # scatterplot
                sns.set_theme(style="darkgrid")
                hist_bins = len(df[col_x].drop_duplicates()) if len(df[col_x].drop_duplicates()) < 70 else 70

                try:  # required seaborn 0.13.2 or above, for lowess line in joint plot.                    #
                    if x_ticks and x_ticks in df:
                        if len(df[x_ticks].drop_duplicates()) <= 8:
                            bin_edges = list(df.groupby(x_ticks)[col_x].first())
                        else:
                            bin_edges = list(pd.cut(df[col_x], bins=6, retbins=True, include_lowest=True)[1].astype(int))
                        bin_edges[0] = df[col_x].min()
                        df[col_x + '_xticks'] = np.where(df[col_x].isin(bin_edges), df[col_x], 0)
                        df[x_ticks] = np.where(df[col_x + '_xticks'] == df[col_x], df[x_ticks], '')
                        scatter_1 = sns.jointplot(x=col_x, y=col_y, data=df, kind="scatter", height=10, marginal_kws=dict(bins=hist_bins, kde=True), s=15)
                        sns.regplot(x=col_x, y=col_y, data=df, scatter=False, ax=scatter_1.ax_joint, lowess=True, line_kws={'color': '#ff9933'})
                        scatter_1.ax_joint.set_xticks(df[col_x + '_xticks'].drop_duplicates())
                        scatter_1.ax_joint.set_xticklabels(df[x_ticks].drop_duplicates())
                        scatter_1.figure.suptitle(title, fontsize=20, weight='bold', y=1.02)
                        scatter_1.ax_joint.set_xlabel(x_label, fontsize=16, weight='bold')
                        scatter_1.ax_joint.set_ylabel(y_label, fontsize=16, weight='bold')
                        df.drop(columns=[col_x + '_xticks'], inplace=True)

                    else:
                        scatter_1 = sns.JointGrid(data=df, x=col_x, y=col_y, height=10)
                        scatter_1.plot_joint(sns.regplot, lowess=True, line_kws={'color': '#ff9933'}, scatter_kws=dict(s=15))
                        scatter_1.plot_marginals(sns.histplot, kde=True)
                        scatter_1.set_axis_labels(x_label, y_label, fontsize=16, fontweight='bold')
                        scatter_1.figure.suptitle(title, fontsize=20, weight='bold', y=1.02)

                except:  # workaround for seaborn below 0.13.2, lowess line not available in joint plot

                    if x_ticks and x_ticks in df:
                        if len(df[x_ticks].drop_duplicates()) <= 8:
                            bin_edges = list(df.groupby(x_ticks)[col_x].first())
                        else:
                            bin_edges = list(pd.cut(df[col_x], bins=6, retbins=True, include_lowest=True)[1].astype(int))
                        bin_edges[0] = df[col_x].min()
                        df[col_x + '_xticks'] = np.where(df[col_x].isin(bin_edges), df[col_x], 0)
                        df[x_ticks] = np.where(df[col_x + '_xticks'] == df[col_x], df[x_ticks], '')
                        scatter_1 = sns.jointplot(x=col_x, y=col_y, data=df, kind="scatter", height=10, marginal_kws=dict(bins=hist_bins, kde=True), s=15)
                        sns.regplot(x=col_x, y=col_y, data=df, scatter=False, ax=scatter_1.ax_joint, lowess=True, line_kws={'color': '#ff9933'})
                        scatter_1.ax_joint.set_xticks(df[col_x + '_xticks'].drop_duplicates())
                        scatter_1.ax_joint.set_xticklabels(df[x_ticks].drop_duplicates())
                        scatter_1.figure.suptitle(title, fontsize=20, weight='bold', y=1.02)
                        scatter_1.ax_joint.set_xlabel(x_label, fontsize=16, weight='bold')
                        scatter_1.ax_joint.set_ylabel(y_label, fontsize=16, weight='bold')
                        plt.tick_params(axis="both", labelsize=14)
                        df.drop(columns=[col_x + '_xticks'], inplace=True)

                    else:
                        scatter_1 = sns.jointplot(x=col_x, y=col_y, data=df, kind="reg", height=10, joint_kws={'line_kws': {'color': '#ff9933'}}, scatter_kws=dict(s=15))
                        scatter_1.figure.suptitle(title, fontsize=20, weight='bold', y=1.02)
                        scatter_1.set_axis_labels(x_label, y_label, fontsize=16, fontweight='bold')

                scatter_1.figure.set_size_inches(custom_fig_size)
                imgdata = BytesIO()
                scatter_1.savefig(imgdata, format='png', )  # dpi=400)
                imgdata.seek(0)
                result = _get_image(imgdata, width=7 * inch)
                plt.clf()
                return result

            elif not histogram:  # boxplot
                box_1 = sns.boxplot(x=col_x, y=col_y, data=df, color='#8080ff', linewidth=2)
                plt.tight_layout()
                box_1.set_title(title, fontsize=20, weight='bold')
                box_1.axhline(y=0.95, c="red", linewidth=2, zorder=2)
                box_1.axhline(y=1.05, c="red", linewidth=2, zorder=2)
                plt.xlabel(x_label, fontsize=16, weight='bold')
                plt.ylabel(y_label, fontsize=16, weight='bold')
                plt.tick_params(axis="both", labelsize=14)
                imgdata = BytesIO()
                box_1.get_figure().savefig(imgdata, format='png', bbox_inches='tight')
                imgdata.seek(0)
                return _get_image(imgdata, width=7 * inch)

            else: #histogram
                hist_bins = len(df[col_x].drop_duplicates()) if len(df[col_x].drop_duplicates()) < 100 else 100
                hist = sns.histplot(data = df, x=col_x, bins=hist_bins, color='#8080ff')
                hist.set_title(title, fontsize=20, weight='bold')
                plt.xlabel(x_label, fontsize=16, weight='bold')
                plt.subplots_adjust(top=0.95)  # Adjust the value as needed
                imgdata = BytesIO()
                hist.get_figure().savefig(imgdata, format='png', bbox_inches='tight')
                imgdata.seek(0)
                return _get_image(imgdata, width=7 * inch)

    except Exception as ex:
        raise RuntimeError('error in ratio study: reg_graph', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _get_image(path, width=7 * inch, max_width=504, max_height=650, save_within_page_limits=True):
    """
    Process an image for ReportLab, resizing it to fit within page limits.
    Credit to James Gilbert.

    Parameters
    ----------
    path : str or BytesIO
        Path or buffer of the image to process.
    width : float, default 7 * inch
        Desired image width.
    max_width : float, default 504
        Maximum allowed image width.
    max_height : float, default 650
        Maximum allowed image height.
    save_within_page_limits : bool, default True
        If True, constrain `width` (and, consequently, height) so the
        image fits within `max_width` and `max_height`.

    Returns
    -------
    reportlab.platypus.Image
        ReportLab Image object sized to fit within the given constraints.
    """
    try:
        if save_within_page_limits:
            width=max_width if width> max_width else width
        img = ImageReader(path)
        iw, ih = img.getSize()
        aspect = ih / float(iw)
        height = (width * aspect)
        if height > max_height and save_within_page_limits:
            width = math.floor(width*(max_height/height))
        return Image(path, width=width, height=(width * aspect), lazy=2)
    except Exception as ex:
        raise RuntimeError('error in ratio study: _get_image', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _validate_inputs(
    df,  # pandas/polars/pyarrow/narwhals DataFrame
    market_value_proxy: str,
    asr: str,
    conf_level: float,
    var_measure: str,
    quantiles: Optional[int],
    window_size: Optional[int],
    band_method: str = "quantile",
    x_scale: str = "log",
    y_scale: str = "linear",
    asr_col='asr'
) -> None:
    if not nw_utils.is_supported_frame(df):
        raise TypeError("`df` must be a pandas, polars, or narwhals DataFrame.")

    cols = nw_utils.frame_columns(df)
    for col, label in [
        (market_value_proxy, "market_value_proxy"),
        (asr, asr_col),
    ]:
        if col not in cols:
            raise ValueError(f"Column '{col}' not found in DataFrame ({label}).")
        col_vals = nw_utils.to_numpy_1d(df[col])
        if not np.issubdtype(col_vals.dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric.")

    # Shared with `indicators._validate_inputs` and `ci_conf_level`, so the
    # percentage spelling works here too and a significance level passed by
    # mistake is rejected rather than drawn as a 5% band.
    from ratio_study.confint import confidence_level_as_fraction
    if conf_level is not None:
        confidence_level_as_fraction(conf_level, 'conf_level')

    if var_measure not in ("sd", "mae", "cod"):
        raise ValueError("`var_measure` must be one of 'sd', 'mae', 'cod'.")

    if band_method not in ("median_ci", "quantile"):
        raise ValueError("`band_method` must be 'median_ci' or 'quantile'.")

    if x_scale not in ("linear", "log"):
        raise ValueError("`x_scale` must be 'linear' or 'log'.")
    if y_scale not in ("linear", "log"):
        raise ValueError("`y_scale` must be 'linear' or 'log'.")

    if quantiles is None and window_size is None:
        raise ValueError(
            "At least one of `quantiles` or `window_size` must be provided."
        )

    if quantiles is not None and window_size is not None:
        warnings.warn(
            "Both `quantiles` and `window_size` provided; `quantiles` will be used.",
            UserWarning,
            stacklevel=3,
        )

    # Backend-agnostic uniqueness check.
    for col, label in [
        (asr, asr_col),
        (market_value_proxy, "market_value_proxy"),
    ]:
        col_vals = nw_utils.to_numpy_1d(df[col])
        unique_count = len(np.unique(col_vals[~np.isnan(col_vals)]))
        if unique_count < 2:
            if col == asr:
                raise ValueError(f"ASR column '{col}' is constant; cannot compute equity.")
            else:
                raise ValueError(
                    f"Market value proxy column '{col}' is constant."
                )

# ---------------------------------------------------------------------------------------------------------------------

def _var_func(series, var_measure: str) -> float:
    """
    Compute a scalar variability metric for a 1-D series.

    Parameters
    ----------
    series : array-like
        1-D array-like of values (numpy/pandas/polars/narwhals Series).
    var_measure : str
        Variability measure to compute. One of:

        * 'sd' -- sample standard deviation.
        * 'mae' -- mean absolute error (mean absolute deviation from the
          mean).
        * 'cod' -- coefficient of dispersion (mean absolute deviation from
          the median, divided by the median).

    Returns
    -------
    float
        The computed variability metric, or NaN if the series is empty or
        (for 'cod') the median is zero.
    """
    s = nw_utils.to_numpy_1d(series).astype(float)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return np.nan
    if var_measure == "sd":
        return float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if var_measure == "mae":
        return float(np.mean(np.abs(s - s.mean())))
    # cod
    asr_md = float(np.median(s))
    if asr_md == 0:
        return np.nan
    return float(np.mean(np.abs(s - asr_md) / abs(asr_md)))


# # ---------------------------------------------------------------------------------------------------------------------
# # Matplotlib static renderer (used when plot_return_object=False)
# # Avoids Kaleido/Orca and any browser subprocess entirely.

# def _plotly_to_mpl(
#     data: pd.DataFrame,
#     plot_title: str,
#     plot_ref: Optional[str],
#     plot_filepath: Optional[str],
#     asr_col='asr'
# ):
#     """
#     Render the equity chart with matplotlib's Agg backend.

#     Avoids Kaleido/Orca and any browser subprocess entirely; used when
#     `plot_return_object` is False.

#     Parameters
#     ----------
#     data : pandas.DataFrame
#         Processed frame with columns: mv, asr, ci_median, lower_band,
#         upper_band, out_of_bounds, inequity.
#     plot_title : str
#         Chart title.
#     plot_ref : str or None
#         Optional annotation text (e.g. data source).
#     plot_filepath : str or None
#         If provided, the figure is saved to this path and the path is
#         returned. If None, the figure is returned as an in-memory PNG
#         buffer.

#     Returns
#     -------
#     io.BytesIO or str
#         * If `plot_filepath` is None, a ``BytesIO`` buffer containing the
#           PNG image (plug straight into ReportLab's ``Image()``).
#         * If `plot_filepath` was provided, the filepath (str) where the
#           image was saved.
#     """
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from io import BytesIO

#     in_bounds  = data["out_of_bounds"] == 0
#     out_bounds = data["out_of_bounds"] != 0

#     fig, ax = plt.subplots(figsize=(10, 5))

#     # ASR scatter
#     ax.scatter(data.loc[in_bounds,  "mv"], data.loc[in_bounds,  asr_col],
#                color="green", s=8, alpha=0.55, label="ASR (within bounds)", zorder=2)
#     ax.scatter(data.loc[out_bounds, "mv"], data.loc[out_bounds, asr_col],
#                color="red",   s=8, alpha=0.70, label="ASR (out of bounds)",  zorder=2)

#     # CI lines
#     ax.plot(data["mv"], data["ci_median"],  color="black", linewidth=2,
#             label="Median ASR")
#     ax.plot(data["mv"], data["upper_band"], color="black", linewidth=1.5,
#             linestyle="--", label="CI bands")
#     ax.plot(data["mv"], data["lower_band"], color="black", linewidth=1.5,
#             linestyle="--")
#     ax.fill_between(data["mv"], data["lower_band"], data["upper_band"],
#                     color="black", alpha=0.05)

#     # Inequity
#     ax.plot(data["mv"], data["inequity"], color="grey", linewidth=2,
#             label="Inequity")

#     # Zero reference
#     ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.4)

#     ax.set_title(plot_title, fontsize=14)
#     ax.set_xlabel("Market Value Proxy")
#     ax.set_ylabel("Centred ASR")
#     ax.legend(loc="upper right", fontsize=8)

#     if plot_ref:
#         ax.annotate(plot_ref, xy=(0.01, 0.99), xycoords="axes fraction",
#                     fontsize=8, color="dimgrey", va="top")

#     plt.tight_layout()

#     if plot_filepath:
#         fig.savefig(plot_filepath, dpi=150)
#         plt.close(fig)
#         return plot_filepath
#     else:
#         buf = BytesIO()
#         fig.savefig(buf, format="png", dpi=150)
#         plt.close(fig)
#         buf.seek(0)
#         return buf


def _ve_display_center(data, center=None, asr_col='asr') -> float:
    """
    Resolve the offset that puts a ``vem`` frame back into ASR units.

    ``vem`` centres the ASR on the jurisdiction-wide median before taking
    the moving median, so 'moving_median' and the CI columns are offsets
    from that median rather than ratios. The charts add this offset back
    so the y-axis reads in ASR units.

    Parameters
    ----------
    data : pandas DataFrame
        Working frame from :func:`ratio_study.indicators.vem`.
    center : float, optional
        The offset, i.e. the ``'asr_center'`` value returned by ``vem``.
        When None it is recovered from the frame: the median of 'asr_raw'
        if present, otherwise the median of ``asr_col`` (which is 0 for a
        centred frame, leaving the chart on the centred scale).
    asr_col : str, default 'asr'
        Name of the centred ASR column.

    Returns
    -------
    float
        The offset to add to the centred quantities.
    """
    if center is not None and not (isinstance(center, float) and math.isnan(center)):
        return float(center)
    if 'asr_raw' in data.columns:
        return float(np.nanmedian(np.asarray(data['asr_raw'], dtype=float)))
    return float(np.nanmedian(np.asarray(data[asr_col], dtype=float)))

# ---------------------------------------------------------------------------------------------------------------------

def _ve_display_asr(data, center: float, asr_col='asr'):
    """
    Return the per-record ASR to scatter, in ASR units.

    Prefers the untouched 'asr_raw' column written by
    :func:`ratio_study.indicators.vem`; falls back to un-centring
    ``asr_col`` with ``center`` for frames that do not carry it.

    Parameters
    ----------
    data : pandas DataFrame
        Working frame from :func:`ratio_study.indicators.vem`.
    center : float
        Offset from :func:`_ve_display_center`.
    asr_col : str, default 'asr'
        Name of the centred ASR column.

    Returns
    -------
    pandas.Series
        The ASR in ASR units.
    """
    if 'asr_raw' in data.columns:
        return data['asr_raw']
    return data[asr_col] + center

# ---------------------------------------------------------------------------------------------------------------------

def _ve_smart_limits(data,
                     center: float,
                     asr_col='asr',
                     acceptable_inequity_level=None,
                     x_scale='log',
                     x_quantiles=(0.01, 0.99),
                     y_coverage=0.95):
    """
    Pick axis limits that frame the bulk of the VEM scatter.

    A sale-price distribution has a long right tail and the ASR has
    outliers at both ends, so the full data range spends most of the plot
    area on a handful of points. This trims both axes to where the points
    actually are: the x-axis to an inner quantile range of the market
    value proxy, and the y-axis to `y_coverage` of the ASR values inside
    that x window. The moving median, its CI band and the
    acceptable-inequity lines are always kept in view, so zooming can
    never hide the statistic the chart is about.

    Parameters
    ----------
    data : pandas DataFrame
        Working frame from :func:`ratio_study.indicators.vem`.
    center : float
        Offset from :func:`_ve_display_center`, i.e. the median ASR.
    asr_col : str, default 'asr'
        Name of the centred ASR column.
    acceptable_inequity_level : float, optional
        Acceptable inequity threshold, so its reference lines stay
        visible.
    x_scale : str, default 'log'
        X-axis scale. On 'log' the padding is multiplicative and
        non-positive values are dropped.
    x_quantiles : tuple, default (0.01, 0.99)
        Lower/upper quantiles of the market value proxy to show.
    y_coverage : float, default 0.95
        Share of the in-window ASR values to cover, centred on the
        median: 0.95 spans the 2.5th to the 97.5th percentile.

    Returns
    -------
    tuple
        ``(xlim, ylim)`` as ``(low, high)`` pairs, or ``(None, None)``
        when no usable point is left to measure.
    """
    mv = np.asarray(data['mv'], dtype=float)
    asr = np.asarray(_ve_display_asr(data, center, asr_col=asr_col), dtype=float)
    usable = np.isfinite(mv) & np.isfinite(asr)
    if x_scale == 'log':
        usable &= mv > 0
    if not usable.any():
        return None, None

    mv_ok, asr_ok = mv[usable], asr[usable]
    x_lo, x_hi = (float(value) for value in np.quantile(mv_ok, x_quantiles))
    if x_hi <= x_lo:
        x_lo, x_hi = float(mv_ok.min()), float(mv_ok.max())
    if x_scale == 'log':
        # Pad outward by a fixed share of the visible span, which on a log
        # axis means multiplying rather than adding.
        pad = (x_hi / x_lo) ** 0.02 if x_lo > 0 and x_hi > x_lo else 1.0
        xlim = (x_lo / pad, x_hi * pad)
    else:
        pad = 0.02 * (x_hi - x_lo)
        xlim = (x_lo - pad, x_hi + pad)

    inside = (mv_ok >= x_lo) & (mv_ok <= x_hi)
    if not inside.any():
        inside = np.ones_like(mv_ok, dtype=bool)
    tail = max(0.0, (1.0 - y_coverage) / 2.0)
    y_lo, y_hi = (float(value) for value in
                  np.quantile(asr_ok[inside], [tail, 1.0 - tail]))

    # Keep the reference lines and the statistic itself on screen even
    # when the scatter is tighter than they are.
    keep = [center]
    if acceptable_inequity_level:
        keep += [center - acceptable_inequity_level,
                 center + acceptable_inequity_level]
    for column in ('moving_median', 'median_ci_lower', 'median_ci_upper'):
        if column not in data.columns:
            continue
        values = np.asarray(data[column], dtype=float)[usable][inside]
        values = values[np.isfinite(values)]
        if values.size:
            keep += [float(values.min()) + center, float(values.max()) + center]

    y_lo, y_hi = min([y_lo] + keep), max([y_hi] + keep)
    y_pad = 0.05 * (y_hi - y_lo) or 0.02
    return xlim, (y_lo - y_pad, y_hi + y_pad)

# ---------------------------------------------------------------------------------------------------------------------

def _ve_y_range(zoom_y, center: float, smart_ylim):
    """
    Resolve the VEM y-axis range from the zoom settings.

    An explicit ``(low, high)`` pair always wins, then the smart-zoom
    range, then the legacy ``zoom_y=True`` window of the median ASR
    +/- 0.3. Everything is in ASR units.

    Parameters
    ----------
    zoom_y : bool, list, or tuple
        The chart's `zoom_y` argument.
    center : float
        Offset from :func:`_ve_display_center`, i.e. the median ASR.
    smart_ylim : tuple or None
        Range from :func:`_ve_smart_limits`, or None when smart zoom is
        off or could not be computed.

    Returns
    -------
    tuple or None
        ``(low, high)``, or None to leave the axis auto-ranged.
    """
    if isinstance(zoom_y, (list, tuple)):
        return tuple(float(value) for value in zoom_y)
    if smart_ylim is not None:
        return smart_ylim
    if zoom_y is True:
        return (center - 0.3, center + 0.3)
    return None

# ---------------------------------------------------------------------------------------------------------------------

def _ve_x_label(x_scale: str) -> str:
    """
    Axis label for the market value proxy, naming the scale in use.

    Parameters
    ----------
    x_scale : str
        X-axis scale, 'linear' or 'log'.

    Returns
    -------
    str
        The x-axis label.
    """
    return 'Value (log scale)' if x_scale == 'log' else 'Value'

# ---------------------------------------------------------------------------------------------------------------------

def ve_scat_median_plot(
    data,  # pandas / polars / narwhals DataFrame
    plot_title: str,
    plot_ref: Optional[str],
    x_scale: str = "log",
    y_scale: str = "linear",
    acceptable_inequity_level: float = 0.1,
    asr_col='asr',
    zoom_y: Union[bool, list, tuple] = False,
    conf_level: Optional[float] = None,
    center: Optional[float] = None,
    smart_zoom: bool = True,
) -> "go.Figure":
    """
    Build the Plotly vertical-equity moving-median chart.

    Parameters
    ----------
    data : DataFrame
        pandas/polars/narwhals DataFrame with columns: mv, asr,
        moving_median, out_of_bounds, and optionally asr_raw,
        median_ci_lower / median_ci_upper.
    plot_title : str
        Chart title.
    plot_ref : str or None
        Optional annotation text (e.g. data source).
    x_scale : str, default 'log'
        X-axis scale, 'linear' or 'log'.
    y_scale : str, default 'linear'
        Y-axis scale, 'linear' or 'log'.
    acceptable_inequity_level : float, default 0.1
        Acceptable inequity threshold, drawn as red dashed reference lines
        at the median ASR plus/minus this value. If None, the reference
        lines are omitted.
    zoom_y : bool, list, or tuple, default False
        Restricts the y-axis range to zoom in on the acceptable-inequity
        band, clipping outlier points. ``False`` leaves the y-axis
        auto-ranged; ``True`` zooms to the median ASR +/- 0.3; a
        ``(low, high)`` list/tuple sets a custom range, in ASR units.
        A list/tuple always wins; otherwise `smart_zoom` takes
        precedence over ``True``.
    conf_level : float, optional
        Confidence level used to compute 'median_ci_lower' /
        'median_ci_upper' (only used for the legend label). If those
        columns are present in ``data``, a blue confidence band is drawn
        around the moving median line regardless of whether this is
        given; it only controls the label text.
    center : float, optional
        The overall median ASR that ``vem`` subtracted, i.e. its
        ``'asr_center'`` result. Everything ``vem`` computes on the
        centred scale -- the moving median, its CI band and the
        acceptable-inequity bands -- is shifted back by this amount so
        the chart reads in ASR units (around the median, typically ~1)
        instead of around 0. Recovered from the frame's 'asr_raw' column
        when omitted.
    smart_zoom : bool, default True
        Frame both axes on the bulk of the scatter instead of its full
        range: the x-axis on the 1st-99th percentile of the market value
        proxy, and the y-axis on 95% of the ASR values inside that
        window (see :func:`_ve_smart_limits`). Low- and high-value
        outliers fall outside the view; the moving median, its CI band
        and the acceptable-inequity lines never do. Set False to plot
        the full range and fall back to `zoom_y`.

    Returns
    -------
    plotly.graph_objects.Figure
        The assembled vertical-equity moving-median chart.
    """
    has_ci_band = "median_ci_lower" in data.columns and "median_ci_upper" in data.columns
    # Plotly accepts numpy arrays directly; the data passed in here was
    # already converted to a pandas DataFrame in ``vem``/``indicators``,
    # Plotly is an optional dependency: it is ~167 MB and this is the only
    # chart that needs it. The static matplotlib counterpart inside
    # `vertical_equity_panel` covers the same ground without it.
    go = _optional.require(
        'plotly.graph_objects', 'interactive',
        've_scat_median_plot (the interactive VEM chart)')

    # so the columns below work uniformly.
    in_bounds  = data["out_of_bounds"] == 0
    out_bounds = data["out_of_bounds"] != 0

    # The indices are computed on the median-centred ASR; the chart is
    # drawn in ASR units, so every centred quantity is shifted back by
    # the overall median.
    center = _ve_display_center(data, center, asr_col=asr_col)
    asr_display = _ve_display_asr(data, center, asr_col=asr_col)
    moving_median_display = data["moving_median"] + center

    fig = go.Figure()

    # ASR scatter – local moving median within the acceptable level (green)
    fig.add_trace(go.Scatter(
        x=data.loc[in_bounds, "mv"],
        y=asr_display[in_bounds],
        mode="markers",
        marker=dict(color="green", size=2, opacity=0.55),
        name=f"{asr_col.upper()} (acceptable local median)",
    ))

    # ASR scatter – local moving median outside the acceptable level (red)
    fig.add_trace(go.Scatter(
        x=data.loc[out_bounds, "mv"],
        y=asr_display[out_bounds],
        mode="markers",
        marker=dict(color="red", size=2, opacity=0.7),
        name=f"{asr_col.upper()} (excessive local median)",
    ))

    # Moving median line (black, the statistic behind the index)
    fig.add_trace(go.Scatter(
        x=data["mv"],
        y=moving_median_display,
        mode="lines",
        line=dict(color="black", width=2),
        name=f"Moving Median {asr_col.upper()}",
    ))

    # Confidence band around the moving median (blue), when present.
    if has_ci_band:
        band_label = f"{conf_level * 100:.0f}% CI" if conf_level else "CI"

        fig.add_trace(go.Scatter(
            x=data["mv"],
            y=data["median_ci_upper"] + center,
            mode="lines",
            line=dict(color="blue", width=1),
            name=f"Upper {band_label} for median",
        ))

        fig.add_trace(go.Scatter(
            x=data["mv"],
            y=data["median_ci_lower"] + center,
            mode="lines",
            line=dict(color="blue", width=1),
            fill="tonexty",
            fillcolor="rgba(0,90,220,0.15)",
            name=f"Lower {band_label} for median",
        ))

    # Overall median ASR – the level the moving median is compared against
    fig.add_hline(
        y=center,
        line=dict(color="rgba(0,0,0,0.25)", dash="dot", width=1),
        annotation=dict(text=f"Median {asr_col.upper()}: {center:.3f}"),
        annotation_position="bottom left",
    )

    if acceptable_inequity_level is not None:
        fig.add_hline(
            y=center + acceptable_inequity_level,
            line=dict(color="red", dash="dash", width=2),
            annotation=dict(text=f"Acceptable Inequity Higher Band: median +{acceptable_inequity_level*100:g}%"),
        )

        fig.add_hline(
            y=center - acceptable_inequity_level,
            line=dict(color="red", dash="dash", width=2),
            annotation=dict(text=f"Acceptable Inequity Lower Band: median -{acceptable_inequity_level*100:g}%"),
        )

    annotations = []
    if plot_ref:
        annotations.append(dict(
            text=plot_ref,
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=10, color="dimgrey"),
            align="left",
        ))

    smart_xlim, smart_ylim = (
        _ve_smart_limits(data, center, asr_col=asr_col,
                         acceptable_inequity_level=acceptable_inequity_level,
                         x_scale=x_scale)
        if smart_zoom else (None, None)
    )

    yaxis = dict(title=asr_col.upper(), type=y_scale)
    y_range = _ve_y_range(zoom_y, center, smart_ylim)
    if y_range is not None:
        yaxis["range"] = list(y_range)

    xaxis = dict(title=_ve_x_label(x_scale), type=x_scale)
    if smart_xlim is not None:
        # A log axis takes its range in log10 units.
        xaxis["range"] = ([math.log10(bound) for bound in smart_xlim]
                          if x_scale == "log" else list(smart_xlim))

    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=16), x=0.5, xanchor="center"),
        xaxis=xaxis,
        yaxis=yaxis,
        width=960,
        height=600,
        margin=dict(t=80, b=120),
        legend=dict(
            orientation="h", yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
        ),
        annotations=annotations,
        template="plotly_white",
        hovermode="x unified",
    )

    return fig

# ---------------------------------------------------------------------------------------------------------------------

def _facet_grid_graphs(df,
                       group_col, 
                       col_y=None, 
                       col_x=None, 
                       y_label='', 
                       x_label='', 
                       title='', 
                       group_row=None, 
                       group_col_label=None, 
                       group_row_label=None, 
                       lowess=0.5, 
                       zoom_in = False, 
                       x_is_discrete=False, 
                       x_ticks=None, 
                       fontsize=14, 
                       facet_col_limit_per_page=None, 
                       export_to='object', 
                       save_within_page_limits=True, 
                       dist_plot=False
                       ):
    """
    Create faceted scatterplots, boxplots, or KDE distribution plots across
    grouping columns, and export the result as an image or file.

    Parameters
    ----------
    df : DataFrame
        pandas/polars/pyarrow/narwhals DataFrame with data.
    group_col : str
        Column used to facet by columns (or as hue, when `dist_plot` is
        True).
    col_y : str, optional
        Column header name for the y axis. Not used when `dist_plot` is
        True.
    col_x : str, optional
        Column header name for the x axis.
    y_label : str, default ''
        Y axis label for the graph.
    x_label : str, default ''
        X axis label for the graph.
    title : str, default ''
        Graph title, word-wrapped to 90 characters.
    group_row : str, optional
        Column used to facet by rows. When provided, `facet_col_limit_per_page`
        is ignored.
    group_col_label : str, optional
        Label prefix used for column facet titles (e.g. 'Group {col_name}').
    group_row_label : str, optional
        Label prefix used for row facet titles (e.g. 'Group {row_name}').
    lowess : float, default 0.5
        Lowess smoothing fraction passed to ``sns.regplot`` for
        scatterplot facets.
    zoom_in : bool, default False
        If True, restrict the y-axis to the range (0.8, 1.2) on
        scatterplot facets.
    x_is_discrete : bool, default False
        If True, boxplot facets are generated; otherwise scatterplot
        facets are generated (ignored when `dist_plot` is True).
    x_ticks : str, optional
        Column header name used to derive custom x-axis tick labels,
        binned by `col_x`.
    fontsize : int, default 14
        Base font size used for facet titles and the overall title.
    facet_col_limit_per_page : int, optional
        Maximum number of column facets per row (``col_wrap``); ignored
        when `group_row` is provided.
    export_to : str, default 'object'
        If 'object', return a ReportLab Image object. Otherwise, a
        directory or full '.png' filepath to save the figure to; if a
        directory is given, a randomly named '.png' file is created
        inside it.
    save_within_page_limits : bool, default True
        Passed through to ``_get_image`` when `export_to` is 'object', to
        constrain the image to page limits.
    dist_plot : bool, default False
        If True, generate a KDE distribution plot (``sns.displot``) with
        `group_col` as hue and `group_row` as column facet, instead of a
        scatterplot/boxplot FacetGrid.

    Returns
    -------
    reportlab.platypus.Image or str
        * If `export_to` is 'object', a ReportLab Image object.
        * Otherwise, the filepath (str) where the figure was saved.
    """
    try:
        # seaborn requires a real pandas DataFrame. Coerce at the boundary.
        df = nw_utils.require_pandas(df, reason="_facet_grid_graphs() uses seaborn")
        df = df.copy()
        if title:
            wrapped_title = "\n".join(textwrap.wrap(title, width=90))
            title_extra_lines = len(wrapped_title.split("\n")) - 1

        sns.set_theme(style="darkgrid")
        if not x_is_discrete and not dist_plot:
            df = df[df[col_x] != 0] # scatterplot
            if not df.empty and len(df) > 1:
                hist_bins = len(df[col_x].drop_duplicates()) if len(df[col_x].drop_duplicates()) < 70 else 70
        if x_ticks and x_ticks in df:
            if len(df[x_ticks].drop_duplicates()) <= 8:
                bin_edges = list(df.groupby(x_ticks)[col_x].first())
            else:
                bin_edges = list(pd.cut(df[col_x], bins=6, retbins=True, include_lowest=True)[1].astype(int))
            bin_edges[0] = df[col_x].min()
            df[col_x + '_xticks'] = np.where(df[col_x].isin(bin_edges), df[col_x], 0)
            df[x_ticks] = np.where(df[col_x + '_xticks'] == df[col_x], df[x_ticks], '')

        col_wrap = facet_col_limit_per_page if group_row is None else None
        #dist_plot
        if dist_plot:
            g = sns.displot(data=df, x=col_x, hue=group_col, col=group_row, kind="kde", fill=False, common_norm=False, palette="crest"
                ,alpha=1, linewidth=2, multiple="layer") # option kind = 'hist'
        else:
            g = sns.FacetGrid(df, row=group_row, col=group_col, margin_titles=False, col_wrap=col_wrap, aspect=1, height=5, dropna=True)

            #boxplot
            if x_is_discrete:
                plot_order = list(df[col_x].drop_duplicates().sort_values())
                g.map(sns.boxplot, x=col_x, y=col_y, data=df, color='#8080ff', linewidth=2, order=plot_order)
            #scatterplot
            else:
                g.map(sns.regplot, col_x, col_y, scatter_kws={"color": "blue", 's':2}, line_kws={"color": "orange"}, lowess=lowess)
                if zoom_in: plt.ylim(0.8, 1.2)
                if x_ticks and x_ticks in df:
                    for ax in g.axes.flat:
                        ax.set_xticks(df[df[x_ticks] != ''][col_x].drop_duplicates())
                        ax.set_xticklabels(df[df[x_ticks] != ''][x_ticks].drop_duplicates(), rotation=45)

        if title and (col_wrap or group_row):
            if col_wrap: #adjustment for title in group graphs
                number_of_rows = math.ceil((len(df[group_col].drop_duplicates()) / facet_col_limit_per_page))
                top_adjust = 0.97 - round((title_extra_lines * 0.05 + 0.17) / number_of_rows,4)
                bottom_adjust = round(0.20 / math.ceil((len(df[group_col].drop_duplicates()) / facet_col_limit_per_page)),4) if x_ticks and x_ticks in df else None
            elif group_row: #adjustment for title in mosaic graphs
                number_of_rows = math.ceil(len(df[group_col].drop_duplicates()))
                top_adjust = 1 - round((title_extra_lines * 0.10 + 0.10) / number_of_rows, 4)
                bottom_adjust = None
                if dist_plot:
                    top_adjust -= 0.1
            h_space_adjust=None
            # h_space_adjust = 0.5 if x_ticks and x_ticks in df else None
            # h_space_adjust = 0.25 if x_is_discrete else h_space_adjust
            g.figure.subplots_adjust(top=top_adjust, hspace=h_space_adjust, bottom=bottom_adjust)

        if title:
            g.figure.suptitle(wrapped_title, fontsize=fontsize+4, weight='bold')
        g.set_xlabels(x_label)
        g.set_ylabels(y_label)
        g.set_titles(row_template= group_row_label+' {row_name}' if group_row is not None and group_row_label is not None else None, col_template=group_col_label + ' {col_name}' if group_col_label is not None and group_col is not None else None, size=fontsize)
        if export_to=='object':
            imgdata = BytesIO()
            g.figure.savefig(imgdata, format='png', )
            imgdata.seek(0)
            result = _get_image(imgdata, width=7 * inch, save_within_page_limits=True)
        elif os.path.exists(export_to):
            if os.path.splitext(export_to)[1] == '.png': #if full path is provided
                g.figure.savefig(export_to)
                result = export_to
            else:  #if only directory is provided
                result = os.path.join(export_to, str(uuid.uuid4()) + '.png')
                g.figure.savefig(result)
        plt.clf()
        return result

    except Exception as ex:
        raise RuntimeError('error in ratio study: _facet_grid_graphs', ex)


# ---------------------------------------------------------------------------------------------------------------------
# Vertical Equity Panel
# ---------------------------------------------------------------------------------------------------------------------

# Gauge geometry, in degrees, using the matplotlib convention (0 = east,
# angles increase counter-clockwise). The dial is a full circle split into
# four fixed regions:
#
#   green   45 deg centred on the top      -> acceptable / within standards
#   red    135 deg on the right            -> progressivity
#   red    135 deg on the left             -> regressivity
#   grey    45 deg centred on the bottom   -> inconclusive
_GAUGE_TOP = 90.0
_GAUGE_GREEN_HALF = 22.5
_GAUGE_RED_SPAN = 135.0
_GAUGE_PROG_EDGE = _GAUGE_TOP - _GAUGE_GREEN_HALF   # 67.5, right edge of green
_GAUGE_REG_EDGE = _GAUGE_TOP + _GAUGE_GREEN_HALF    # 112.5, left edge of green
_GAUGE_BOTTOM = 270.0

_GAUGE_GREEN = '#5cb85c'
_GAUGE_RED = '#d9534f'
_GAUGE_GREY = '#bdbdbd'
_GAUGE_MUTED = '#eeeeee'
_GAUGE_NEEDLE = '#212121'
_VE_SCATTER_GREEN = '#2e7d32'
_VE_SCATTER_BLUE = '#1f77b4'
_VE_CI_BLUE = '#005adc'

_GAUGE_STATUS_COLOURS = {
    'acceptable': '#2e7d32',
    'progressive': '#b71c1c',
    'regressive': '#b71c1c',
    'inconclusive': '#616161',
    'unavailable': '#9e9e9e',
}

# Geometry of the gauge row on the panel: the horizontal span four
# gauges occupy, used to keep the dial size constant (and the row
# centred) when one of them is dropped.
_GAUGE_ROW_COLUMNS = 4
_GAUGE_ROW_SPAN = 0.89

# Heights of the read-out's detail lines, in gauge data units. Two slots:
# the VEI gauge can carry both a differing test verdict and the CoD.
_GAUGE_DETAIL_Y = (-2.15, -2.45)

# ``vem``'s acceptable conclusion, whose severity and prevalence are nil.
_VEM_NO_INEQUITY = 'acceptable'

# The VEM legend is drawn at this share of the tick size: six entries at
# full size crowd the plot it sits on.
_VEM_LEGEND_SCALE = 0.6

_GAUGE_STATUS_LABELS = {
    'acceptable': 'Acceptable',
    'progressive': 'Progressivity',
    'regressive': 'Regressivity',
    'inconclusive': 'Inconclusive',
    'unavailable': 'Not available',
}

# The VEI's own word for a passing verdict is 'Acceptable' (see
# ``VEI_ACCEPTABLE`` in indicators.py, which cannot be imported here without
# a cycle -- indicators imports this module). Comparing its conclusion
# against the shared 'Acceptable' label made every acceptable VEI look like it
# disagreed with its own needle, so each gauge carried a redundant
# 'VEI test: Acceptable' note.
_VEI_STATUS_LABELS = {**_GAUGE_STATUS_LABELS, 'acceptable': 'Acceptable'}

# Explicit style for the panel, so a previous ``seaborn.set()`` call
# elsewhere in the session cannot change how the report figure renders.
_VE_PANEL_RC = {
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#bdbdbd',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.6,
    'text.color': '#212121',
    'axes.labelcolor': '#212121',
    'xtick.color': '#616161',
    'ytick.color': '#616161',
    'legend.frameon': False,
}


def _ve_coerce_float(value):
    """
    Coerce an indicator value to a float, tolerating significance markers.

    Indicator values such as PRB may arrive as strings carrying a
    non-significance marker (e.g. ``'0.043*'``), as produced by
    :func:`ratio_study.indicators.prb`.

    Parameters
    ----------
    value : float, int, str, or None
        The raw indicator value.

    Returns
    -------
    tuple
        * value : float or None -- the numeric value, or None when it could
          not be parsed or is NaN.
        * flagged : bool -- True when the input was a string carrying
          trailing/leading characters beyond the number itself (i.e. a
          non-significance marker was present).
    """
    if value is None:
        return None, False

    if isinstance(value, str):
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
        if match is None:
            return None, True
        return float(match.group(0)), match.group(0) != value.strip()

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, False

    if math.isnan(numeric):
        return None, False
    return numeric, False

# ---------------------------------------------------------------------------------------------------------------------

def _gauge_angle(value, lo, hi, outer_lo, outer_hi, status):
    """
    Map a signed indicator value onto a needle angle on the gauge dial.

    The scale is piecewise linear so that the acceptable band always lands
    inside the green wedge and neutrality always points straight up,
    regardless of how asymmetric the band is (PRD's ``(0.98, 1.03)``, for
    instance). ``status`` decides which region the needle may occupy, and
    the value is clamped into that region, so the needle and the reported
    conclusion can never contradict each other.

    Parameters
    ----------
    value : float
        Signed indicator value, expressed so that positive means
        progressivity and negative means regressivity.
    lo, hi : float
        Regressive and progressive edges of the acceptable band, in the
        same signed units as `value` (normally ``lo < 0 < hi``).
    outer_lo, outer_hi : float
        Full-scale ends of the dial on the regressive and progressive
        sides; values at or beyond these point at the outer edge of the
        corresponding red wedge.
    status : str
        One of 'acceptable', 'progressive', 'regressive', 'inconclusive',
        or 'unavailable'.

    Returns
    -------
    float
        Needle angle in degrees.
    """
    if status in ('inconclusive', 'unavailable'):
        return _GAUGE_BOTTOM

    if status == 'acceptable':
        clamped = min(max(value, lo), hi)
        if clamped >= 0:
            fraction = (clamped / hi) if hi else 0.0
            return _GAUGE_TOP - _GAUGE_GREEN_HALF * fraction
        fraction = (clamped / lo) if lo else 0.0
        return _GAUGE_TOP + _GAUGE_GREEN_HALF * fraction

    if status == 'progressive':
        span = outer_hi - hi
        fraction = 0.0 if span <= 0 else min(max((value - hi) / span, 0.0), 1.0)
        return _GAUGE_PROG_EDGE - _GAUGE_RED_SPAN * fraction

    # regressive: ``span`` is negative, so the ratio is still in [0, 1].
    span = outer_lo - lo
    fraction = 0.0 if span == 0 else min(max((value - lo) / span, 0.0), 1.0)
    return _GAUGE_REG_EDGE + _GAUGE_RED_SPAN * fraction

# ---------------------------------------------------------------------------------------------------------------------

def _gauge_status(value, lo, hi, conclusive=True):
    """
    Classify a signed indicator value against its acceptable band.

    Parameters
    ----------
    value : float or None
        Signed indicator value; positive means progressivity.
    lo, hi : float
        Regressive and progressive edges of the acceptable band.
    conclusive : bool, default True
        False when the indicator is not statistically significant. A value
        inside the band is still 'acceptable' (there is no evidence of
        bias and the point estimate is acceptable); a value outside the
        band that is not significant is 'inconclusive'.

    Returns
    -------
    str
        'unavailable', 'acceptable', 'inconclusive', 'progressive', or
        'regressive'.
    """
    if value is None:
        return 'unavailable'
    if lo <= value <= hi:
        return 'acceptable'
    if not conclusive:
        return 'inconclusive'
    return 'progressive' if value > hi else 'regressive'

# ---------------------------------------------------------------------------------------------------------------------

def _draw_ve_gauge(ax, spec, label_fontsize=14, value_fontsize=16, tick_fontsize=9):
    """
    Draw a single vertical-equity gauge on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on; its frame and ticks are turned off.
    spec : dict
        Gauge specification as built by ``_ve_gauge_specs``, with keys
        ``'label'``, ``'value_text'``, ``'detail'``, ``'status'``,
        ``'angle'``, ``'tick_lo'`` and ``'tick_hi'``.
    label_fontsize, value_fontsize, tick_fontsize : int
        Font sizes for the indicator name, the value read-out, and the
        band-edge tick labels.

    Returns
    -------
    None
    """
    ax.set_aspect('equal')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-2.58, 1.52)
    ax.axis('off')
    ax.grid(False)

    unavailable = spec['status'] == 'unavailable'
    green = _GAUGE_MUTED if unavailable else _GAUGE_GREEN
    red = _GAUGE_MUTED if unavailable else _GAUGE_RED
    grey = _GAUGE_MUTED if unavailable else _GAUGE_GREY

    ring_width = 0.30
    regions = [
        (_GAUGE_PROG_EDGE - _GAUGE_RED_SPAN, _GAUGE_PROG_EDGE, red),   # right / progressive
        (_GAUGE_PROG_EDGE, _GAUGE_REG_EDGE, green),                    # top / acceptable
        (_GAUGE_REG_EDGE, _GAUGE_REG_EDGE + _GAUGE_RED_SPAN, red),     # left / regressive
        (_GAUGE_BOTTOM - _GAUGE_GREEN_HALF,
         _GAUGE_BOTTOM + _GAUGE_GREEN_HALF, grey),                     # bottom / inconclusive
    ]
    for theta1, theta2, colour in regions:
        ax.add_patch(Wedge((0, 0), 1.0, theta1, theta2, width=ring_width,
                           facecolor=colour, edgecolor='white', linewidth=1.2,
                           zorder=2))

    # Band-edge ticks: the regressive edge on the left of the green wedge,
    # the progressive edge on its right.
    for angle, text, ha in ((_GAUGE_REG_EDGE, spec['tick_lo'], 'right'),
                            (_GAUGE_PROG_EDGE, spec['tick_hi'], 'left')):
        radians = np.deg2rad(angle)
        cos, sin = np.cos(radians), np.sin(radians)
        ax.plot([0.98 * cos, 1.12 * cos], [0.98 * sin, 1.12 * sin],
                color='#757575', lw=1.0, zorder=3)
        if text:
            ax.text(1.18 * cos, 1.18 * sin, text, ha=ha, va='bottom',
                    fontsize=tick_fontsize, color='#616161', zorder=3)

    # Needle
    if not unavailable:
        radians = np.deg2rad(spec['angle'])
        ax.annotate(
            '', xy=(0.78 * np.cos(radians), 0.78 * np.sin(radians)), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', mutation_scale=16, linewidth=2.0,
                            color=_GAUGE_NEEDLE, shrinkA=0, shrinkB=0),
            zorder=4,
        )
    ax.add_patch(Circle((0, 0), 0.07,
                        facecolor=_GAUGE_NEEDLE if not unavailable else _GAUGE_GREY,
                        edgecolor='white', linewidth=1.0, zorder=5))

    # The read-out sits below the dial rather than inside it: the needle
    # sweeps the whole interior, so a centred value would be struck
    # through whenever a result is strongly regressive or progressive.
    # Each line gets its own band of clear space -- dial, value,
    # conclusion, detail -- so the read-out does not run together.
    status_colour = _GAUGE_STATUS_COLOURS[spec['status']]
    ax.text(0, 1.39, spec['label'], ha='center', va='center',
            fontsize=label_fontsize, fontweight='bold')
    ax.text(0, -1.42, spec['value_text'], ha='center', va='center',
            fontsize=value_fontsize, fontweight='bold', color=status_colour)
    ax.text(0, -1.80, _GAUGE_STATUS_LABELS[spec['status']], ha='center', va='center',
            fontsize=label_fontsize - 2, color=status_colour)
    # Detail lines sit at fixed heights rather than flowing from a
    # multi-line string, so every gauge on the row keeps the same
    # geometry however many lines it carries.
    detail = spec.get('detail') or ()
    if isinstance(detail, str):
        detail = (detail,)
    for line, y in zip(detail, _GAUGE_DETAIL_Y):
        ax.text(0, y, line, ha='center', va='center',
                fontsize=tick_fontsize + 1, color='#616161')

# ---------------------------------------------------------------------------------------------------------------------

def _unwrap_vei_result(vei_result):
    """
    Return the flat metrics dictionary from a ``vei_analysis`` result.

    ``vei_analysis`` returns either the metrics dictionary itself
    (``metrics_only=True``) or a wrapper with ``'vei_metrics'`` and
    ``'group_stats'`` keys. This normalizes both shapes, and copes with
    the ``output_format='df'`` variants by taking the first row.

    Parameters
    ----------
    vei_result : dict or DataFrame or None
        Result as produced by ``vei_analysis``.

    Returns
    -------
    dict
        Flat metrics dictionary, with ``'pg_stats'`` populated when the
        per-group statistics are available. Empty dict when `vei_result`
        is None.
    """
    if vei_result is None:
        return {}

    if isinstance(vei_result, dict) and 'vei_metrics' in vei_result:
        metrics = vei_result['vei_metrics']
        group_stats = vei_result.get('group_stats')
    else:
        metrics = vei_result
        group_stats = None

    if isinstance(metrics, pd.DataFrame):
        metrics = {} if metrics.empty else metrics.iloc[0].to_dict()
    metrics = dict(metrics)

    if not metrics.get('pg_stats') and isinstance(group_stats, list):
        metrics['pg_stats'] = group_stats
    if not metrics.get('pg_stats') and metrics.get('VEI_pg_stats'):
        metrics['pg_stats'] = metrics['VEI_pg_stats']
    return metrics

# ---------------------------------------------------------------------------------------------------------------------

def _secondary_indicator_specs(asr, proxy, assessment=None, market_value=None,
                               vei_metrics=None, weights=None, req_significance=0.05):
    """
    Compute the Appendix F secondary vertical-equity tests for the panel strip.

    Appendix F lists several further tests and recommends running more than
    one, "as they may provide additional insights into potential valuation
    bias". None of them carries an IAAO acceptable range, so the three
    hypothesis tests are coloured by significance only and the two Gini
    indices are left neutral -- a number to read, not a verdict.

    Parameters
    ----------
    asr, proxy : array-like
        Ratios and the market value proxy the tests run on.
    assessment, market_value : array-like, optional
        Needed for the Gini indices. Omitted, MKI and RRI report as 'n/a'.
    vei_metrics : dict, optional
        VEI result; its Kruskal-Wallis is reused rather than recomputed.
    weights : array-like, optional
        Observation weights for the Gini indices.
    req_significance : float, default 0.05
        p-value below which a test is treated as having found something.

    Returns
    -------
    list of dict
        ``label``, ``value``, ``note`` and ``flag`` (True when a test found
        significant structure, None for the un-graded indices).
    """
    def _fmt_p(p):
        return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

    items = []

    # --- Gini family (Appendix F: Modified Kakwani / reranking) ----------
    mki = rri = None
    if assessment is not None and market_value is not None and len(asr) > 3:
        try:
            from .gini import gini_stats as _gini_stats
            g = _gini_stats(market_value, assessment, weights=weights, run_regression=True)
            mki, rri = g.get('Modified_Kakwani_Index'), g.get('Reranking_Index')
        except Exception:
            mki = rri = None

    items.append({
        'label': 'MKI',
        'value': 'n/a' if mki is None else f'{mki:.3f}',
        'note': ('Gini index unavailable' if mki is None else
                 'equity at 1.00 | ' + ('regressive' if mki < 1 else
                                        'progressive' if mki > 1 else 'neutral')),
        'flag': None,
    })
    items.append({
        'label': 'RRI',
        'value': 'n/a' if rri is None else f'{rri:.2%}',
        'note': 'reranking | horizontal' if rri is not None else 'Gini index unavailable',
        'flag': None,
    })

    # --- Mann-Whitney: lowest half vs highest half of the value range ----
    # Section 6.4's "two property groups are appraised at equal percentages of
    # market value", applied to the bottom and top of the value range.
    try:
        from scipy.stats import mannwhitneyu
        order = np.argsort(proxy)
        ranked = np.asarray(asr, dtype=float)[order]
        half = len(ranked) // 2
        low, high = ranked[:half], ranked[-half:]
        if half >= 5:
            _, p_mw = mannwhitneyu(low, high, alternative='two-sided')
            items.append({'label': 'Mann-Whitney', 'value': _fmt_p(p_mw),
                          'note': 'low vs high half', 'flag': bool(p_mw < req_significance)})
        else:
            raise ValueError('too few sales')
    except Exception:
        items.append({'label': 'Mann-Whitney', 'value': 'n/a',
                      'note': 'too few sales', 'flag': None})

    # --- Kruskal-Wallis across every quantile group (Appendix E.4) -------
    kw = (vei_metrics or {}).get('kruskal_wallis')
    if kw:
        items.append({'label': 'Kruskal-Wallis', 'value': _fmt_p(kw['pvalue']),
                      'note': f"H = {kw['statistic']:,.1f}",
                      'flag': bool(kw['pvalue'] < req_significance)})
    else:
        items.append({'label': 'Kruskal-Wallis', 'value': 'n/a',
                      'note': 'no quantile groups', 'flag': None})

    # --- Spearman rank: ratio against the value proxy --------------------
    try:
        from scipy.stats import spearmanr
        rho, p_sp = spearmanr(np.asarray(proxy, dtype=float), np.asarray(asr, dtype=float))
        items.append({
            'label': 'Spearman', 'value': f'{rho:+.3f}',
            'note': f"{_fmt_p(p_sp)} | " + ('regressive' if rho < 0 else 'progressive'),
            'flag': bool(p_sp < req_significance),
        })
    except Exception:
        items.append({'label': 'Spearman', 'value': 'n/a', 'note': '', 'flag': None})

    return items


def _draw_secondary_strip(ax, items):
    """
    Render the secondary-indicator strip beneath the gauges.

    One cell per test, separated by hairlines: label above, value below,
    and a one-phrase note. Deliberately quiet next to the gauges -- these
    are supporting evidence, not the acceptance verdict.
    """
    # Labelled so callers (and the layout tests) can tell the strip apart from
    # the gauges, which are likewise axis-free.
    ax.set_label('secondary_indicators')
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor='#f7f7f9', edgecolor='#dcdce2',
                           linewidth=0.8, zorder=0))

    n = len(items)
    for i, item in enumerate(items):
        centre = (i + 0.5) / n
        if i:
            ax.plot([i / n, i / n], [0.16, 0.84], transform=ax.transAxes,
                    color='#dcdce2', linewidth=0.8, zorder=1)

        colour = ('#c0392b' if item['flag'] else
                  '#2e7d32' if item['flag'] is False else '#37474f')

        ax.text(centre, 0.86, item['label'], transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, color='#6b6b76',
                fontweight='bold')
        ax.text(centre, 0.48, item['value'], transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color=colour,
                fontweight='bold')
        ax.text(centre, 0.14, item['note'], transform=ax.transAxes,
                ha='center', va='center', fontsize=7.5, color='#8a8a94')


def _ve_gauge_specs(prb_value=None,
                    prb_not_significant=False,
                    prd_value=None,
                    vei_metrics=None,
                    vem_result=None,
                    cod_value=None,
                    assessment_standards=None,
                    vei_acceptable_range=10.0,
                    vem_acceptable_inequity_level=0.1,
                    outer_multiple=3.0):
    """
    Build the four gauge specifications for the vertical equity panel.

    Every gauge is expressed in *signed progressivity units*: positive
    means progressivity (needle to the right), negative means
    regressivity (needle to the left). PRD is therefore reflected
    (``1 - PRD``), since a PRD below 1 indicates progressivity.

    Parameters
    ----------
    prb_value : float or str, optional
        PRB coefficient, optionally carrying a non-significance marker.
    prb_not_significant : bool, default False
        True when the PRB slope is not statistically significant. This is
        the ``prb_sig_flag`` returned by
        :func:`ratio_study.indicators.prb`.
    prd_value : float, optional
        PRD (price-related differential).
    vei_metrics : dict, optional
        Flat VEI metrics as returned by ``_unwrap_vei_result``.
    vem_result : dict, optional
        Result dictionary from :func:`ratio_study.indicators.vem`.
    cod_value : float, optional
        Coefficient of dispersion as a proportion (e.g. 0.083), reported
        under the VEI gauge. Omitted from the read-out when None.
    assessment_standards : dict, optional
        Acceptable bands keyed by indicator name. ``'PRB'`` defaults to
        ``(-0.10, 0.10)`` and ``'PRD'`` to ``(0.98, 1.03)``.
    vei_acceptable_range : float, default 10.0
        Acceptable |VEI| in percent. Overridden by the VEI result's own
        ``'acceptable_range'`` when present.
    vem_acceptable_inequity_level : float, default 0.1
        Acceptable inequity level used by ``vem``, as a proportion.
    outer_multiple : float, default 3.0
        Full scale of each red wedge, as a multiple of the corresponding
        band edge. A value at ``outer_multiple`` times the band edge puts
        the needle at the outer end of the red region.

    Returns
    -------
    list of dict
        One specification per gauge, in PRB, PRD, VEI, VEM order. Each
        ``'detail'`` is a tuple of extra read-out lines (empty when there
        are none), drawn under the conclusion.
    """
    standards = assessment_standards or {}
    prb_band = standards.get('PRB', (-0.10, 0.10))
    prd_band = standards.get('PRD', (0.98, 1.03))
    specs = []

    # --- PRB: signed directly, positive means progressivity. -------------
    prb_numeric, marker = _ve_coerce_float(prb_value)
    prb_lo, prb_hi = float(prb_band[0]), float(prb_band[1])
    prb_status = _gauge_status(prb_numeric, prb_lo, prb_hi,
                              conclusive=not (marker or prb_not_significant))
    specs.append({
        'label': 'PRB',
        'value_text': 'n/a' if prb_numeric is None else f"{prb_numeric:+.3f}",
        'detail': (),
        'status': prb_status,
        'tick_lo': f"{prb_lo:+.2f}",
        'tick_hi': f"{prb_hi:+.2f}",
        'angle': _gauge_angle(prb_numeric or 0.0, prb_lo, prb_hi,
                             prb_lo * outer_multiple, prb_hi * outer_multiple,
                             prb_status),
    })

    # --- PRD: reflected, since PRD < 1 indicates progressivity. ----------
    prd_numeric, _ = _ve_coerce_float(prd_value)
    prd_signed = None if prd_numeric is None else 1.0 - prd_numeric
    prd_hi = 1.0 - float(prd_band[0])   # progressive edge, e.g. +0.02
    prd_lo = 1.0 - float(prd_band[1])   # regressive edge,  e.g. -0.03
    prd_status = _gauge_status(prd_signed, prd_lo, prd_hi)
    specs.append({
        'label': 'PRD',
        'value_text': 'n/a' if prd_numeric is None else f"{prd_numeric:.3f}",
        'detail': (),
        'status': prd_status,
        'tick_lo': f"{float(prd_band[1]):.2f}",
        'tick_hi': f"{float(prd_band[0]):.2f}",
        'angle': _gauge_angle(prd_signed or 0.0, prd_lo, prd_hi,
                             prd_lo * outer_multiple, prd_hi * outer_multiple,
                             prd_status),
    })

    # --- VEI: already a signed percentage. --------------------------------
    # The dial shows where the point estimate falls; ``vei_analysis`` may
    # reach a different verdict from its own significance test, and when it
    # does, that verdict is reported underneath rather than overriding the
    # needle. Hiding a large point estimate behind a 'Acceptable' label
    # would make the gauge say the opposite of the number printed on it.
    vei_metrics = vei_metrics or {}
    vei_numeric, _ = _ve_coerce_float(vei_metrics.get('vei'))
    vei_range = float(vei_metrics.get('acceptable_range') or vei_acceptable_range)
    vei_conclusion = str(vei_metrics.get('conclusion') or '')
    if vei_numeric is None or not vei_conclusion:
        vei_status = 'unavailable'
    elif vei_conclusion in ('Inconclusive', 'Sample size too small'):
        vei_status = 'inconclusive'
    else:
        vei_status = _gauge_status(vei_numeric, -vei_range, vei_range)

    vei_detail = []
    if vei_conclusion and vei_conclusion not in ('Inconclusive',
                                                 'Sample size too small'):
        if vei_conclusion != _VEI_STATUS_LABELS[vei_status]:
            vei_detail.append(f"VEI test: {vei_conclusion}")
    elif vei_conclusion == 'Sample size too small':
        vei_detail.append('sample size too small')

    # The CoD rides along under the VEI gauge: it is the dispersion the
    # VEI groups are drawn from, and reading the two together is how a
    # ratio study is normally interpreted.
    cod_numeric, _ = _ve_coerce_float(cod_value)
    if cod_numeric is not None:
        vei_detail.append(f"COD: {cod_numeric * 100:.1f}%")

    specs.append({
        'label': 'VEI',
        # ``vei_analysis`` reports 0 when it bails out on sample size, which
        # would read as a perfectly neutral result rather than no result.
        'value_text': 'n/a' if (vei_numeric is None
                                or vei_conclusion == 'Sample size too small')
                      else f"{vei_numeric:+.1f}%",
        'detail': tuple(vei_detail),
        'status': vei_status,
        'tick_lo': f"-{vei_range:g}%",
        'tick_hi': f"+{vei_range:g}%",
        'angle': _gauge_angle(vei_numeric or 0.0, -vei_range, vei_range,
                             -vei_range * outer_multiple, vei_range * outer_multiple,
                             vei_status),
    })

    # --- VEM: severity is unsigned, so the direction supplies the sign. ---
    vem_result = vem_result or {}
    vem_conclusion = str(vem_result.get('Inequity_conclusion') or '')
    vem_severity, _ = _ve_coerce_float(vem_result.get('Inequity_severity'))
    vem_prevalence, _ = _ve_coerce_float(vem_result.get('Inequity_prevalence'))
    vem_range = float(vem_acceptable_inequity_level or 0.1) * 100.0

    if not vem_conclusion:
        vem_status = 'unavailable'
    elif vem_conclusion == 'Progressivity':
        vem_status = 'progressive'
    elif vem_conclusion == 'Regressivity':
        vem_status = 'regressive'
    elif vem_conclusion == 'Inconclusive':
        vem_status = 'inconclusive'
    else:  
        vem_status = 'acceptable'

    # ``Inequity_severity`` is a magnitude, so the direction supplies the
    # sign. An inconclusive result has no established direction, so it is
    # shown unsigned to avoid implying one.
    sign = -1.0 if vem_status == 'regressive' else 1.0
    vem_signed = 0.0 if vem_severity is None else sign * abs(vem_severity)
    if vem_status == 'acceptable':
        vem_signed = 0.0

    if vem_status == 'unavailable':
        vem_value_text = 'n/a'
    elif vem_status == 'inconclusive':
        vem_value_text = f"{abs(vem_signed):.1f}%"
    else:
        vem_value_text = f"{vem_signed:+.1f}%"

    specs.append({
        'label': 'VEM',
        'value_text': vem_value_text,
        'detail': () if vem_prevalence is None
                  else (f"prevalence {vem_prevalence:.1f}%",),
        'status': vem_status,
        'tick_lo': f"-{vem_range:g}%",
        'tick_hi': f"+{vem_range:g}%",
        'angle': _gauge_angle(vem_signed, -vem_range, vem_range,
                             -vem_range * outer_multiple, vem_range * outer_multiple,
                             vem_status),
    })

    return specs

# ---------------------------------------------------------------------------------------------------------------------

def _currency_tick(value, _pos=None):
    """
    Format an axis tick value as an abbreviated dollar amount.

    Parameters
    ----------
    value : float
        Tick value.
    _pos : int, optional
        Tick position, supplied by matplotlib and unused.

    Returns
    -------
    str
        Abbreviated dollar label, e.g. ``'$1.2M'`` or ``'$450K'``.
    """
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"${value/1e9:.1f}B"
    if magnitude >= 1e6:
        return f"${value/1e6:.1f}M"
    if magnitude >= 1e3:
        return f"${value/1e3:.0f}K"
    return f"${value:,.0f}"

# ---------------------------------------------------------------------------------------------------------------------

def _vem_axes(ax,
                    data,
                    asr_col='asr',
                    acceptable_inequity_level=0.1,
                    x_scale='log',
                    y_scale='linear',
                    zoom_y=True,
                    conf_level=None,
                    center=None,
                    smart_zoom=True,
                    show_bounds=True,
                    smoothing_label='',
                    title='',
                    label_fontsize=13,
                    tick_fontsize=10,
                    legend_fontsize=None):
    """
    Draw the vertical-equity moving-median chart on matplotlib axes.

    This is the static counterpart of :func:`ve_scat_median_plot`, drawn
    with matplotlib so it can be composed into a single panel figure and
    embedded in the ReportLab report without invoking Kaleido/Orca.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    data : DataFrame
        Working frame from :func:`ratio_study.indicators.vem`, with
        columns 'mv', `asr_col`, 'moving_median', 'out_of_bounds' and
        optionally 'asr_raw', 'median_ci_lower' / 'median_ci_upper'.
    asr_col : str, default 'asr'
        Name of the median-centred ASR column.
    acceptable_inequity_level : float, default 0.1
        Acceptable inequity threshold, drawn as red dashed reference lines
        at the median ASR plus/minus this value. Falsy values omit the
        lines.
    show_bounds : bool, default True
        Colour the scatter by the 'out_of_bounds' flag -- green where the
        local median is acceptable, red where it is not -- and label both
        groups in the legend. False draws a single plain blue scatter with
        no legend entry, for panels that do not report a VEM verdict.
    smoothing_label : str, default ''
        Rolling window (or quantile count) behind the moving median, e.g.
        ``'120'``, added to its legend entry -- the smoothing describes
        that line rather than the chart as a whole.
    x_scale, y_scale : str, default 'log' / 'linear'
        Axis scales, 'linear' or 'log'.
    zoom_y : bool, list, or tuple, default True
        Y-axis range. True zooms to the median ASR +/- 0.3, a
        ``(low, high)`` pair sets a custom range in ASR units, and False
        leaves the axis auto-ranged. A pair always wins; otherwise
        `smart_zoom` takes precedence over True.
    conf_level : float, optional
        Confidence level behind the CI band; used for the legend label.
    center : float, optional
        The overall median ASR that ``vem`` subtracted, i.e. its
        ``'asr_center'`` result. The moving median, its CI band and the
        acceptable-inequity bands are all shifted back by this amount so
        the axes read in ASR units (around the median, typically ~1)
        instead of around 0. Recovered from the frame's 'asr_raw' column
        when omitted.
    smart_zoom : bool, default True
        Frame both axes on the bulk of the scatter instead of its full
        range: the x-axis on the 1st-99th percentile of the market value
        proxy, and the y-axis on 95% of the ASR values inside that
        window (see :func:`_ve_smart_limits`). Low- and high-value
        outliers fall outside the view; the moving median, its CI band
        and the acceptable-inequity lines never do. Set False to plot
        the full range and fall back to `zoom_y`.
    title : str, default ''
        Axes title.
    label_fontsize, tick_fontsize : int
        Font sizes for axis labels/title and for tick text.
    legend_fontsize : float, optional
        Font size for the legend. Defaults to 60% of `tick_fontsize`,
        with padding and spacing scaled to match, so the six entries do
        not crowd the plot.

    Returns
    -------
    None
    """
    data = nw_utils.require_pandas(data, reason="_vem_axes()")

    missing = [column for column in (asr_col, 'mv', 'moving_median', 'out_of_bounds')
               if column not in data.columns]
    if missing:
        raise ValueError(
            f"vem frame is missing column(s) {missing}. Pass the frame "
            "returned by vem() under its 'dataframe' key, and set "
            "`vem_asr_col` if it was computed with a custom `asr_col`."
        )

    in_bounds = data['out_of_bounds'] == 0
    out_bounds = ~in_bounds

    # The indices are computed on the median-centred ASR; the axes are
    # drawn in ASR units, so every centred quantity is shifted back by
    # the overall median.
    center = _ve_display_center(data, center, asr_col=asr_col)
    asr_display = _ve_display_asr(data, center, asr_col=asr_col)

    if show_bounds:
        ax.scatter(data.loc[in_bounds, 'mv'], asr_display[in_bounds],
                   s=3, alpha=0.45, color=_VE_SCATTER_GREEN, linewidths=0,
                   label=f"{asr_col.upper()} (acceptable local median)", zorder=2)
        ax.scatter(data.loc[out_bounds, 'mv'], asr_display[out_bounds],
                   s=3, alpha=0.55, color=_GAUGE_RED, linewidths=0,
                   label=f"{asr_col.upper()} (excessive local median)", zorder=2)
    else:
        # No VEM verdict on this panel, so nothing distinguishes the two
        # groups: one plain scatter, kept out of the legend.
        ax.scatter(data['mv'], asr_display,
                   s=3, alpha=0.45, color=_VE_SCATTER_BLUE, linewidths=0,
                   label='_nolegend_', zorder=2)

    if 'median_ci_lower' in data.columns and 'median_ci_upper' in data.columns:
        band_label = (f"{conf_level*100:.0f}% CI for local median"
                      if conf_level else "CI for local median")
        ax.fill_between(data['mv'],
                        data['median_ci_lower'] + center,
                        data['median_ci_upper'] + center,
                        color=_VE_CI_BLUE, alpha=0.15, linewidth=0,
                        label=band_label, zorder=3)
        ax.plot(data['mv'], data['median_ci_lower'] + center,
                color=_VE_CI_BLUE, lw=0.8, zorder=3)
        ax.plot(data['mv'], data['median_ci_upper'] + center,
                color=_VE_CI_BLUE, lw=0.8, zorder=3)

    # The smoothing belongs to this line, so it is named here rather
    # than in the chart title.
    moving_median_label = f"Moving median {asr_col.upper()}"
    if smoothing_label:
        moving_median_label += f" (window {smoothing_label})"
    ax.plot(data['mv'], data['moving_median'] + center, color='black', lw=1.8,
            label=moving_median_label, zorder=4)
    ax.axhline(center, color='black', lw=0.8, ls=':', alpha=0.4, zorder=1,
               label=f"Median {asr_col.upper()} ({center:.3f})")

    if acceptable_inequity_level:
        ax.axhline(center + acceptable_inequity_level, color=_GAUGE_RED, ls='--',
                   lw=1.3, zorder=1,
                   label=f"Acceptable inequity ±{acceptable_inequity_level*100:g}%")
        ax.axhline(center - acceptable_inequity_level, color=_GAUGE_RED, ls='--',
                   lw=1.3, zorder=1, label='_nolegend_')

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if x_scale == 'log':
        # The default log locator leaves only the decade ticks, which on a
        # residential value range means two labels for the whole axis.
        # Five subdivisions per decade keeps 4-5 reference points on a
        # smart-zoomed axis, which typically spans about one decade.
        ax.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0))
        )

    smart_xlim, smart_ylim = (
        _ve_smart_limits(data, center, asr_col=asr_col,
                         acceptable_inequity_level=acceptable_inequity_level,
                         x_scale=x_scale)
        if smart_zoom else (None, None)
    )
    if smart_xlim is not None:
        ax.set_xlim(*smart_xlim)
    y_range = _ve_y_range(zoom_y, center, smart_ylim)
    if y_range is not None:
        ax.set_ylim(*y_range)

    ax.xaxis.set_major_formatter(FuncFormatter(_currency_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(_ve_x_label(x_scale), fontsize=label_fontsize)
    ax.set_ylabel(asr_col.upper(), fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')

    # 'best' rather than a fixed corner: which corner is free depends on
    # whether the pattern is regressive or progressive. Six entries at
    # the tick size covered a quarter of the plot, so the box is scaled
    # down to about 60% of its natural size -- padding and spacing with
    # it, or the smaller type would just sit in the same box.
    legend = ax.legend(loc='best', fontsize=legend_fontsize or _VEM_LEGEND_SCALE * tick_fontsize,
                       frameon=True, framealpha=0.85, borderpad=0.25,
                       labelspacing=0.3, handlelength=1.4,
                       handletextpad=0.35, borderaxespad=0.3,
                       markerscale=2.0)
    legend.get_frame().set_edgecolor('#e0e0e0')

# ---------------------------------------------------------------------------------------------------------------------

def _vei_boxplot_axes(ax,
                      vei_metrics,
                      short_labels=True,
                      title='',
                      show_outliers=False,
                      label_fontsize=13,
                      tick_fontsize=10):
    """
    Draw the VEI percentile-group boxplot on matplotlib axes.

    This is the static, composable counterpart of :func:`vei_boxplot`,
    drawn straight from the per-group ASR vectors kept in the
    ``vei_analysis`` result.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    vei_metrics : dict
        Flat VEI metrics from ``_unwrap_vei_result``, requiring
        ``'pg_stats'`` entries with 'asr_vector' and 'med_proxy'.
    short_labels : bool, default True
        Abbreviate the median proxy tick labels (e.g. ``'$1.23M'``). Each
        tick also carries the group's CoD on a second line.
    title : str, default ''
        Axes title.
    show_outliers : bool, default False
        Whether to plot points beyond the whiskers. Off by default: in a
        panel-sized axes the fliers stretch the y-axis far enough to
        flatten the group medians the VEI is read from.
    label_fontsize, tick_fontsize : int
        Font sizes for axis labels/title and for tick text.

    Returns
    -------
    None
    """
    pg_stats = (vei_metrics or {}).get('pg_stats') or []
    groups = [np.asarray(group.get('asr_vector'), dtype=float)
              for group in pg_stats if group.get('asr_vector') is not None]

    if not groups:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5,
                'VEI percentile groups unavailable\n(sample too small)',
                ha='center', va='center', fontsize=label_fontsize,
                color='#757575', transform=ax.transAxes)
        if title:
            ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
        return

    positions = np.arange(1, len(groups) + 1)
    boxes = ax.boxplot(groups, positions=positions, widths=0.62,
                       patch_artist=True, showfliers=show_outliers,
                       flierprops=dict(marker='o', markersize=2,
                                       markerfacecolor='#9e9e9e',
                                       markeredgecolor='none', alpha=0.5),
                       medianprops=dict(color='white', linewidth=1.4),
                       whiskerprops=dict(color='#015da8', linewidth=1.0),
                       capprops=dict(color='#015da8', linewidth=1.0))
    for patch in boxes['boxes']:
        patch.set_facecolor('#015da8')
        patch.set_edgecolor('#015da8')
        patch.set_alpha(0.9)

    sample_median = vei_metrics.get('Median')
    if sample_median is None:
        sample_median = float(np.median(np.concatenate(groups)))
    ax.axhline(float(sample_median), ls='--', color=_GAUGE_RED, lw=1.3,
               label=f"Overall median ASR ({float(sample_median):.3f})")

    labels = _format_proxy_cod_labels([group for group in pg_stats
                                       if group.get('asr_vector') is not None],
                                      short=short_labels)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel(_VEI_X_LABEL, fontsize=label_fontsize)
    ax.set_ylabel('ASR', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')

    legend = ax.legend(loc='upper right', fontsize=tick_fontsize,
                       frameon=True, framealpha=0.85)
    legend.get_frame().set_edgecolor('#e0e0e0')

    # Appendix E.5 provenance, matching the standalone `vei_boxplot`: which
    # ranking method drew these groups, since R6 and R7 can differ.
    _method = str(vei_metrics.get('percentile_method') or 'R7').upper()
    _note = f'{len(groups)} quantile groups, {_method} ranking'
    if vei_metrics.get('compare_all_buckets'):
        _note += (f" | widest-gap pair: {vei_metrics.get('group_compared_low')}"
                  f"-{vei_metrics.get('group_compared_high')}")
    ax.text(0.992, 0.02, _note, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#666666', style='italic',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))

# ---------------------------------------------------------------------------------------------------------------------

def _showing_inline():
    """
    Report whether matplotlib figures are rendered inline by the frontend.

    Used so the panel does not render twice in a notebook (once from
    ``plt.show()``, once from the returned figure's display hook) while
    still displaying when called from a plain script.

    Returns
    -------
    bool
        True when the active matplotlib backend draws inline (Jupyter,
        VS Code notebooks, etc.).
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    return 'inline' in backend or 'ipympl' in backend or 'widget' in backend

# ---------------------------------------------------------------------------------------------------------------------

def vertical_equity_panel(df,
                          assessment_header=None,
                          market_value_header=None,
                          mkt_value_proxy_header=None,
                          prb_mvp_header=None,
                          mvp_sale_price_share=0.5,
                          weights_header=None,
                          asr=None,
                          mkt_value_proxy=None,
                          prb_value=None,
                          prb_not_significant=None,
                          prd_value=None,
                          vei_result=None,
                          vem_result=None,
                          cod_value=None,
                          assessment_standards=None,
                          req_significance=0.05,
                          vei_conf_level=0.9,
                          vei_acceptable_range=10,
                          vei_percentile_method='R7',
                          vei_use_log_normal_delta=True,
                          vem_acceptable_inequity_level=0.1,
                          vem_use_mvp=False,
                          vem_window_size=None,
                          vem_conf_level=0.9,
                          vem_zoom_y=True,
                          vem_smart_zoom=True,
                          vem_asr_col='asr',
                          include_vem=True,
                          gauge_outer_multiple=3.0,
                          x_scale='log',
                          title='Vertical Equity Panel',
                          subtitle='',
                          figsize=(8.5, 11.0),
                          return_object=False,
                          graph_filepath='',
                          show=None):
    """
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
    """
    try:
        from ratio_study.indicators import (
            asr_median as _asr_median, cod as _cod, prb as _prb, prd as _prd,
            vem as _vem, vei_analysis as _vei_analysis,
            market_value_proxy as _market_value_proxy,
        )

        standards = {'PRB': (-0.05, 0.05), 'PRD': (0.98, 1.03)}
        standards.update({key: value for key, value
                          in (assessment_standards or {}).items()
                          if key in ('PRB', 'PRD')})

        # --- assemble the arrays the indicators need ---------------------
        needs_data = any(value is None for value in
                         (prb_value, prd_value, vei_result, vem_result))
        data_pd = None
        if needs_data or asr is None:
            if df is None:
                raise ValueError(
                    "`df` is required unless every indicator result is supplied."
                )
            data_pd = nw_utils.require_pandas(df, reason="vertical_equity_panel()")

        asr_arr = None if asr is None else nw_utils.to_numpy_1d(asr).astype(float)
        assessment_arr = market_value_arr = weights_arr = None

        if data_pd is not None:
            if not (assessment_header and market_value_header):
                raise ValueError(
                    "`assessment_header` and `market_value_header` are required "
                    "to compute the vertical equity indicators from `df`."
                )
            for header in (assessment_header, market_value_header,
                           mkt_value_proxy_header, weights_header):
                if header is not None and header not in nw_utils.frame_columns(data_pd):
                    raise ValueError(f"Column '{header}' not found in DataFrame.")

            assessment_arr = data_pd[assessment_header].to_numpy().astype(float)
            market_value_arr = data_pd[market_value_header].to_numpy().astype(float)
            keep = (~np.isnan(assessment_arr) & ~np.isnan(market_value_arr)
                    & (assessment_arr > 0) & (market_value_arr > 0))

            if weights_header:
                weights_arr = data_pd[weights_header].to_numpy().astype(float)
                keep &= ~np.isnan(weights_arr)

            assessment_arr = assessment_arr[keep]
            market_value_arr = market_value_arr[keep]
            if weights_arr is not None:
                weights_arr = weights_arr[keep]
                weights_arr = weights_arr / weights_arr.mean()
            if asr_arr is None:
                asr_arr = assessment_arr / market_value_arr
            else:
                asr_arr = asr_arr[keep]

            if mkt_value_proxy is not None:
                proxy_arr = nw_utils.to_numpy_1d(mkt_value_proxy).astype(float)[keep]
            elif mkt_value_proxy_header:
                proxy_arr = data_pd[mkt_value_proxy_header].to_numpy().astype(float)[keep]
            else:
                # Same proxy as ``ratios``: `mvp_sale_price_share` of the sale
                # amount, the rest the assessment restated at the overall
                # level of assessment.
                level_of_assessment = float(_asr_median(asr_arr, weights=weights_arr))
                proxy_arr = _market_value_proxy(
                    market_value_arr, assessment_arr, asr_md=level_of_assessment,
                    mvp_sale_price_share=mvp_sale_price_share)

            if prb_mvp_header is not None:
                prb_mvp_arr = data_pd[prb_mvp_header].to_numpy().astype(float)[keep]
            else:
                prb_mvp_arr = None
        else:
            proxy_arr = (None if mkt_value_proxy is None
                         else nw_utils.to_numpy_1d(mkt_value_proxy).astype(float))

        # --- fill in whatever was not handed to us -----------------------
        if cod_value is None and asr_arr is not None and len(asr_arr) > 3:
            cod_value = _cod(asr_arr, weights=weights_arr)

        if prd_value is None and market_value_arr is not None:
            prd_value = _prd(market_value_arr, asr_arr, weights=weights_arr)

        if prb_value is None and market_value_arr is not None:
            prb_results = _prb(market_value=market_value_arr, assessment=assessment_arr, asr=asr_arr, market_value_proxy=prb_mvp_arr,
                               req_significance=req_significance,
                               non_significant_marker='*', weights=weights_arr,
                               mvp_sale_price_share=mvp_sale_price_share)
            prb_value = prb_results[0]
            if prb_not_significant is None:
                prb_not_significant = prb_results[1]

        if vei_result is None and asr_arr is not None and proxy_arr is not None:
            vei_result = _vei_analysis(
                asr=asr_arr, mkt_value_proxy=proxy_arr,
                conf_level=vei_conf_level, acceptable_range=vei_acceptable_range,
                percentile_method=vei_percentile_method, output_format='dict',
                metrics_only=True, gen_plot=False, print_settings=False,
                vei_use_log_normal_delta=vei_use_log_normal_delta,
            )
        vei_metrics = _unwrap_vei_result(vei_result)

        if vem_result is None and market_value_arr is not None:
            count = len(market_value_arr)
            window = vem_window_size or (1000 if count > 10000
                                               else max(int(count / 10), 1))
            vem_result = _vem(
                # `vem_use_mvp` decides which values fill this column --
                # the proxy's or the true market value's -- but its name
                # is always `market_value_header`, so `_vem`'s own
                # `market_value_proxy` (the column name it reads) has to
                # name that column, not `mkt_value_proxy_header` -- which
                # is not a column in this frame at all, only in the
                # caller's original `df`.
                df=pd.DataFrame({market_value_header: proxy_arr if vem_use_mvp else market_value_arr,
                                 'asr': asr_arr}),
                market_value_proxy=market_value_header,
                window_size=window,
                conf_level=vem_conf_level,
                acceptable_inequity_level=vem_acceptable_inequity_level,
                generate_plot=False,
            )
        vem_result = vem_result or {}

        specs = _ve_gauge_specs(
            prb_value=prb_value,
            prb_not_significant=bool(prb_not_significant),
            prd_value=prd_value,
            vei_metrics=vei_metrics,
            vem_result=vem_result,
            cod_value=cod_value,
            assessment_standards=standards,
            vei_acceptable_range=vei_acceptable_range,
            vem_acceptable_inequity_level=vem_acceptable_inequity_level,
            outer_multiple=gauge_outer_multiple,
        )
        if not include_vem:
            specs = [spec for spec in specs if spec['label'] != 'VEM']

        # --- lay the panel out -------------------------------------------
        # Letter-page proportions, stacked: gauges, then VEI, then VEM.
        # The gauges get their own gridspec so they stay centred on the
        # page (the charts below need a wider left margin for their
        # y-axis labels) and keep the same dial size whether there are
        # three of them or four.
        with plt.rc_context(_VE_PANEL_RC):
            fig = plt.figure(figsize=figsize)

            gauge_wspace = 0.18
            gauge_cell = (_GAUGE_ROW_SPAN
                          / (_GAUGE_ROW_COLUMNS + (_GAUGE_ROW_COLUMNS - 1) * gauge_wspace))
            gauge_used = len(specs) * gauge_cell * (1 + gauge_wspace) - gauge_cell * gauge_wspace
            gauge_grid = fig.add_gridspec(
                1, len(specs), wspace=gauge_wspace,
                left=0.5 - gauge_used / 2, right=0.5 + gauge_used / 2,
                top=0.905, bottom=0.712,
            )
            # Appendix F's supporting tests, in a quiet strip under the dials:
            # they add insight but carry no IAAO acceptance range, so they get
            # a band rather than a gauge.
            strip_grid = fig.add_gridspec(1, 1, left=0.10, right=0.96,
                                          top=0.694, bottom=0.610)
            # The VEI ticks carry two lines (median proxy over group CoD),
            # so the gap between the charts has to clear both those and
            # the axis label before the VEM title starts.
            chart_grid = fig.add_gridspec(
                # Enough room for the VEI ticks (two lines each) and its
                # axis label to clear the VEM title below them.
                2, 1, hspace=0.58,
                left=0.10, right=0.96, top=0.572, bottom=0.055,
            )

            for column, spec in enumerate(specs):
                _draw_ve_gauge(fig.add_subplot(gauge_grid[0, column]), spec)

            _draw_secondary_strip(
                fig.add_subplot(strip_grid[0]),
                _secondary_indicator_specs(
                    asr_arr, proxy_arr,
                    assessment=assessment_arr, market_value=market_value_arr,
                    vei_metrics=vei_metrics, weights=weights_arr,
                    req_significance=req_significance),
            )

            _vei_boxplot_axes(
                fig.add_subplot(chart_grid[0]), vei_metrics,
                title=_vei_chart_title(vei_metrics),
            )

            vem_title = _vem_chart_title(vem_result, include_verdict=include_vem)
            vem_ax = fig.add_subplot(chart_grid[1])
            if vem_result.get('dataframe') is not None:
                _vem_axes(
                    vem_ax, vem_result['dataframe'],
                    asr_col=vem_asr_col,
                    # Without the VEM gauge there is no verdict on the
                    # panel, so the chart drops everything that reports
                    # one and shows the moving median on its own.
                    acceptable_inequity_level=(vem_acceptable_inequity_level
                                               if include_vem else None),
                    x_scale=x_scale, zoom_y=vem_zoom_y,
                    conf_level=vem_conf_level,
                    center=vem_result.get('asr_center'),
                    smart_zoom=vem_smart_zoom,
                    show_bounds=include_vem,
                    smoothing_label=_vem_smoothing_label(vem_result),
                    title=vem_title,
                )
            else:
                _vem_unavailable_axes(vem_ax, title=vem_title)

            # The title carries the level it describes ("Vertical Equity
            # Panel — property_type: Single family detached (not on water)"),
            # which easily outruns the figure width. Since the panel is saved
            # without a tight bounding box, an over-long line is clipped at
            # both ends rather than expanding the canvas, so wrap it and
            # shrink the type when it is long. Each wrapped line pushes the
            # subtitle down so the two cannot overlap.
            title_size = 18 if len(title) <= 60 else (15 if len(title) <= 100 else 13)
            title_lines = textwrap.wrap(title, width=max(38, int(58 * 18 / title_size))) or ['']
            title_top = 0.985
            fig.suptitle('\n'.join(title_lines), fontsize=title_size,
                         fontweight='bold', y=title_top, va='top')
            if subtitle:
                # Stack the subtitle below the *rendered height* of the title
                # rather than at a fixed offset: a fixed offset put a 12 pt
                # subtitle inside an 18 pt title's line box, so the two
                # touched. Work in points against the figure height, leave a
                # clear gap, and anchor the subtitle by its top so its own
                # height does not eat back into that gap.
                fig_height_pt = fig.get_figheight() * 72.0
                title_block = len(title_lines) * title_size * 1.3 / fig_height_pt
                subtitle_gap = 9.0 / fig_height_pt
                fig.text(0.5, title_top - title_block - subtitle_gap, subtitle,
                         ha='center', va='top', fontsize=12, color='#616161')

        # Saved without a tight bounding box so the image keeps the
        # letter-page proportions `figsize` asks for.
        if return_object:
            imgdata = BytesIO()
            fig.savefig(imgdata, format='png', dpi=150)
            plt.close(fig)
            imgdata.seek(0)
            return [title if not subtitle else f"{title} – {subtitle}",
                    _get_image(imgdata, width=7 * inch)]

        if graph_filepath:
            if not graph_filepath.lower().endswith('.png'):
                graph_filepath = graph_filepath + '.png'
            fig.savefig(graph_filepath, dpi=150)
            plt.close(fig)
            return graph_filepath

        inline = _showing_inline()
        display = (not inline) if show is None else show
        if display:
            plt.show()
        if inline:
            # The returned Figure is auto-rendered by IPython's own display
            # formatter when this call is the last expression in a cell.
            # Leaving the figure open lets the inline backend's end-of-cell
            # flush render the same panel again, so close it here to keep
            # that to one image.
            plt.close(fig)
        return fig

    except Exception as ex:
        raise RuntimeError('error in ratio study: vertical_equity_panel', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _vem_smoothing_label(vem_result) -> str:
    """
    Name the smoothing a ``vem`` result was computed with.

    Parameters
    ----------
    vem_result : dict or None
        Result from :func:`ratio_study.indicators.vem`, read for its
        ``'window_size'`` / ``'quantiles'`` keys.

    Returns
    -------
    str
        e.g. ``'120'`` for a rolling window, ``'10 quantiles'`` for
        quantile groups, or ``''`` when neither is recorded.
    """
    result = vem_result or {}
    window_size = result.get('window_size')
    quantiles = result.get('quantiles')
    if window_size:
        return f"{int(window_size)}"
    if quantiles:
        return f"{int(quantiles)} quantiles"
    return ''

# ---------------------------------------------------------------------------------------------------------------------

def _vei_chart_title(vei_metrics) -> str:
    """
    Title for the VEI chart: the indicator, its value and its verdict.

    Parameters
    ----------
    vei_metrics : dict or None
        Flat VEI metrics as returned by :func:`_unwrap_vei_result`, read
        for ``'vei'`` and ``'conclusion'``.

    Returns
    -------
    str
        e.g. ``'Vertical Equity Indicator: -35.0%, Regressivity'``,
        dropping whichever part is unavailable.
    """
    metrics = vei_metrics or {}
    value, _ = _ve_coerce_float(metrics.get('vei'))
    conclusion = str(metrics.get('conclusion') or '')
    parts = []
    # ``vei_analysis`` reports 0 when it bails out on sample size, which
    # would read as a real result.
    if value is not None and conclusion != 'Sample size too small':
        parts.append(f"{value:+.1f}%")
    if conclusion:
        parts.append(conclusion)
    return 'Vertical Equity Indicator' + (f": {', '.join(parts)}" if parts else '')

# ---------------------------------------------------------------------------------------------------------------------

def _vem_chart_title(vem_result, include_verdict=True) -> str:
    """
    Title for the VEM chart: its verdict, or what the chart shows.

    The smoothing is not named here -- it belongs to the moving median
    itself and rides on that legend entry (see :func:`_vem_axes`).

    Parameters
    ----------
    vem_result : dict or None
        Result from :func:`ratio_study.indicators.vem`.
    include_verdict : bool, default True
        If True, report the conclusion with its severity and prevalence.
        If False (the panel's `include_vem=False`, where no VEM verdict is
        shown at all), describe the chart instead.

    Returns
    -------
    str
        e.g. ``'Vertical Equity: Regressivity, severity: 15.8%,
        prevalence: 44.0%'``, or ``'Moving Median of ASR'`` without the
        verdict. Falls back to the latter when there is no conclusion to
        report. An acceptable result carries the conclusion alone, since
        its severity and prevalence are nil by construction.
    """
    result = vem_result or {}
    conclusion = str(result.get('Inequity_conclusion') or '')

    if not include_verdict or not conclusion:
        return 'Moving Median of ASR'

    parts = [conclusion]
    if conclusion.strip().lower() != _VEM_NO_INEQUITY:
        severity, _ = _ve_coerce_float(result.get('Inequity_severity'))
        prevalence, _ = _ve_coerce_float(result.get('Inequity_prevalence'))
        if severity is not None:
            parts.append(f"severity: {severity:.1f}%")
        if prevalence is not None:
            parts.append(f"prevalence: {prevalence:.1f}%")
    return f"Vertical Equity: {', '.join(parts)}"

# ---------------------------------------------------------------------------------------------------------------------

def _vem_unavailable_axes(ax, title=''):
    """
    Render a placeholder where the VEM moving-median chart would go.

    Used when ``vem`` produced no working frame, e.g. for samples
    below its 30-record minimum.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the placeholder message on.
    title : str, default ''
        Axes title; falls back to the unparameterised chart title.

    Returns
    -------
    None
    """
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, 'VEM moving median unavailable\n(sample too small)',
            ha='center', va='center', fontsize=12, color='#757575',
            transform=ax.transAxes)
    ax.set_title(title or _vem_chart_title(None),
                 fontsize=13, fontweight='bold')

# ---------------------------------------------------------------------------------------------------------------------
# Exploratory chart API
# ---------------------------------------------------------------------------------------------------------------------
#
# The charts above grew out of the PDF pipeline: they take pre-computed
# result objects and hand back ReportLab flowables, which is what the report
# builder wants and the opposite of what an analyst at a notebook prompt
# wants. The functions below are the direct-use surface. They all follow the
# same three rules, so learning one teaches the rest:
#
#   1. Signature starts ``(df, assessment_header, market_value_header)`` --
#      the same first three arguments as :func:`ratio_study.ratios`, so a
#      chart call is a copy-paste away from the study call.
#   2. The ratio is derived internally. No caller ever has to build an 'asr'
#      column, run `vei_analysis` first, or thread a `vem` result through.
#      Pass ``asr_col=`` when a ratio column already exists.
#   3. Output is three-way and identical everywhere:
#        f(...)                        -> matplotlib Figure (renders inline)
#        f(..., graph_filepath='x.png')-> saves, returns the path
#        f(..., return_object=True)    -> ReportLab flowable for the PDF

_ASR_DEFAULT_BANDS = (0.90, 1.10)


def _finish_chart(fig, title, return_object, graph_filepath, show=None):
    """
    The package's standard three-way chart return.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Rendered figure.
    title : str
        Title, used as the flowable's label when `return_object` is True.
    return_object : bool
        If True, return ``[title, ReportLab Image]`` for the PDF report.
    graph_filepath : str
        If set (and `return_object` is False), save a PNG here and return
        the path. ``'.png'`` is appended when missing.
    show : bool, optional
        Force ``plt.show()`` on or off. Left as None, the figure is shown
        only when the frontend is *not* rendering inline, so a notebook
        does not draw it twice.

    Returns
    -------
    list, str or matplotlib.figure.Figure
        Per the three cases above.
    """
    if return_object:
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        return [title, _get_image(buffer)]
    if graph_filepath:
        if not graph_filepath.lower().endswith('.png'):
            graph_filepath += '.png'
        fig.savefig(graph_filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return graph_filepath
    inline = _showing_inline()
    if (not inline) if show is None else show:
        plt.show()
    if inline:
        # The returned Figure is auto-rendered by IPython's own display
        # formatter when this call is the last expression in a cell. Leaving
        # the figure open lets the inline backend's end-of-cell flush render
        # the same panel again, so close it here to keep that to one image.
        plt.close(fig)
    return fig


def _asr_frame(df, assessment_header=None, market_value_header=None, asr_col=None):
    """
    Return a pandas copy of `df` carrying a usable ratio column.

    Every chart in this section needs the same thing -- the frame plus the
    assessment-to-sale ratio -- and callers should not have to build it.
    Either name an existing ratio column with `asr_col`, or give the two
    value columns and have the ratio derived here.

    Parameters
    ----------
    df : DataFrame
        pandas, polars, pyarrow or narwhals frame.
    assessment_header, market_value_header : str, optional
        Assessed value and (time-adjusted) sale price columns. Required
        unless `asr_col` is given.
    asr_col : str, optional
        Name of a ratio column already present in `df`.

    Returns
    -------
    tuple of (pandas.DataFrame, str)
        The frame and the name of the ratio column to plot.

    Raises
    ------
    ValueError
        If neither `asr_col` nor both value columns are usable.
    """
    frame = nw_utils.require_pandas(df, reason='chart helpers use seaborn/matplotlib').copy()

    if asr_col:
        if asr_col not in frame.columns:
            raise ValueError(f"asr_col='{asr_col}' is not a column of the frame.")
        return frame, asr_col

    missing = [name for name, col in (('assessment_header', assessment_header),
                                      ('market_value_header', market_value_header))
               if not col]
    if missing:
        raise ValueError(
            'Pass assessment_header and market_value_header (or an existing '
            f"asr_col). Missing: {', '.join(missing)}.")
    absent = [c for c in (assessment_header, market_value_header) if c not in frame.columns]
    if absent:
        raise ValueError(f"Column(s) not found: {absent}")

    market = pd.to_numeric(frame[market_value_header], errors='coerce')
    assessment = pd.to_numeric(frame[assessment_header], errors='coerce')
    frame['asr'] = np.where(market > 0, assessment / market.replace(0, np.nan), np.nan)
    return frame, 'asr'


def _asr_zoom_limits(x_values, ratio_values, log_x=False,
                     x_quantiles=(0.01, 0.99), y_coverage=0.95, keep=()):
    """
    Axis limits that frame the bulk of a ratio scatter, ignoring the tails.

    Sale prices have a long right tail and the ASR has outliers at both ends,
    so the full data range spends most of the plot area on a handful of
    points. This is the same trimming the VEM chart uses (`_ve_smart_limits`),
    reduced to plain x/y arrays so the scatter and facet charts can share it.

    Parameters
    ----------
    x_values, ratio_values : array-like
        The two axes, same length.
    log_x : bool, default False
        Pad the x range multiplicatively and drop non-positive values.
    x_quantiles : tuple, default (0.01, 0.99)
        Inner quantile range of x to show.
    y_coverage : float, default 0.95
        Share of the in-window ratios to cover, centred on the median.
    keep : iterable of float
        Ratio values that must stay visible whatever the trim decides --
        the median line, a lowess trend, and so on.

    Returns
    -------
    tuple
        ``(xlim, ylim)``, either of which may be None when there is nothing
        usable to measure.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(ratio_values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if log_x:
        ok &= x > 0
    x, y = x[ok], y[ok]
    if x.size < 10:
        return None, None

    lo_q, hi_q = np.quantile(x, x_quantiles)
    if not np.isfinite(lo_q) or not np.isfinite(hi_q) or hi_q <= lo_q:
        xlim = None
        window = np.ones_like(y, dtype=bool)
    else:
        pad = ((hi_q / lo_q) ** 0.03 if log_x and lo_q > 0 else None)
        xlim = ((lo_q / pad, hi_q * pad) if pad
                else (lo_q - (hi_q - lo_q) * 0.03, hi_q + (hi_q - lo_q) * 0.03))
        window = (x >= lo_q) & (x <= hi_q)

    inside = y[window]
    if inside.size < 5:
        return xlim, None
    tail = (1 - y_coverage) / 2
    low, high = np.quantile(inside, [tail, 1 - tail])
    for value in keep:
        if value is not None and np.isfinite(value):
            low, high = min(low, value), max(high, value)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return xlim, None
    margin = (high - low) * 0.05
    return xlim, (low - margin, high + margin)


def _draw_asr_reference(ax, median, bands=_ASR_DEFAULT_BANDS, horizontal=True):
    """
    Draw the median ratio and the level-of-valuation band on an axis.

    The IAAO Standard on Ratio Studies calls 0.90-1.10 acceptable for the level
    of valuation for any class of property, so the band gives every ratio
    chart the same acceptable frame of reference.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    median : float
        Sample median ratio, drawn as a solid line.
    bands : tuple of (float, float) or None
        Lower/upper acceptable level, drawn as dashed lines. None to skip.
    horizontal : bool, default True
        True for charts with the ratio on the y axis, False for the x axis.
    """
    line = ax.axhline if horizontal else ax.axvline
    if median is not None and np.isfinite(median):
        line(median, color='#1565c0', linewidth=1.6, zorder=3,
             label=f'Median {median:.4f}')
    if bands:
        for edge in bands:
            line(edge, color='#c62828', linewidth=1.1, linestyle='--', zorder=3)
        # One legend entry for the pair, via an empty proxy artist -- a
        # zero-width dashed line is rejected by matplotlib.
        ax.plot([], [], color='#c62828', linewidth=1.1, linestyle='--',
                label=f'Acceptable level {bands[0]:.2f}-{bands[1]:.2f}')


def asr_scatter(df,
                assessment_header=None,
                market_value_header=None,
                x=None,
                *,
                asr_col=None,
                x_label=None,
                y_label='Assessment-to-sale ratio',
                log_x=False,
                lowess=True,
                zoom=True,
                bands=None,
                point_size=8,
                alpha=0.35,
                marginals=True,
                marginal_bins=120,
                title=None,
                figsize=(9.5, 6.5),
                return_object=False,
                graph_filepath='',
                show=None):
    """
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
    """
    frame, ratio_col = _asr_frame(df, assessment_header, market_value_header, asr_col)
    x = x or market_value_header
    if x is None:
        raise ValueError('Pass x= (or market_value_header, used as the default x).')
    if x not in frame.columns:
        raise ValueError(f"Column not found: '{x}'")

    frame = frame[[x, ratio_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if log_x:
        frame = frame[frame[x] > 0]
    if len(frame) < 2:
        raise ValueError('Not enough finite observations to plot.')

    x_label = x_label if x_label is not None else str(x).replace('_', ' ').capitalize()
    median = float(frame[ratio_col].median())
    title = title or f'Ratio by {x_label.lower()}'

    fig = plt.figure(figsize=figsize)
    if marginals:
        grid = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 5),
                                wspace=0.04, hspace=0.05)
        ax = fig.add_subplot(grid[1, 0])
        ax_top = fig.add_subplot(grid[0, 0], sharex=ax)
        ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
        for spine_ax in (ax_top, ax_right):
            spine_ax.tick_params(labelbottom=False, labelleft=False,
                                 bottom=False, left=False)
            for side in spine_ax.spines.values():
                side.set_visible(False)
    else:
        ax = fig.add_subplot(111)
        ax_top = ax_right = None

    ax.scatter(frame[x], frame[ratio_col], s=point_size, alpha=alpha,
               color='#3949ab', linewidths=0, zorder=2)
    _draw_asr_reference(ax, median, bands)

    if lowess and len(frame) >= 10:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
            smoothed = _lowess(frame[ratio_col].to_numpy(), frame[x].to_numpy(),
                               frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color='#ff8f00',
                    linewidth=2.4, zorder=4, label='Lowess trend')
        except Exception:
            # A missing smoother is not worth losing the scatter over.
            pass

    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(_currency_tick))

    # Trim the axes to where the points are. Without this a handful of
    # very large sales, or one ratio of 6, takes most of the plot area and
    # the cloud that carries the finding is squeezed into a corner.
    if zoom:
        xlim, ylim = _asr_zoom_limits(frame[x], frame[ratio_col], log_x=log_x,
                                      keep=(median,) + tuple(bands or ()))
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    if marginals:
        ax_top.hist(frame[x], bins=marginal_bins, color='#9fa8da')
        ax_right.hist(frame[ratio_col], bins=marginal_bins, orientation='horizontal',
                      color='#9fa8da')
        ax_top.set_title(title, fontsize=14, fontweight='bold', pad=10)
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')

    return _finish_chart(fig, title, return_object, graph_filepath, show)


def asr_histogram(df,
                  assessment_header=None,
                  market_value_header=None,
                  *,
                  asr_col=None,
                  bins=60,
                  clip=(0.0, 2.0),
                  bands=_ASR_DEFAULT_BANDS,
                  title=None,
                  x_label='Assessment-to-sale ratio',
                  figsize=(9.5, 5.5),
                  return_object=False,
                  graph_filepath='',
                  show=None):
    """
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
    """
    frame, ratio_col = _asr_frame(df, assessment_header, market_value_header, asr_col)
    ratios_series = pd.to_numeric(frame[ratio_col], errors='coerce').dropna()
    if ratios_series.empty:
        raise ValueError('No finite ratios to plot.')

    median = float(ratios_series.median())
    drawn = ratios_series
    excluded = 0
    if clip:
        drawn = ratios_series[(ratios_series >= clip[0]) & (ratios_series <= clip[1])]
        excluded = len(ratios_series) - len(drawn)

    title = title or 'Distribution of assessment-to-sale ratios'
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(drawn, bins=bins, color='#5c6bc0', edgecolor='white', linewidth=0.4)
    _draw_asr_reference(ax, median, bands, horizontal=False)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    ax.legend(loc='upper right', fontsize=9)
    footer = f'{len(ratios_series):,} sales'
    if excluded:
        footer += f'  ({excluded:,} outside {clip[0]:g}-{clip[1]:g} not drawn)'
    ax.text(0.99, -0.14, footer, transform=ax.transAxes, ha='right',
            va='top', fontsize=9, color='#616161')

    return _finish_chart(fig, title, return_object, graph_filepath, show)


def asr_boxplot(df,
                assessment_header=None,
                market_value_header=None,
                by=None,
                *,
                asr_col=None,
                min_count=5,
                max_groups=25,
                order_by='group',
                bands=_ASR_DEFAULT_BANDS,
                show_values=False,
                value_fmt='{:.3f}',
                title=None,
                x_label=None,
                figsize=(11.0, 6.0),
                return_object=False,
                graph_filepath='',
                show=None):
    """
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
    """
    if not by:
        raise ValueError("Pass by='<column>' -- the category to split the ratio on.")
    frame, ratio_col = _asr_frame(df, assessment_header, market_value_header, asr_col)
    if by not in frame.columns:
        raise ValueError(f"Column not found: '{by}'")

    frame = frame[[by, ratio_col]].dropna()
    counts = frame[by].value_counts()
    keep = counts[counts >= min_count]
    dropped_small = int(len(counts) - len(keep))
    dropped_excess = 0
    if len(keep) > max_groups:
        dropped_excess = int(len(keep) - max_groups)
        keep = keep.nlargest(max_groups)
    frame = frame[frame[by].isin(keep.index)]
    if frame.empty:
        raise ValueError(f'No category of {by!r} has at least {min_count} sales.')

    stats = frame.groupby(by, observed=True)[ratio_col].agg(['median', 'count'])
    if order_by == 'median':
        order = stats.sort_values('median').index.tolist()
    elif order_by == 'count':
        order = stats.sort_values('count', ascending=False).index.tolist()
    else:
        order = sorted(stats.index.tolist(), key=str)

    median = float(frame[ratio_col].median())
    x_label = x_label if x_label is not None else str(by).replace('_', ' ').capitalize()
    title = title or f'Ratio by {x_label.lower()}'

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot([frame.loc[frame[by] == group, ratio_col].to_numpy() for group in order],
               tick_labels=[f'{group}\n(n={int(stats.loc[group, "count"]):,})' for group in order],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#c5cae9', edgecolor='#3949ab'),
               medianprops=dict(color='#1a237e', linewidth=1.8),
               whiskerprops=dict(color='#3949ab'), capprops=dict(color='#3949ab'))
    _draw_asr_reference(ax, median, bands)

    # The box shows the shape; the number is what goes in the report. Printing
    # the group median above each box saves reading it off the axis, which is
    # the §11.2.1 comparison a reader is actually making -- each stratum's
    # level against the overall one.
    if show_values:
        for position, group in enumerate(order, start=1):
            group_median = float(stats.loc[group, 'median'])
            ax.annotate(value_fmt.format(group_median),
                        xy=(position, group_median), xytext=(0, 7),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold', color='#1a237e',
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.8, pad=1.2), zorder=6)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Assessment-to-sale ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    ax.legend(loc='upper right', fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    # Never silently truncate: say what was left out.
    if dropped_small or dropped_excess:
        note = []
        if dropped_small:
            note.append(f'{dropped_small} categor{"y" if dropped_small == 1 else "ies"} under {min_count} sales')
        if dropped_excess:
            note.append(f'{dropped_excess} beyond the largest {max_groups}')
        # Figure coordinates, not axis: rotated category labels can run well
        # past the axis box and would collide with a note pinned below it.
        fig.text(0.99, 0.005, 'Not shown: ' + '; '.join(note),
                 ha='right', va='bottom', fontsize=9, color='#616161')

    return _finish_chart(fig, title, return_object, graph_filepath, show)


def _drop_thin_facets(frame, col, row, tile_min_records):
    """
    Drop the facets holding too few sales to plot, and say which went.

    A panel drawn from four sales still gets a lowess curve and a full set
    of axes, and reads like the panel next to it drawn from four hundred.
    Removing those panels is the honest layout; naming them in a note is
    what keeps the removal from being silent.

    Parameters
    ----------
    frame : DataFrame
        Facet frame, already cleaned of missing keys.
    col, row : str or None
        Facet dimensions. A tile is a `col` value, or a `row`-`col` pair
        when both are given.
    tile_min_records : int
        Smallest sample a tile may be drawn from.

    Returns
    -------
    DataFrame
        The frame without the thin tiles.
    """
    keys = [c for c in (row, col) if c]
    counts = frame.groupby(keys, observed=True).size()
    thin = counts[counts < tile_min_records]
    if thin.empty:
        return frame

    dimensions = f'{row} / {col}' if row else str(col)
    lines = [f'asr_facet: {len(thin)} {dimensions} panel(s) not displayed -- '
             f'fewer than {tile_min_records:,} records, too few to read a pattern from:']
    for key, n in thin.items():
        # A one-column groupby indexes by scalar on some pandas versions
        # and by 1-tuple on others.
        values = key if isinstance(key, tuple) else (key,)
        label = (f'{row}={values[0]}, {col}={values[-1]}' if row
                 else f'{col}={values[-1]}')
        lines.append(f'  {label} : {n:,} record(s)')
    print('\n'.join(lines))

    thin_keys = set(thin.index)
    if row:
        keep = [pair not in thin_keys for pair in zip(frame[row], frame[col])]
    else:
        keep = ~frame[col].isin(thin_keys)
    return frame[keep]


def _widen_to_span(low, high, centre, span, log_scale=False):
    """
    Widen an axis window to a given span without letting go of its data.

    Per-panel windows only stay comparable if every panel is the same
    number of dollars (or ratio points) wide; the panel's own centre is
    what makes it readable. This grows the narrow windows to the common
    span around `centre`, then slides the window back if that centring
    would have pushed data off the edge.

    Parameters
    ----------
    low, high : float
        The window the panel's own data asks for.
    centre : float
        Where to centre the widened window -- the panel's median.
    span : float or None
        Target width. Ignored when it is no wider than the window.
    log_scale : bool, default False
        Measure and centre multiplicatively, for a log axis.

    Returns
    -------
    tuple of (float, float)
        The widened window.
    """
    if log_scale:
        if min(low, high, centre) <= 0:
            return low, high
        start, stop = _widen_to_span(np.log10(low), np.log10(high),
                                     np.log10(centre), span)
        return 10.0 ** start, 10.0 ** stop

    if span is None or not np.isfinite(span) or span <= (high - low):
        return low, high
    half = span / 2.0
    start, stop = centre - half, centre + half
    # A median sitting well off-centre in its own window -- the usual case
    # for a right-skewed price distribution -- would otherwise clip the
    # long side. Slide, rather than clip.
    if start > low:
        start, stop = low, low + span
    elif stop < high:
        start, stop = high - span, high
    return start, stop


def _tile_zoom_limits(frame, x, ratio_col, col, row, log_x=False,
                      match_y=True, fallback=(None, None)):
    """
    One zoom window per facet, centred on that facet's own medians.

    A window measured across the whole frame is centred on nothing in
    particular: a neighbourhood assessed at 0.75 gets its cloud shoved
    against the bottom of the panel, and a cheap neighbourhood gets its
    prices squeezed into the left inch of an axis stretched by an
    expensive one. Measuring each panel separately fixes both, and
    holding the span common across panels keeps the slopes comparable --
    which is the whole reason to draw a grid.

    Parameters
    ----------
    frame : DataFrame
        The facet frame.
    x, ratio_col : str
        Axis columns.
    col, row : str or None
        Facet dimensions.
    log_x : bool, default False
        Measure the x span multiplicatively.
    match_y : bool, default True
        Give every panel the same y span. False lets each panel size its
        own y axis, which reads shape at the cost of comparable levels.
    fallback : tuple, default (None, None)
        ``(xlim, ylim)`` for panels with too little data to measure.

    Returns
    -------
    dict
        ``{(row_value, col_value): (xlim, ylim)}``, with ``row_value``
        None when there is no `row`. Either limit may be None.
    """
    keys = [c for c in (row, col) if c]
    windows = {}
    for key, group in frame.groupby(keys, observed=True):
        # Iterating a one-column groupby yields a 1-tuple on some pandas
        # versions and a scalar on others.
        values = key if isinstance(key, tuple) else (key,)
        row_value, col_value = (values[0], values[-1]) if row else (None, values[-1])
        xlim, ylim = _asr_zoom_limits(group[x], group[ratio_col], log_x=log_x)
        windows[(row_value, col_value)] = [
            xlim or fallback[0],
            ylim or fallback[1],
            float(group[x].median()),
            float(group[ratio_col].median()),
        ]

    def _widest(index, log_scale):
        spans = []
        for window in (w[index] for w in windows.values()):
            if not window:
                continue
            low, high = window
            if not log_scale:
                spans.append(high - low)
            elif low > 0 and high > 0:
                spans.append(np.log10(high) - np.log10(low))
        return max(spans) if spans else None

    # A common x span only makes sense multiplicatively. On a linear price
    # axis, matching a $200k stratum to a $2M one would leave the cheap
    # panel a sliver of data in a window running well below zero, which is
    # the squeeze per-panel zoom exists to undo. On a log axis the span is
    # a number of decades, and sharing it costs nothing.
    x_span = _widest(0, log_x) if log_x else None
    y_span = _widest(1, False) if match_y else None

    limits = {}
    for tile, (xlim, ylim, x_centre, y_centre) in windows.items():
        if xlim:
            xlim = _widen_to_span(xlim[0], xlim[1], x_centre, x_span, log_scale=log_x)
        if ylim:
            ylim = _widen_to_span(ylim[0], ylim[1], y_centre, y_span)
        limits[tile] = (xlim, ylim)
    return limits


def _facet_axes(grid, col_order, row_order):
    """
    Pair each panel of a FacetGrid with the facet values it stands for.

    Parameters
    ----------
    grid : seaborn.FacetGrid
        The grid, already built with these orders.
    col_order : list
        Column values, in the order given to the grid.
    row_order : list or None
        Row values, or None when the grid has no row dimension.

    Returns
    -------
    dict
        ``{(row_value, col_value): Axes}``, with ``row_value`` None when
        there is no row dimension.
    """
    if row_order is None:
        return {(None, value): ax for value, ax in zip(col_order, grid.axes.flat)}
    return {(row_value, col_value): grid.axes[i, j]
            for i, row_value in enumerate(row_order)
            for j, col_value in enumerate(col_order)}


def asr_facet(df,
              assessment_header=None,
              market_value_header=None,
              x=None,
              *,
              col=None,
              row=None,
              asr_col=None,
              col_wrap=3,
              max_facets=12,
              tile_min_records=30,
              log_x=False,
              lowess=True,
              zoom=True,
              bands=None,
              share_y=True,
              height=3.1,
              aspect=1.25,
              title=None,
              x_label=None,
              return_object=False,
              graph_filepath='',
              show=None):
    """
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
    """
    if not col:
        raise ValueError("Pass col='<column>' -- the stratum each panel represents.")
    frame, ratio_col = _asr_frame(df, assessment_header, market_value_header, asr_col)
    x = x or market_value_header
    if x is None:
        raise ValueError('Pass x= (or market_value_header, used as the default x).')
    for needed in [c for c in (x, col, row) if c]:
        if needed not in frame.columns:
            raise ValueError(f"Column not found: '{needed}'")

    keep_cols = [c for c in (x, ratio_col, col, row) if c]
    frame = frame[keep_cols].dropna()
    frame[x] = pd.to_numeric(frame[x], errors='coerce')
    frame = frame.dropna(subset=[x])
    if log_x:
        frame = frame[frame[x] > 0]

    if tile_min_records:
        if tile_min_records < 0:
            raise ValueError('tile_min_records must be 0 (draw every panel) or a positive count.')
        # Thin panels go before the max_facets cap, so the cap spends its
        # budget on panels that will actually be drawn.
        frame = _drop_thin_facets(frame, col, row, int(tile_min_records))
        if frame.empty:
            raise ValueError(f'No {col} panel has {tile_min_records:,} or more records. '
                             'Lower tile_min_records, or facet on a coarser column.')

    counts = frame[col].value_counts()
    dropped = 0
    if len(counts) > max_facets:
        dropped = int(len(counts) - max_facets)
        frame = frame[frame[col].isin(counts.nlargest(max_facets).index)]
    if frame.empty:
        raise ValueError('Nothing left to plot after filtering.')

    median = float(frame[ratio_col].median())
    x_label = x_label if x_label is not None else str(x).replace('_', ' ').capitalize()
    title = title or f'Ratio by {x_label.lower()}, faceted by {str(col).replace("_", " ")}'

    # A panel centred on its own medians needs its own limits, which
    # matplotlib cannot give a shared axis. The common span computed by
    # `_tile_zoom_limits` is what carries comparability instead.
    per_tile_zoom = str(zoom).strip().lower() not in ('false', 'none', '', 'overall')
    col_order = sorted(frame[col].dropna().unique(), key=str)
    row_order = sorted(frame[row].dropna().unique(), key=str) if row else None

    # A `row` and `col` together facet on their full cross product: most
    # (row, col) pairs usually have no sales at all, and seaborn still
    # reserves a blank panel for each one. Folding the two dimensions into
    # a single facet column -- built only from the pairs that actually
    # occur in `frame`, ordered the same way -- packs the real panels
    # without gaps, at the same width the two-dimensional grid would have
    # used (`len(col_order)`), rather than leaving them scattered through
    # a mostly-empty grid.
    if row:
        facet_col, facet_row = '_asr_facet_combo', None
        facet_wrap = len(col_order)
        frame[row] = pd.Categorical(frame[row], categories=row_order, ordered=True)
        frame[col] = pd.Categorical(frame[col], categories=col_order, ordered=True)
        frame[facet_col] = frame[row].astype(str) + ' | ' + frame[col].astype(str)
        facet_col_order = list(frame.sort_values([row, col])[facet_col].drop_duplicates())
        facet_row_order = None
    else:
        facet_col, facet_row = col, None
        facet_wrap = col_wrap
        facet_col_order, facet_row_order = col_order, None

    sns.set_theme(style='darkgrid')
    grid = sns.FacetGrid(
        frame, col=facet_col, row=facet_row,
        col_wrap=facet_wrap,
        height=height, aspect=aspect,
        sharey=share_y and not per_tile_zoom,
        sharex=not per_tile_zoom,
        col_order=facet_col_order, row_order=facet_row_order)
    grid.map_dataframe(sns.scatterplot, x=x, y=ratio_col, s=9,
                       alpha=0.35, color='#3949ab', linewidth=0)
    if lowess:
        grid.map_dataframe(sns.regplot, x=x, y=ratio_col, scatter=False,
                           lowess=True, line_kws={'color': '#ff8f00', 'linewidth': 2})

    for ax in grid.axes.flat:
        if bands:
            for edge in bands:
                ax.axhline(edge, color='#c62828', linewidth=1.0, linestyle='--')
        ax.axhline(median, color='#1565c0', linewidth=1.4)
        if log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(FuncFormatter(_currency_tick))
            # A per-panel window often spans less than a decade, leaving a
            # single major tick. Label the 2x and 5x minor ticks as prices
            # too -- and only those, since nine labelled minor ticks in a
            # panel this narrow would overlap into a smear.
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
            ax.xaxis.set_minor_formatter(FuncFormatter(_currency_tick))

    # Panels are small, so a single outlying sale costs proportionally more
    # of the plot than it does on a full-size scatter. Trim to the bulk --
    # per panel, so each one is centred on the stratum it describes rather
    # than on a frame-wide median no single panel sits at.
    if zoom:
        overall = _asr_zoom_limits(frame[x], frame[ratio_col], log_x=log_x,
                                   keep=(median,) + tuple(bands or ()))
        if per_tile_zoom:
            limits = _tile_zoom_limits(frame, x, ratio_col, facet_col, facet_row,
                                       log_x=log_x, match_y=share_y,
                                       fallback=overall)
            for tile, ax in _facet_axes(grid, facet_col_order, facet_row_order).items():
                xlim, ylim = limits[tile]
                if xlim:
                    ax.set_xlim(*xlim)
                if ylim:
                    ax.set_ylim(*ylim)
        else:
            for ax in grid.axes.flat:
                if overall[0]:
                    ax.set_xlim(*overall[0])
                if overall[1]:
                    ax.set_ylim(*overall[1])

    grid.set_axis_labels(x_label, 'Assessment-to-sale ratio')
    # `facet_col` already carries the full 'row | col' label when a `row`
    # was given (see above), so a single col_template covers both cases.
    grid.set_titles(col_template='{col_name}')
    if not zoom:
        zoom_note = ''
    elif per_tile_zoom:
        zoom_note = ('   |   each panel centred on its own medians'
                     + (', common y scale' if share_y else ''))
    else:
        zoom_note = '   |   axes trimmed to the bulk of the data'
    subtitle = (f'Blue line: overall median {median:.4f}'
                + (f'   |   red dashes: {bands[0]:.2f}-{bands[1]:.2f} acceptable level' if bands else '')
                + zoom_note)
    if dropped:
        subtitle += f'   |   {dropped} smaller {col} value(s) not shown'
    # Stack title, subtitle and the panel grid in points against the figure
    # height. A grid one row tall and a grid four rows tall have very
    # different figure heights, so any fixed fraction collides on one of them.
    figure_height_pt = grid.figure.get_figheight() * 72.0
    title_top = 0.995
    subtitle_y = title_top - (15 * 1.3 + 8.0) / figure_height_pt
    grid.figure.suptitle(title, fontsize=15, fontweight='bold',
                         y=title_top, va='top')
    grid.figure.text(0.5, subtitle_y, subtitle, ha='center', va='top',
                     fontsize=10, color='#616161')
    # Clear the subtitle's own height plus room for the facet titles.
    grid.figure.subplots_adjust(top=subtitle_y - (10 * 1.3 + 20.0) / figure_height_pt)

    return _finish_chart(grid.figure, title, return_object, graph_filepath, show)


def lorenz_plot(df,
                assessment_header=None,
                market_value_header=None,
                *,
                weights_header=None,
                title=None,
                graph_filepath='',
                show=None):
    """
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
    """
    from ratio_study.gini import gini_stats, gini_lorenz_plot

    frame = nw_utils.require_pandas(df, reason='lorenz_plot() uses matplotlib')
    for needed in (assessment_header, market_value_header):
        if not needed or needed not in frame.columns:
            raise ValueError('Pass assessment_header and market_value_header '
                             'naming columns present in the frame.')

    weights = frame[weights_header] if weights_header else None
    stats = gini_stats(frame[market_value_header], frame[assessment_header],
                       weights=weights, run_regression=False)
    return gini_lorenz_plot(frame[market_value_header], frame[assessment_header],
                            stats['Gini_Sale_Prices'], stats['Concentration_Index'],
                            title or 'Lorenz and concentration curves',
                            filepath=graph_filepath)


def vem_plot(df,
             assessment_header=None,
             market_value_header=None,
             *,
             asr_col=None,
             quantiles=None,
             window_size=None,
             conf_level=0.90,
             ci_method='binom',
             acceptable_inequity_level=0.10,
             x_scale='log',
             title='Vertical Equity Moving Median',
             plot_ref=None,
             graph_filepath=None):
    """
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
    """
    from ratio_study.indicators import vem as _vem

    frame, ratio_col = _asr_frame(df, assessment_header, market_value_header, asr_col)
    if ratio_col != 'asr':
        frame = frame.assign(asr=frame[ratio_col])
    proxy = market_value_header
    if proxy is None or proxy not in frame.columns:
        raise ValueError('vem_plot() needs market_value_header: the moving '
                         'median is taken across the value range.')

    result = _vem(frame, market_value_proxy=proxy, asr_col='asr',
                  quantiles=quantiles, window_size=window_size,
                  conf_level=conf_level, ci_method=ci_method,
                  acceptable_inequity_level=acceptable_inequity_level,
                  x_scale=x_scale, generate_plot=True,
                  plot_return_object=graph_filepath is None,
                  plot_title=title, plot_ref=plot_ref,
                  plot_filepath=graph_filepath)
    return result['plot']
