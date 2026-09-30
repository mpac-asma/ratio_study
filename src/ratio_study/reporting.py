import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from reportlab.platypus import *
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from typing import Optional, Union

import os
import re
from datetime import datetime, timezone

import traceback
from ratio_study.confint import _check_acceptance, _compare_ci
from ratio_study.indicators import VEI_INEQUITY_CONCLUSIONS
from ratio_study.helpers import save_errors_to_text, custom_format, _use_agg_if_headless
from . import nw_utils

# -------------------------------
# Reporting functions
# -------------------------------

class ConditionalSpacer(Spacer):
    """ ConditionalSpacer: Class overrides method in platypus flowables
        Used to determine available space in flowable
    """

    def wrap(self, availWidth, availHeight):
        try:
            height = min(self.height, availHeight - 1e-8)
            return availWidth, height
        except Exception as ex:
            raise RuntimeError('error in ratio study: ConditionalSpacer > wrap:', ex)
        
# ---------------------------------------------------------------------------------------------------------------------

def _blank_missing_for_excel(frame):
    """
    Replace missing values with empty strings, whatever the column dtype.

    ``DataFrame.fillna('')`` raises ``TypeError: Invalid value '' for dtype
    boolean`` on nullable-extension columns -- which is exactly what the
    ``<indicator>_acceptable`` flags from `_acceptance_binaries` are, and they
    carry ``pd.NA`` wherever a group was too small for the indicator to be
    computed. Casting to ``object`` first gives every column somewhere to put
    the blank, and leaves numbers as numbers for the Excel writer.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame to blank out. Not modified in place.

    Returns
    -------
    pandas.DataFrame
        Copy with every NA replaced by ``''``.
    """
    return frame.astype(object).where(frame.notna(), '')


def _export_dataframes_as_excel_tabs(dict_of_dataframes
                                    , filepath
                                    , excel_filename
                                    , sensitivity_tag=None
                                    , site_id=None):
    """
    Generate an Excel file with one tab per dataframe provided.

    Parameters
    ----------
    dict_of_dataframes : dict
        Mapping of sheet name to either a frame-like object (pandas,
        polars, pyarrow or narwhals DataFrame) or a dict containing the
        frame under key ``'dataframe'``. Each frame-like value is
        coerced to a real pandas DataFrame because ``pd.ExcelWriter``
        with the ``xlsxwriter`` engine is pandas-only.
    filepath : str
        Directory in which to save the Excel file.
    excel_filename : str
        Filename of the Excel file to create.
    sensitivity_tag : str, optional, default None
        Sensitivity label tag written as custom document properties
        (e.g. MSIP labels). Requires ``site_id`` to also be provided.
    site_id : str, optional, default None
        Site ID associated with ``sensitivity_tag``. Requires
        ``sensitivity_tag`` to also be provided.

    Returns
    -------
    None
        The workbook is written to disk at ``filepath/excel_filename``;
        nothing is returned.
    """
    try:
        if len(dict_of_dataframes.keys()) > 0:
            # Coerce all frame values to pandas for the xlsxwriter engine.
            for key in list(dict_of_dataframes.keys()):
                if isinstance(dict_of_dataframes[key], dict) and 'dataframe' in dict_of_dataframes[key]:
                    frame = dict_of_dataframes[key]['dataframe']
                    if not isinstance(frame, pd.DataFrame):
                        dict_of_dataframes[key]['dataframe'] = nw_utils.require_pandas(
                            frame, reason="_export_dataframes_as_excel_tabs() uses pd.ExcelWriter"
                        )
                else:
                    if not isinstance(dict_of_dataframes[key], pd.DataFrame):
                        dict_of_dataframes[key] = nw_utils.require_pandas(
                            dict_of_dataframes[key], reason="_export_dataframes_as_excel_tabs()"
                        )
            with pd.ExcelWriter(os.path.join(filepath, excel_filename), engine="xlsxwriter") as writer:
                    workbook = writer.book
                    bold_format = workbook.add_format({'bold': True})
                    format_center = workbook.add_format({'align': 'center'})
                    for dataframe in dict_of_dataframes.keys():
                        sheetname = 'Gini Decomposition by Factors' if dataframe == 'gini_factors' else dataframe[:30]
                        if 'dataframe' in dict_of_dataframes[dataframe]:
                            _blank_missing_for_excel(dict_of_dataframes[dataframe]['dataframe']).to_excel(writer, sheet_name=sheetname, index=False)
                            worksheet = writer.sheets[sheetname]
                            for i, col in enumerate(dict_of_dataframes[dataframe]['dataframe'].columns, start=0):
                                max_len = max(dict_of_dataframes[dataframe]['dataframe'][col].astype(str).map(len).max(), len(col))
                                worksheet.set_column(i, i, max_len, format_center)
                            worksheet.set_row(0, None, bold_format)
                            worksheet.set_column(0, 0, None, bold_format)
                            worksheet.freeze_panes(1, 1)
                    if sensitivity_tag and site_id:
                        worksheet = _add_excel_sensitivity_tag(workbook, writer, sensitivity_tag, site_id)

        # else:
        #     if gen_log:
        #         logging.warning('No sheets available to export to Excel. Please check input data')
    except Exception as ex:
        raise RuntimeError('error in ratio study: export_dataframes_as_excel_tabs', ex)


# -------------------------------
# Gen PDF functions
# -------------------------------

def _pdf_create_table(input_df):
    """
    Convert a dataframe into a reportlab Table flowable. Credits to James Gilbert.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Dataframe, typically grouped by property code, whose column
        headers and rows will be rendered as the table.

    Returns
    -------
    reportlab.platypus.Table
        Table flowable, left-aligned with a bold header row, ready to
        append to a PDF document's elements list.
    """
    try:
        ts = [('ALIGN', (0, 0), (-1, -1), 'CENTRE'),
              ('FONT', (0, 0), (-1, 0), 'Times-Bold'),
              ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
              ('BOX', (0, 0), (-1, -1), 0.25, colors.black)]

        list_table = [input_df.columns[:, ].values.astype(str).tolist()] + input_df.values.tolist()
        table = Table(list_table, repeatRows=1, style=ts)
        table.hAlign = "LEFT"
        return table
    except Exception as ex:
        raise RuntimeError('error in ratio study: pdf_create_table', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _pdf_breakdown_ratios_by_grouping(df, output_columns, select_groupings=[], return_list=False):
    """
    Break down a ratios dataframe by grouping for use in tables and graphs.

    Parameters
    ----------
    df : pandas.DataFrame
        Ratios dataframe containing at least a ``'Level'`` column (and
        typically a ``'Group'`` column) to break down by.
    output_columns : list of str
        Columns to include for each grouping. ``'Count'`` and
        ``'Group'`` are automatically prepended when present in ``df``
        and not already listed.
    select_groupings : list of str, optional, default []
        Specific values of ``df['Level']`` to break out. If empty, all
        unique values of ``df['Level']`` are used.
    return_list : bool, default False
        If True, return each grouping as a ``[keys, values]`` pair
        instead of a DataFrame.

    Returns
    -------
    dict
        Mapping of grouping name (a value from ``df['Level']``) to:

        * a filtered ``pandas.DataFrame`` restricted to
          ``output_columns``, when ``return_list=False``;
        * a ``[list_of_keys, list_of_values]`` pair, when
          ``return_list=True``.

        For the ``'Overall'`` grouping, the ``'Group'`` column is
        dropped if present.
    """
    try:
        result = {}
        if len(select_groupings) == 0:
            groupings = df['Level'].drop_duplicates().tolist()
        else:
            groupings = select_groupings

        for header in ['Count', 'Group']:
            if header not in output_columns and header in df:
                output_columns = [header] + output_columns

        for grouping in groupings:
            grouping_ratios = df[df['Level'] == grouping][output_columns]
            if grouping == 'Overall' and 'Group' in grouping_ratios:
                grouping_ratios.drop(columns='Group', inplace=True)
            if return_list:
                result[grouping] = [list(grouping_ratios.keys()), list(grouping_ratios.values())]
            else:
                result[grouping] = grouping_ratios

        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: pdf_breakdown_ratios_by_grouping', ex)

# ---------------------------------------------------------------------------------------------------------------------
def _make_label_func(message):
    def add_label(canvas, message):
        canvas.saveState()
        canvas.setFont("Helvetica", 10)
        canvas.drawString(20, LETTER[1] - 20, message)
        canvas.restoreState()
    return add_label

# ---------------------------------------------------------------------------------------------------------------------

def _add_excel_sensitivity_tag(workbook_obj, writer, sensitivity_tag=None, site_id=None):
    try:
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Add the document properties. Note that these should all be in text format.
        workbook_obj.set_custom_property(f"MSIP_Label_{sensitivity_tag}_Enabled", "true", "text")
        workbook_obj.set_custom_property(f"MSIP_Label_{sensitivity_tag}_SetDate", timestamp, "text"
        )
        workbook_obj.set_custom_property(f"MSIP_Label_{sensitivity_tag}_Method", "Privileged", "text")
        workbook_obj.set_custom_property(f"MSIP_Label_{sensitivity_tag}_Name", sensitivity_tag, "text")
        workbook_obj.set_custom_property(f"MSIP_Label_{sensitivity_tag}_SiteId", site_id, "text")
        # workbook.set_custom_property(f"MSIP_Label_{sensitivity_tag}_ActionId", action_id, "text")
        workbook_obj.set_custom_property(f"MSIP_Label_{sensitivity_tag}_ContentBits", "1", "text")
        workbook_obj.set_custom_property(f"MSIP_Label_{sensitivity_tag}_Tag", "50, 0, 1, 1", "text")

    except:
        pass

    return workbook_obj

# ---------------------------------------------------------------------------------------------------------------------

def _pdf_output(export_pdf_filepath
               , ratio_tables_dfs
               , model_name
               , base_date=''
               , overall_graphs=[]
               , group_graphs=[]
               , group_mosaic_graphs=[]
               , dist_group_mosaic_graphs=[]
               , pdf_grouping_aliases={}
               , gini_lorenz_plots=[]
               , starting_comments=[]
               , ending_comments=[]
               , classification_header = None
               ):
    """
    Build and save the ratio study PDF report using reportlab/platypus.
    Modified version of James Gilbert's function.

    Parameters
    ----------
    export_pdf_filepath : str
        Output filepath for the saved PDF.
    ratio_tables_dfs : dict
        Dictionary output from ``_pdf_breakdown_ratios_by_grouping``,
        mapping grouping name (including ``'Overall'``, if present) to
        its ratios table.
    model_name : str
        Model name shown in the report title.
    base_date : str or int, default ''
        Base date shown as a prefix in the report title (e.g.
        ``'2026'`` or ``'2026 - Sales Ratio Study'``).
    overall_graphs : list, optional, default []
        Overall-level graphs to append. Each entry is either a
        flowable/figure, or a ``[caption, graph]`` pair to be
        centred with a caption above it.
    group_graphs : list, optional, default []
        Same as ``overall_graphs`` but for graphs broken down by group.
    group_mosaic_graphs : list, optional, default []
        Group graphs already composed into mosaics, appended as-is.
    dist_group_mosaic_graphs : list, optional, default []
        Group distribution-plot graphs already composed into mosaics,
        appended as-is.
    pdf_grouping_aliases : dict, optional, default {}
        Mapping of grouping name (as found in ``ratio_tables_dfs``) to
        a display alias used in section headings.
    gini_lorenz_plots : list of str, optional, default []
        Filepaths of Lorenz-curve plot images to embed. Each file is
        deleted from disk after the PDF is built.
    starting_comments : str or list of str, optional, default []
        Comment paragraph(s) inserted near the top of the report,
        before the ratio tables. The special value ``'spacer'``
        inserts a blank spacer instead of text.
    ending_comments : str or list of str, optional, default []
        Comment paragraph(s) inserted near the end of the report,
        after all graphs/plots. The special value ``'spacer'`` inserts
        a blank spacer instead of text.
    classification_header : str, optional, default None
        If provided, drawn as a classification label on the first and
        later pages of the document.

    Returns
    -------
    None
        The PDF is written to ``export_pdf_filepath``; nothing is
        returned. Any files listed in ``gini_lorenz_plots`` are removed
        from disk once the document has been built.
    """
    try:
        _use_agg_if_headless()
        if base_date:
            if isinstance(base_date, int): base_date = str(base_date)
        elements = []
        styles = getSampleStyleSheet()
        centered_style = styles['Normal'].clone('Centered')
        centered_style.alignment = 1
        doc = SimpleDocTemplate(export_pdf_filepath, pagesize=LETTER, topMargin=50, bottomMargin=20, rightMargin=42, leftMargin=42)
        title_text = ((base_date + ' - ') if base_date else '') + "Sales Ratio Study" + ((' - ' + model_name) if model_name != '' else '')
        elements.append(Paragraph(title_text, styles['Title']))
        elements.append(KeepTogether(ConditionalSpacer(0, 0.4 * inch)))
        if starting_comments:
            if isinstance(starting_comments, str):
                starting_comments = [starting_comments]
            for comment in starting_comments:
                if comment == 'spacer':
                    elements.append(KeepTogether(ConditionalSpacer(0, 0.2 * inch)))
                else:
                    elements.append(Paragraph(comment, styles['Normal']))

        if len(ratio_tables_dfs.keys()) > 0:
            if 'Overall' in ratio_tables_dfs:
                overall_ratios = ratio_tables_dfs.pop('Overall')
                if 'Overall' in pdf_grouping_aliases.keys():
                    overall_alias = pdf_grouping_aliases['Overall']
                else:
                    overall_alias = 'Overall'
                elements.append(Paragraph("Time Adjusted Sales Ratios - " + overall_alias, styles['Heading3']))
                elements.append(_pdf_create_table(overall_ratios))
                elements.append(KeepTogether(ConditionalSpacer(0, 0.2 * inch)))
            for key in ratio_tables_dfs:
                group_ratios = ratio_tables_dfs[key]

                if key in pdf_grouping_aliases.keys():
                    group_alias = pdf_grouping_aliases[key]
                else:
                    group_alias = key

                elements.append(Paragraph("Time Adjusted Sales Ratios by " + group_alias, styles['Heading3']))
                elements.append(_pdf_create_table(group_ratios))
                elements.append(KeepTogether(ConditionalSpacer(0, 0.2 * inch)))
        # overall graphs
        if overall_graphs:
            for graph in overall_graphs:
                if isinstance(graph, list):
                    elements.append(KeepTogether([Paragraph(graph[0], centered_style), ConditionalSpacer(0, 0.2 * inch), graph[1]]))
                else:
                    elements.append((ConditionalSpacer(0, 0.5 * inch)))
                    elements.append(graph)

        # group graphs
        if group_graphs:
            for graph in group_graphs:
                try:
                    if isinstance(graph, list):
                        elements.append(KeepTogether([Paragraph(graph[0], centered_style), ConditionalSpacer(0, 0.2 * inch), graph[1]]))
                    else:
                        elements.append((ConditionalSpacer(0, 0.5 * inch)))
                        elements.append(graph)
                except:
                    print(1)

        # mosaic group graphs
        if group_mosaic_graphs:
            for graph in group_mosaic_graphs:
                elements.append((ConditionalSpacer(0, 0.5 * inch)))
                elements.append(graph)

        # mosaic group dist_plot graphs
        if dist_group_mosaic_graphs:
            for graph in dist_group_mosaic_graphs:
                elements.append((ConditionalSpacer(0, 0.5 * inch)))
                elements.append(graph)

        # lorenz plots
        if gini_lorenz_plots:
            for plot in gini_lorenz_plots:
                elements.append((ConditionalSpacer(0, 0.5 * inch)))
                elements.append(Image(plot))

        if ending_comments:
            if isinstance(ending_comments, str):
                ending_comments = [ending_comments]
            elements.append(KeepTogether(ConditionalSpacer(0, 1 * inch)))
            for comment in ending_comments:
                if comment == 'spacer':
                    elements.append(KeepTogether(ConditionalSpacer(0, 0.2 * inch)))
                else:
                    elements.append(Paragraph(comment, styles['Normal']))

        report_elements = []
        for element in elements:
            # Check if the element is an image
            if isinstance(element, plt.Figure):
                # Add the image to the list of flowables
                report_elements.append(Image(element, width=700, height=400))
            else:
                report_elements.append(element)
        if classification_header:
            doc.build(report_elements
                      , onFirstPage=_make_label_func(classification_header)
                      , onLaterPages=_make_label_func(classification_header))
        else:
            doc.build(report_elements)

        for plot in gini_lorenz_plots:
            os.remove(plot)
    except Exception as ex:
        raise RuntimeError('error in ratio study: pdf_output', ex)


# ---------------------------------------------------------------------------------------------------------------------


def compare_ratios(r1, r2, suffixes=('_r1', '_r2'), difference=True, pct_chg=True, merging_cols=['Level', 'Group'], indicators=None):
    """
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
    """
    try:
        r1c = r1.copy() if hasattr(r1, 'copy') else r1
        r2c = r2.copy() if hasattr(r2, 'copy') else r2
        # get r1/r2 from dictionary if dictionary was provided instead of dataframe
        if isinstance(r1c, dict):
            if 'df_all_ratios' in r1c:
                r1c = r1c['df_all_ratios']
            else:
                raise RuntimeError('error in ratio study: ratios_compare - r1 and r2 must be either dataframes or dictionaries containing a dataframe under key "df_all_ratios".')
        if isinstance(r2c, dict):
            if 'df_all_ratios' in r2c:
                r2c = r2c['df_all_ratios']
            else:
                raise RuntimeError('error in ratio study: ratios_compare - r1 and r2 must be either dataframes or dictionaries containing a dataframe under key "df_all_ratios".')

        # ``pd.merge`` is the simplest, and merging is cheap (typically
        # one row per group), so coerce to pandas at the boundary.
        if not isinstance(r1c, pd.DataFrame):
            r1c = nw_utils.require_pandas(
                r1c, reason="compare_ratios() uses pd.merge"
            )
        if not isinstance(r2c, pd.DataFrame):
            r2c = nw_utils.require_pandas(
                r2c, reason="compare_ratios() uses pd.merge"
            )

        if isinstance(suffixes, list) or isinstance(suffixes, set):
            suffixes = tuple(suffixes)

        merging_cols = [x for x in merging_cols if x in r1c and x in r2c]

        if indicators is None:
            indicators = r1c.columns

        indicator_cols = [x for x in indicators if x in r1c.columns and x in r2c.columns and x not in merging_cols]

        calc_cols = []
        for df in (r1c, r2c):
            for col in indicator_cols:
                if not pd.api.types.is_numeric_dtype(df[col]) and 'prb' in col.lower():
                    # A formatted PRB carries a significance marker
                    # ('-0.059*'), which has to go before the value can be
                    # differenced -- but the sign and any percent
                    # formatting are part of the number and must survive.
                    df[col + '_c'] = pd.to_numeric(
                        df[col].apply(_as_float), errors='coerce'
                    )
                else:
                    df[col + '_c'] = df[col]

                if col + '_c' not in calc_cols:
                    calc_cols.append(col + '_c')

        merged_ratios = pd.merge(r1c, r2c, on=merging_cols, suffixes=suffixes)

        resulting_cols = merging_cols.copy()
        for col in calc_cols:
            resulting_cols += [col[:-2] + suffixes[0], col[:-2] + suffixes[1]]
            try:
                if difference:
                    col_diff = f"{col[:-2]}_diff"
                    if merged_ratios[col + suffixes[0]].dtype not in ['object', 'str'] and merged_ratios[col + suffixes[1]].dtype not in ['object', 'str']:
                        merged_ratios[col_diff] = merged_ratios[col + suffixes[1]] - merged_ratios[col + suffixes[0]]
                        merged_ratios[col_diff] = merged_ratios[col_diff].apply(lambda x: '-' if pd.isna(x) or x == 0 else f"{round(x, 4)}{'↑' if x > 0 else '↓'}")
                        resulting_cols.append(col_diff)
                if pct_chg:
                    col_pct = f"{col[:-2]}_%"
                    if merged_ratios[col + suffixes[0]].dtype not in ['object', 'str'] and merged_ratios[col + suffixes[1]].dtype not in ['object', 'str']:
                        merged_ratios[col_pct] = np.where((merged_ratios[col + suffixes[0]] != 0) & (~np.isnan(merged_ratios[col + suffixes[1]])), (merged_ratios[col + suffixes[1]] / merged_ratios[col + suffixes[0]] - 1) * 100, np.nan)
                        merged_ratios[col_pct] = merged_ratios[col_pct].apply(lambda x: '-' if pd.isna(x) or x == 0 else f"{round(x, 4)}{'%↑' if x > 0 else '%↓'}")
                        resulting_cols.append(col_pct)
            except Exception as ex:
                raise RuntimeError(
                    f"error in ratio study: ratios_compare\n{traceback.format_exc()}"
                ) from ex

        return merged_ratios[resulting_cols]
    except Exception as ex:
        raise RuntimeError(
            f"error in ratio study: ratios_compare\n{traceback.format_exc()}"
        ) from ex
    
# ---------------------------------------------------------------------------------------------------------------------

# --- indicator value parsing -----------------------------------------------

# Sample sizes below this cannot support any indicator at all, as opposed to
# merely making them unreliable (``min_count_threshold``).
_UNUSABLE_SAMPLE = 5

# Columns that are never checked: identifiers, the count itself, and the
# derived confidence-interval / p-value columns.
_NEVER_CHECKED = ('Level', 'Group_by', 'Count', 'Weighted_Mean')

# IAAO 11.2.1 (Uniformity among Strata): a stratum's level of assessment
# should fall within this fraction of the overall level of assessment --
# 5 percent, not 5 percentage points.
_STRATA_MEDIAN_UNIFORMITY_PCT = 0.05

# IAAO 11.2.2 (Uniformity between Classes): no two strata's levels of
# assessment should differ from each other by more than this fraction --
# 10 percent, not 10 percentage points.
_INTERCLASS_MEDIAN_UNIFORMITY_PCT = 0.10

# Leading signed number of a cell, so a marker or a percent sign can follow.
_LEADING_NUMBER = re.compile(r'-?\s*\d*\.?\d+')


def _split_marker(value):
    """
    Split a possibly marked indicator value into its number and its marker.

    Indicators can arrive carrying a trailing marker, e.g. ``'-0.059*'``
    for a PRB that is not significant at the required level, and can be
    formatted as a percentage. The percent sign is read as part of the
    number (``'8.34%'`` -> 0.0834), not as a marker, and the sign of the
    value is kept.

    Parameters
    ----------
    value : float, int, str, or None
        Raw cell value.

    Returns
    -------
    tuple
        * number : float or None -- the numeric value, or None when the
          cell holds no usable number (including NaN).
        * marked : bool -- True when the cell was a string carrying
          anything beyond the number itself.
    """
    if isinstance(value, str):
        text = value.strip()
        match = _LEADING_NUMBER.search(text)
        if match is None:
            return None, True
        number = float(match.group(0).replace(' ', ''))
        remainder = text.replace(match.group(0), '', 1)
        if remainder.startswith('%'):
            number /= 100
            remainder = remainder[1:]
        return number, bool(remainder)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, False
    return (None, False) if math.isnan(number) else (number, False)


def _as_float(value):
    """
    Numeric value of a cell, or None when it holds no usable number.

    Parameters
    ----------
    value : float, int, str, or None
        Raw cell value.

    Returns
    -------
    float or None
    """
    return _split_marker(value)[0]


def _standard_for(col, assessment_standards):
    """
    The raw assessment standard for a column, if there is one.

    ``'CoD_rsd'`` and ``'CoDw'`` are variants of the CoD and share its
    standard.

    Parameters
    ----------
    col : str
        Column name.
    assessment_standards : dict
        Standards keyed by column name.

    Returns
    -------
    tuple, list, or None
        The standard as given, or None when the column has none.
    """
    return assessment_standards.get('CoD' if col in ('CoD_rsd', 'CoDw') else col)


def _standard_band(standard,
                   col,
                   is_group,
                   overall_median=None,
                   compare_groups_median_to_overall_median=True):
    """
    Resolve a standard into the ``(low, high)`` band a value is checked against.

    Statistics drawn from a stratum get more latitude than the population
    ones, per IAAO guidance: the CoD's upper bound is 5 points higher, and
    a stratum's median ASR (its level of assessment) is compared with the
    overall level of assessment +/- 5% rather than with the fixed band.

    Parameters
    ----------
    standard : tuple, list, or None
        Standard as held in ``assessment_standards``. Anything that is not
        a two-element numeric pair (e.g. the ``'Mkt_desc'`` entry) yields
        None.
    col : str
        Column the standard applies to.
    is_group : bool
        True for a stratum row, False for the overall row.
    overall_median : float, optional
        Overall median ASR, used for the stratum-median comparison.
    compare_groups_median_to_overall_median : bool, default True
        Whether to compare a stratum's median with `overall_median`
        instead of the fixed band.

    Returns
    -------
    tuple or None
        ``(low, high)``, or None when the standard is not usable.
    """
    if not isinstance(standard, (tuple, list)) or len(standard) != 2:
        return None
    low, high = _as_float(standard[0]), _as_float(standard[1])
    if low is None or high is None:
        return None
    if not is_group:
        return (low, high)
    if col in ('CoD', 'CoD_rsd', 'CoDw'):
        return (low, round(high + 0.05, 2))
    if col == 'Median':
        if compare_groups_median_to_overall_median and overall_median is not None:
            return (round(overall_median - 0.05, 4), round(overall_median + 0.05, 4))
        return (low, round(high + 0.05, 2))
    return (low, high)


def _strata_median_pct_change(group_median, group_ci, overall_median, overall_ci):
    """
    Percent change between a stratum's Median and the overall's, per IAAO
    11.2.1 (Uniformity among Strata).

    The two confidence intervals are compared at the bounds nearest each
    other -- the stratum's upper bound against the overall's lower bound
    when the stratum runs low, or the stratum's lower bound against the
    overall's upper bound when it runs high -- and the gap between them
    is expressed as a percent of the overall's bound, not a
    percentage-point difference. This reproduces the worked example in
    the standard: a neighborhood median of 0.85 (CI 0.81-0.94) against a
    market-area median of 1.01 (CI 0.98-1.05) is a 4.08% change --
    (0.98 - 0.94) / 0.98 -- not 13%.

    Parameters
    ----------
    group_median : float
        Stratum's Median point estimate.
    group_ci : tuple
        Stratum's ``(low, high)`` Median confidence interval.
    overall_median : float
        Overall Median point estimate.
    overall_ci : tuple
        Overall's ``(low, high)`` Median confidence interval.

    Returns
    -------
    float or None
        Percent change as a fraction (``0.0408`` for a 4.08% gap). Zero
        or negative means the intervals already overlap, which is
        trivially uniform. None when the reference bound is zero and the
        percent is undefined.
    """
    if group_median <= overall_median:
        reference, compared = overall_ci[0], group_ci[1]
        gap = reference - compared
    else:
        reference, compared = overall_ci[1], group_ci[0]
        gap = compared - reference
    return gap / reference if reference else None


def _interclass_median_pct_change(median_a, ci_a, median_b, ci_b):
    """
    Percent change between two peer strata's Medians, per IAAO 11.2.2
    (Uniformity between Classes).

    The same nearest-bounds-gap-over-reference measure as
    :func:`_strata_median_pct_change`, but between two strata rather than a
    stratum and the overall -- there is no "overall" role between peers, so
    the lower-median stratum's own bound stands in as the reference, which
    keeps the result the same regardless of which stratum is passed first.

    Parameters
    ----------
    median_a, median_b : float
        The two strata's Median point estimates.
    ci_a, ci_b : tuple
        The two strata's ``(low, high)`` Median confidence intervals.

    Returns
    -------
    float or None
        Percent change as a fraction, or None when the reference bound is
        zero. See :func:`_strata_median_pct_change`.
    """
    if median_a <= median_b:
        return _strata_median_pct_change(median_b, ci_b, median_a, ci_a)
    return _strata_median_pct_change(median_a, ci_a, median_b, ci_b)


def _vertical_equity_direction(col, value):
    """
    Read a PRB or PRD value as progressivity or regressivity.

    Parameters
    ----------
    col : str
        ``'PRB'`` or ``'PRD'``.
    value : float
        Indicator value.

    Returns
    -------
    str
        'progressive' or 'regressive'.
    """
    if col == 'PRB':
        return 'progressive' if value > 0 else 'regressive'
    return 'progressive' if value < 1 else 'regressive'      # PRD


def _small_sample_comment(count, min_count_threshold):
    """
    Comment for a row whose sample is too small to conclude from, if any.

    Parameters
    ----------
    count : float, int, str, or None
        Row's sample count.
    min_count_threshold : int
        Minimum count for indicators to be considered reliable.

    Returns
    -------
    str or None
        The comment, or None when the sample is large enough.
    """
    number = _as_float(count)
    if number is None:
        return None
    if number < _UNUSABLE_SAMPLE:
        return (f'Sample too small ({count}): some ratio indicators could not '
                'be calculated, while the remaining may not reliably indicate '
                'adherence to standards.')
    if number < min_count_threshold:
        return (f'Small sample ({count}): indicators may not reliably indicate '
                'adherence to standards.')
    return None


def _acceptance_binaries(df,
                         assessment_standards,
                         overall_median=None,
                         compare_groups_median_to_overall_median=True):
    """
    One True/False column per assessment standard.

    A flag reports whether the indicator's point estimate sits inside the
    band that applies to its row -- it does not repeat the comment
    logic's extra conditions (an overlapping confidence interval, or a
    sample too small to conclude from), so a value can be flagged
    not-acceptable without carrying a comment. Rows where the indicator or
    its standard is missing are left as NA rather than False.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame holding the indicator columns and a ``'Level'`` column.
    assessment_standards : dict
        Standards keyed by column name.
    overall_median : float, optional
        Overall median ASR, for the stratum-median comparison.
    compare_groups_median_to_overall_median : bool, default True
        Passed through to :func:`_standard_band`.

    Returns
    -------
    dict
        ``{'<indicator>_acceptable': Series}`` for every standard that
        names a column present in ``df``, in the order the standards were
        given.
    """
    flags = {}
    for key, standard in assessment_standards.items():
        if key not in df.columns:
            continue
        column = pd.Series(pd.NA, index=df.index, dtype='boolean')
        for idx in df.index:
            band = _standard_band(
                standard, key, df.at[idx, 'Level'] != 'Overall',
                overall_median=overall_median,
                compare_groups_median_to_overall_median=compare_groups_median_to_overall_median,
            )
            value = _as_float(df.at[idx, key])
            if band is not None and value is not None:
                column[idx] = _check_acceptance(value, band) == 'statistic within standards'
        flags[f'{key}_acceptable'] = column
    return flags

# ---------------------------------------------------------------------------------------------------------------------

def _add_comments(df,
                  req_significance,
                  assessment_standards,
                  results_cols,
                  results_format,
                  skip,
                  compare_groups_median_to_overall_median=True,
                  std_non_acceptable_symbol=chr(8869),
                  small_sample_symbol=chr(248),
                  min_count_threshold=15,
                  ci_conf_level=None,
                  comments_header='Comments Ratio Study Standard metrics',
                  VE_bundle_comments=False,
                  add_binaries=True):
    """
    Add descriptive comments and acceptance markers to indicators that are
    not statistically significant or fall outside assessment standards.

    Works in three passes: the checks run on the unformatted values, then
    ``custom_format`` formats them, then the marker symbols are appended
    to the formatted strings.

    Parameters
    ----------
    df : pandas.DataFrame (or supported frame-like)
        Input dataframe with one row per Level/Group and indicator
        columns to check.
    req_significance : float
        Required significance level (p-value threshold). Example: 0.05
        for 5%.
    assessment_standards : dict
        Assessment standards to check indicators against. Keys must
        match column names (``'CoD'`` is used for both ``'CoD_rsd'``
        and ``'CoDw'``); values are ``(min, max)`` tuples describing
        the acceptable range. Entries that are not a numeric pair (e.g.
        ``'Mkt_desc'``) are ignored.
    results_cols : list of str
        Column names to be iterated over and checked.
    results_format : dict
        Formatting instructions passed through to ``custom_format``
        for each column in ``results_cols``.
    skip : list of str
        Columns in ``results_cols`` to be skipped from the
        significance/standards checks.
    compare_groups_median_to_overall_median : bool, default True
        Gates the horizontal-equity checks, both only meaningful when a
        `group_by` produced a ``'Group'`` column:

        * Stratum vs. overall, IAAO 11.2.1 (uniformity among strata) -- a
          group's ``'Median'`` is checked against the overall ``'Median'``
          instead of the fixed ``assessment_standards`` range. When both
          ``'Median_ci'`` values are available, the check compares the two
          confidence intervals and flags a stratum only when the gap
          between their nearest bounds exceeds 5 percent of the overall's
          bound (see :func:`_strata_median_pct_change`); otherwise it falls
          back to a fixed +/- 5 percentage-point band around the overall
          ``'Median'`` point estimate.
        * Stratum vs. stratum, IAAO 11.2.2 (uniformity between classes) --
          every pair of strata sharing a ``'Level'`` (grouping dimension)
          is compared the same way, and flagged when the gap between their
          nearest ``'Median_ci'`` bounds exceeds 10 percent of the lower
          stratum's own bound (see :func:`_interclass_median_pct_change`).
          Unlike the first check, there is no fixed-band fallback here: a
          pair without both confidence intervals is not compared.
    std_non_acceptable_symbol : str, default chr(8869)
        Marker symbol appended to a formatted value when its indicator
        fails a significance or standards check.
    small_sample_symbol : str, default chr(248)
        Marker symbol appended to ``'Count'`` when the sample size is
        below ``min_count_threshold``.
    ci_conf_level : float, optional
        Confidence level of the ``<indicator>_ci`` columns, used to label
        the comments that are decided by an interval rather than by a
        p-value. Distinct from `req_significance`: that one is the
        significance threshold for the tests, this one is the width of the
        intervals the acceptance checks read. Left unset, those comments
        say "confidence interval" without naming a level.
    min_count_threshold : int, default 15
        Minimum sample count for indicators to be considered reliable;
        rows below this are flagged with ``small_sample_symbol`` and a
        small-sample comment, and their indicators are not checked.
    comments_header : str, default 'Comments Ratio Study Standard metrics'
        Name of the column used to store standard ratio-study comments.
    VE_bundle_comments : bool, default False
        If True, also accumulate a consolidated
        ``'Vertical_Equity_Analysis'`` column summarizing the PRB, PRD
        and VEI conclusions.
    add_binaries : bool, default True
        If True, add one boolean ``'<indicator>_acceptable'`` column per
        assessment standard naming a column in ``df`` (e.g. ``'CoD'`` ->
        ``'CoD_acceptable'``), True where the value sits inside the band
        that applies to its row. These read the point estimate only, so
        unlike the comments they ignore an overlapping confidence
        interval and are still filled in for small samples; NA where the
        indicator or its standard is missing. See
        :func:`_acceptance_binaries`.

    Returns
    -------
    pandas.DataFrame
        Input dataframe with ``comments_header`` (and
        ``'Vertical_Equity_Analysis'`` when ``VE_bundle_comments=True``,
        and the ``'<indicator>_acceptable'`` columns when
        ``add_binaries=True``), indicator values formatted via
        ``custom_format``, and non-acceptable/insignificant values
        annotated with the appropriate marker symbol.

    Notes
    -----
    Preferred ASCII/unicode marker characters: 9437, 8869, 0x2099, 248.
    """
    try:
        # Heavy use of per-cell access and string operations below --
        # convert to pandas at the boundary.
        if not isinstance(df, pd.DataFrame):
            df = nw_utils.require_pandas(
                df, reason="_add_comments() uses per-cell access"
            )
        df = df.copy()

        cols_to_check = [col for col in results_cols
                         if col not in skip
                         and col not in _NEVER_CHECKED
                         and '_ci' not in col
                         and 'Pvalue' not in col]

        # Two different levels reach this function and they are not
        # interchangeable: `req_significance` is the p-value threshold the
        # significance tests use, while `ci_conf_level` is the width of the
        # intervals the acceptance checks read. Labelling a CI-based
        # finding with the significance level (or leaving it unlabelled)
        # is what made the comments unverifiable against the table.
        confidence_pct = round((1 - req_significance) * 100)
        ci_pct = None
        if ci_conf_level is not None:
            from ratio_study.confint import confidence_level_as_fraction
            ci_pct = round(confidence_level_as_fraction(ci_conf_level, 'ci_conf_level') * 100)
        interval_label = f'{ci_pct}% confidence interval' if ci_pct else 'confidence interval'
        overall_median = None
        if 'Median' in df.columns:
            overall_medians = df.loc[df['Level'] == 'Overall', 'Median']
            if len(overall_medians):
                overall_median = _as_float(overall_medians.iloc[0])
        overall_median_ci = None
        if 'Median_ci' in df.columns:
            overall_median_cis = df.loc[df['Level'] == 'Overall', 'Median_ci']
            if len(overall_median_cis) and isinstance(overall_median_cis.iloc[0], tuple):
                overall_median_ci = overall_median_cis.iloc[0]

        comments = {}           # row -> list of comment fragments
        ve_notes = {}           # row -> list of vertical-equity fragments
        flagged = {}            # row -> columns to mark with the symbol
        small_samples = []      # rows whose Count gets the small-sample mark

        for idx in df.index:
            notes, bundle, flags = [], [], []
            is_group = df.at[idx, 'Level'] != 'Overall'

            # Nothing at this sample size supports a conclusion, so the
            # row gets the small-sample comment and no indicator checks.
            small_sample = _small_sample_comment(
                df.at[idx, 'Count'] if 'Count' in df.columns else None,
                min_count_threshold,
            )
            if small_sample:
                comments[idx] = [small_sample]
                small_samples.append(idx)
                continue

            for col in cols_to_check:
                value = df.at[idx, col]

                # PRB arrives as a string when it is not significant. The
                # marker is stripped so the value still formats as a
                # number; the symbol is re-applied in the marker pass.
                if col.lower() == 'prb' and isinstance(value, str):
                    number, marked = _split_marker(value)
                    if marked:
                        notes.append(f'{col} not significant at {confidence_pct}%')
                        flags.append(col)
                    if number is not None:
                        # pandas >= 3 infers a string column as its own 'str'
                        # dtype, which rejects a float; object holds both and
                        # is what older pandas already used here.
                        if df[col].dtype != object:
                            df[col] = df[col].astype(object)
                        df.at[idx, col] = number
                    continue

                # An explicit p-value settles the column on its own.
                if f'{col}_Pvalue' in df.columns:
                    pvalue = _as_float(df.at[idx, f'{col}_Pvalue'])
                    if pvalue is not None and pvalue > req_significance:
                        notes.append(f'{col} not significant at {confidence_pct}%: '
                                     f'P-value {round(pvalue, 4)}')
                        flags.append(col)
                    continue

                if col == 'VEI_conclusion':
                    # `vei_analysis` reports 'Acceptable', 'Regressivity' or
                    # 'Progressivity'; only the last two are findings. This
                    # used to test `value != 'Acceptable'`, a word the VEI
                    # never returns, so a acceptable VEI was reported as
                    # "not acceptable: Acceptable" on every row. A sample too
                    # small for a verdict is covered by the small-sample
                    # comment and is not an inequity finding either.
                    if value in VEI_INEQUITY_CONCLUSIONS:
                        notes.append(f'Vertical Equity Indicator does not pass: {value}.')
                        flags.append('VEI')
                        if VE_bundle_comments:
                            bundle.append(f'VEI: {value}')
                    continue

                if (col == 'Median' and is_group
                        and compare_groups_median_to_overall_median
                        and overall_median is not None):
                    group_median = _as_float(value)
                    group_ci = df.at[idx, 'Median_ci'] if 'Median_ci' in df.columns else None
                    pct_change = None
                    if (group_median is not None
                            and isinstance(group_ci, tuple)
                            and isinstance(overall_median_ci, tuple)):
                        pct_change = _strata_median_pct_change(
                            group_median, group_ci, overall_median, overall_median_ci)
                    if pct_change is not None:
                        if pct_change > _STRATA_MEDIAN_UNIFORMITY_PCT:
                            notes.append(
                                'Median suggests a review under IAAO 11.2.1 '
                                '(uniformity among strata): '
                                f'{round(pct_change * 100, 2)}% different from '
                                'the overall level of assessment, more than the '
                                f'{round(_STRATA_MEDIAN_UNIFORMITY_PCT * 100)}% '
                                f'limit (compared at the nearest bounds of the '
                                f'two {interval_label}s)'
                            )
                            flags.append(col)
                        continue
                    # No usable confidence interval on one side or the
                    # other: fall back to the fixed-band comparison below.

                band = _standard_band(
                    _standard_for(col, assessment_standards), col,
                    is_group,
                    overall_median=overall_median,
                    compare_groups_median_to_overall_median=compare_groups_median_to_overall_median,
                )
                confidence_interval = df.at[idx, f'{col}_ci'] if f'{col}_ci' in df.columns else None

                # Without a standard, the only check available is whether
                # the confidence interval can rule out zero.
                if band is None:
                    if (isinstance(confidence_interval, tuple)
                            and confidence_interval[0] <= 0 <= confidence_interval[1]):
                        notes.append(
                            f'{col} could be zero: its {interval_label} '
                            'includes 0')
                        flags.append(col)
                    continue

                number = _as_float(value)
                if number is None:
                    continue
                if _check_acceptance(number, band) != 'statistic outside standards':
                    continue
                # A confidence interval that still reaches the standard
                # leaves the breach unconfirmed. Say so rather than saying
                # nothing: the value in the table is outside the standard
                # either way, and a silent omission reads as a pass.
                if (isinstance(confidence_interval, tuple)
                        and _compare_ci(confidence_interval, band)
                        != 'confidence interval outside standards'):
                    notes.append(
                        f'{col} is outside assessment standards {band}, but its '
                        f'{interval_label} still reaches them, so the breach is '
                        'not confirmed')
                    continue

                if isinstance(confidence_interval, tuple):
                    notes.append(
                        f'{col} suggests a review regarding assessment '
                        f'standards {band}: its {interval_label} is entirely '
                        'outside them')
                else:
                    notes.append(f'{col} suggests a review regarding assessment '
                                 f'standards {band}')
                flags.append(col)
                if VE_bundle_comments and col in ('PRB', 'PRD'):
                    bundle.append(f'{col}: {_vertical_equity_direction(col, number)}.')

            comments[idx] = notes
            ve_notes[idx] = bundle
            flagged[idx] = flags

        # IAAO 11.2.2 (Uniformity between Classes) -- only meaningful with a
        # `group_by`, which is exactly when the frame carries a 'Group'
        # column. A peer-to-peer check, unlike the stratum-vs-overall one
        # above, so it runs once every row's own comments are settled, and
        # only compares strata that share a 'Level' (grouping dimension) --
        # strata from different groupings are not peers.
        if ('Group' in df.columns and 'Median' in df.columns
                and compare_groups_median_to_overall_median):
            for level_value in df.loc[df['Level'] != 'Overall', 'Level'].unique():
                peers = [idx for idx in df.index
                        if df.at[idx, 'Level'] == level_value and idx not in small_samples]
                for i, idx_a in enumerate(peers):
                    median_a = _as_float(df.at[idx_a, 'Median'])
                    ci_a = df.at[idx_a, 'Median_ci'] if 'Median_ci' in df.columns else None
                    if median_a is None or not isinstance(ci_a, tuple):
                        continue
                    for idx_b in peers[i + 1:]:
                        median_b = _as_float(df.at[idx_b, 'Median'])
                        ci_b = df.at[idx_b, 'Median_ci'] if 'Median_ci' in df.columns else None
                        if median_b is None or not isinstance(ci_b, tuple):
                            continue
                        pct_change = _interclass_median_pct_change(median_a, ci_a, median_b, ci_b)
                        if pct_change is None or pct_change <= _INTERCLASS_MEDIAN_UNIFORMITY_PCT:
                            continue
                        pct_text = round(pct_change * 100, 2)
                        limit_text = round(_INTERCLASS_MEDIAN_UNIFORMITY_PCT * 100)
                        for this_idx, other_idx in ((idx_a, idx_b), (idx_b, idx_a)):
                            comments[this_idx].append(
                                'Median (level of assessment) does not pass IAAO 11.2.2 '
                                f'(uniformity between classes): {pct_text}% different '
                                f"from Group {df.at[other_idx, 'Group']!r}, more than "
                                f'the {limit_text}% limit (compared at the nearest '
                                f'bounds of the two {interval_label}s) -- suggests a '
                                'review regarding assessment standards'
                            )
                            flagged[this_idx].append('Median')

        df[comments_header] = pd.Series(
            {idx: ' | '.join(notes) for idx, notes in comments.items()}
        ).reindex(df.index, fill_value='').str.replace('_', ' ')
        if VE_bundle_comments:
            df['Vertical_Equity_Analysis'] = pd.Series(
                {idx: ''.join(f' | {note}' for note in notes)
                 for idx, notes in ve_notes.items()}
            ).reindex(df.index, fill_value='')
        if add_binaries:
            for name, column in _acceptance_binaries(
                df, assessment_standards, overall_median=overall_median,
                compare_groups_median_to_overall_median=compare_groups_median_to_overall_median,
            ).items():
                df[name] = column

        df = custom_format(df, results_cols, results_format,
                           skip=list(skip) + [comments_header], all_str=True)

        # Mark the formatted values: one symbol for a failed
        # significance/standards check, another for a small sample.
        for idx, flags in flagged.items():
            for col in dict.fromkeys(flags):        # de-duplicated, in order
                if col in df.columns:
                    df.at[idx, col] = f'{df.at[idx, col]}{std_non_acceptable_symbol}'
        for idx in small_samples:
            if 'Count' in df.columns:
                df.at[idx, 'Count'] = f'{df.at[idx, "Count"]}{small_sample_symbol}'

        return df

    except Exception as ex:
        raise RuntimeError(
            f"error in ratio study: _add_comments\n{traceback.format_exc()}"
        ) from ex
