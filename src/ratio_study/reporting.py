import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from reportlab.platypus import *
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas as _rl_canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from typing import Optional, Union

import io
import os
import re
import copy
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape
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
# PDF branding: logo, watermark, page numbers, cover and index
# ---------------------------------------------------------------------------------------------------------------------

_LOGO_POSITIONS = ('top_left', 'top_center', 'top_right')
_NUMBER_POSITIONS = ('bottom_left', 'bottom_center', 'bottom_right')
_WATERMARK_POSITIONS = ('fill', 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right')
_PAGE_MARGIN_X = 42


def _check_image(path, what):
    """Return ``path`` as a string after checking that it is a readable image file."""
    if path is None or str(path) == '':
        raise ValueError(f'{what}: an image file path is required')
    if not os.path.isfile(str(path)):
        raise FileNotFoundError(f'{what}: image file not found: {path}')
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(str(path)) as img:
            img.verify()
    except Exception as ex:
        raise ValueError(f'{what}: {path} is not a readable image file ({ex})') from ex
    return str(path)


def _choice(value, allowed, default, what, aliases=None):
    """Validate ``value`` against ``allowed`` (case-insensitive), with ``default`` when None."""
    if value is None:
        return default
    key = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    if aliases and key in aliases:
        key = aliases[key]
    if key not in allowed:
        raise ValueError(f'{what}: {value!r} is not valid; choose one of {list(allowed)}')
    return key


def _as_options(value, what, shorthand_key=None):
    """Expand a shorthand (``True``, a path) into an options dict; ``None`` when disabled."""
    if value is None or value is False:
        return None
    if isinstance(value, dict):
        return dict(value)
    if value is True:
        return {}
    if shorthand_key and isinstance(value, (str, Path)):
        return {shorthand_key: value}
    raise ValueError(f'{what}: expected a dict, True/False or a file path, got {type(value).__name__}')


def _normalize_pdf_branding(logo=None, watermark=None, page_numbers=None, cover=None, index=None):
    """
    Validate and normalise the optional PDF branding arguments.

    Every argument accepts ``None``/``False`` (off), and the shorthands
    described in :func:`ratio_study.ratio_study.ratios_report`. Image paths
    are checked here, so a typo fails before the study runs rather than after.

    Returns
    -------
    dict
        Keys ``logo``, ``watermark``, ``page_numbers``, ``cover``, ``index``;
        each ``None`` (off) or a fully populated options dict.
    """
    out = {'logo': None, 'watermark': None, 'page_numbers': None, 'cover': None, 'index': None}

    opts = _as_options(logo, 'pdf_logo', 'path')
    if opts is not None:
        out['logo'] = {
            'path': _check_image(opts.get('path'), 'pdf_logo'),
            'pages': _choice(opts.get('pages'), ('first', 'all'), 'first', 'pdf_logo pages',
                             {'first_only': 'first', 'first_page': 'first', 'all_pages': 'all'}),
            'position': _choice(opts.get('position'), _LOGO_POSITIONS, 'top_left', 'pdf_logo position',
                                {'left': 'top_left', 'center': 'top_center', 'right': 'top_right'}),
            'max_height': float(opts.get('max_height', 80)),
            'max_width': float(opts.get('max_width', 300)),
        }

    opts = _as_options(watermark, 'pdf_watermark', 'path')
    if opts is not None:
        opacity = float(opts.get('opacity', 0.15))
        if not 0 <= opacity <= 1:
            raise ValueError(f'pdf_watermark opacity: {opacity} is not between 0 (invisible) and 1 (opaque)')
        scale = float(opts.get('scale', 0.5))
        if not 0 < scale <= 1:
            raise ValueError(f'pdf_watermark scale: {scale} is not in (0, 1]')
        out['watermark'] = {
            'path': _check_image(opts.get('path'), 'pdf_watermark'),
            'position': _choice(opts.get('position'), _WATERMARK_POSITIONS, 'fill', 'pdf_watermark position'),
            'opacity': opacity,
            'scale': scale,
        }

    opts = _as_options(page_numbers, 'pdf_page_numbers')
    if opts is not None:
        font_size = float(opts.get('font_size', 9))
        if font_size <= 0:
            raise ValueError(f'pdf_page_numbers font_size: {font_size} must be positive')
        out['page_numbers'] = {
            'position': _choice(opts.get('position'), _NUMBER_POSITIONS, 'bottom_center', 'pdf_page_numbers position',
                                {'left': 'bottom_left', 'center': 'bottom_center', 'right': 'bottom_right'}),
            'font_size': font_size,
            'format': _choice(opts.get('format'), ('number', 'page_of_n'), 'number', 'pdf_page_numbers format',
                              {'n': 'number', 'page_n': 'number', 'page_x_of_n': 'page_of_n', 'x_of_n': 'page_of_n'}),
        }

    opts = _as_options(cover, 'pdf_cover', 'title')
    if opts is not None:
        cover_logo = _as_options(opts.get('logo'), 'pdf_cover logo', 'path')
        if cover_logo is not None:
            cover_logo = {'path': _check_image(cover_logo.get('path'), 'pdf_cover logo'),
                          'max_height': float(cover_logo.get('max_height', 300)),
                          'max_width': float(cover_logo.get('max_width', 468))}
        authors = opts.get('authors')
        if isinstance(authors, str):
            authors = [authors]
        out['cover'] = {'title': opts.get('title'), 'subtitle': opts.get('subtitle'),
                        'authors': list(authors) if authors else [],
                        'date': opts.get('date'), 'logo': cover_logo}

    opts = _as_options(index, 'pdf_index', 'title')
    if opts is not None:
        if out['cover'] is None:
            import warnings
            warnings.warn('pdf_index needs pdf_cover; the index page was not added', stacklevel=3)
        else:
            out['index'] = {'title': opts.get('title')}
    return out


def _fit(width, height, max_width, max_height):
    """Scale ``(width, height)`` down to fit the box, keeping the aspect ratio."""
    ratio = min(max_width / width, max_height / height)
    return width * ratio, height * ratio


def _image_size(reader):
    width, height = reader.getSize()
    return float(width), float(height)


def _watermark_reader(path, opacity):
    """Image reader of the watermark with its alpha channel scaled by ``opacity``."""
    from PIL import Image as _PILImage
    with _PILImage.open(path) as img:
        img = img.convert('RGBA')
    alpha = img.getchannel('A').point(lambda a: int(round(a * opacity)))
    img.putalpha(alpha)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return ImageReader(buffer)


def _draw_watermark(canv, reader, cfg):
    page_w, page_h = LETTER
    img_w, img_h = _image_size(reader)
    margin = 0.5 * inch
    if cfg['position'] == 'fill':
        w, h = _fit(img_w, img_h, page_w, page_h)
        x, y = (page_w - w) / 2, (page_h - h) / 2
    else:
        w, h = _fit(img_w, img_h, page_w * cfg['scale'], page_h * cfg['scale'])
        pos = cfg['position']
        x = margin if 'left' in pos else (page_w - w - margin if 'right' in pos else (page_w - w) / 2)
        y = margin if 'bottom' in pos else (page_h - h - margin if 'top' in pos else (page_h - h) / 2)
    canv.drawImage(reader, x, y, w, h, mask='auto')


def _logo_box(cfg, reader, has_label):
    """Return ``(x, y, w, h)`` of the logo on the page."""
    page_w, page_h = LETTER
    img_w, img_h = _image_size(reader)
    w, h = _fit(img_w, img_h, cfg['max_width'], cfg['max_height'])
    top = page_h - (30 if has_label else 12)
    pos = cfg['position']
    x = _PAGE_MARGIN_X if pos == 'top_left' else (page_w - _PAGE_MARGIN_X - w if pos == 'top_right' else (page_w - w) / 2)
    return x, top - h, w, h


def _draw_page_number(canv, cfg, number, total):
    text = f'Page {number} of {total}' if cfg['format'] == 'page_of_n' else str(number)
    canv.saveState()
    canv.setFont('Helvetica', cfg['font_size'])
    y = 16
    pos = cfg['position']
    if pos == 'bottom_left':
        canv.drawString(_PAGE_MARGIN_X, y, text)
    elif pos == 'bottom_right':
        canv.drawRightString(LETTER[0] - _PAGE_MARGIN_X, y, text)
    else:
        canv.drawCentredString(LETTER[0] / 2, y, text)
    canv.restoreState()


def _numbered_canvas_class(cfg):
    """
    Canvas that stamps page numbers once the page total is known.

    Pages flagged ``_rs_numbered`` by the page handlers are counted; the cover
    and the index page are not flagged, so they get no number and do not count.
    """
    class _NumberedCanvas(_rl_canvas.Canvas):
        def __init__(self, *args, **kwargs):
            self._saved_states = []
            super().__init__(*args, **kwargs)

        def _startPage(self):
            super()._startPage()
            self._rs_numbered = False

        def showPage(self):
            self._saved_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            states = self._saved_states
            total = sum(1 for st in states if st.get('_rs_numbered'))
            number = 0
            for state in states:
                self.__dict__.update(state)
                if state.get('_rs_numbered'):
                    number += 1
                    _draw_page_number(self, cfg, number, total)
                super().showPage()
            super().save()
    return _NumberedCanvas


class _TocMarker(Flowable):
    """Zero-size flowable recording the page a section starts on."""

    def __init__(self, level, text):
        super().__init__()
        self.level, self.text = level, text
        self.width = self.height = 0

    def wrap(self, availWidth, availHeight):
        return 0, 0

    def draw(self):
        pass


def _find_markers(flowable):
    if isinstance(flowable, _TocMarker):
        return [flowable]
    found = []
    for child in getattr(flowable, '_content', None) or []:
        found.extend(_find_markers(child))
    return found


class _ReportDoc(BaseDocTemplate):
    """Document that feeds the index and tracks front-matter pages."""

    front_pages = 0

    def beforeDocument(self):
        self.front_pages = 0

    def afterFlowable(self, flowable):
        for marker in _find_markers(flowable):
            self.notify('TOCEntry', (marker.level, marker.text, self.page - self.front_pages))


def _shrink_images(flowable, limit, reserve=0):
    """Scale images taller than ``limit`` down to fit, in place, looking inside KeepTogether blocks."""
    if isinstance(flowable, Image):
        if flowable.drawHeight > limit - reserve:
            ratio = (limit - reserve) / flowable.drawHeight
            flowable.drawWidth *= ratio
            flowable.drawHeight *= ratio
        return flowable
    content = getattr(flowable, '_content', None)
    if content:
        # a caption and spacer share the page with the image
        extra = 60 if any(isinstance(c, Paragraph) for c in content) else 0
        for child in content:
            _shrink_images(child, limit, reserve + extra)
    return flowable


def _pdf_clean_text(text):
    return _xml_escape(re.sub(r'<[^>]+>', '', str(text)))


def _cover_date_text(value):
    from datetime import date
    if value is None or value == '':
        value = date.today()
    if isinstance(value, (datetime, date)):
        return f'{value:%B} {value.day}, {value.year}'
    return str(value)


def _draw_cover(canv, cfg, default_title):
    """Paint the cover page directly on the canvas."""
    page_w, page_h = LETTER
    styles = getSampleStyleSheet()
    width = page_w - 2 * 72
    y = page_h - 90
    if cfg['logo']:
        reader = ImageReader(cfg['logo']['path'])
        img_w, img_h = _image_size(reader)
        w, h = _fit(img_w, img_h, cfg['logo']['max_width'], cfg['logo']['max_height'])
        canv.drawImage(reader, (page_w - w) / 2, y - h, w, h, mask='auto')
        y -= h
    y = min(y - 40, page_h * 0.62)

    def put(text, style, gap):
        nonlocal y
        para = Paragraph(text, style)
        _, ph = para.wrap(width, page_h)
        y -= ph
        para.drawOn(canv, 72, y)
        y -= gap

    title_style = ParagraphStyle('CoverTitle', parent=styles['Title'], fontSize=30, leading=36)
    sub_style = ParagraphStyle('CoverSub', parent=styles['Normal'], fontSize=16, leading=20, alignment=1)
    text_style = ParagraphStyle('CoverText', parent=styles['Normal'], fontSize=13, leading=18, alignment=1)
    put(_pdf_clean_text(cfg['title'] or default_title), title_style, 14)
    if cfg['subtitle']:
        put(_pdf_clean_text(cfg['subtitle']), sub_style, 30)
    if cfg['authors']:
        put('<br/>'.join(_pdf_clean_text(a) for a in cfg['authors']), text_style, 30)
    put(_pdf_clean_text(_cover_date_text(cfg['date'])), text_style, 0)


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
               , logo=None
               , watermark=None
               , page_numbers=None
               , cover=None
               , index=None
               , report_title=None
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
    report_title : str, optional
        Replaces the default report title (base date, "Sales Ratio Study"
        and model name) with this exact text.
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
        If provided, drawn as a classification label on every page.
    logo : str, Path or dict, optional, default None
        Logo drawn at the top of the report pages: an image path, or a dict
        with ``path``, ``pages`` (``'first'`` or ``'all'``), ``position``
        (``'top_left'``, ``'top_center'`` or ``'top_right'``) and
        ``max_height`` / ``max_width`` in points.
    watermark : str, Path or dict, optional, default None
        Watermark image drawn behind the content of every report page: a
        path, or a dict with ``path``, ``position`` (``'fill'``,
        ``'center'``, ``'top_left'``, ``'top_right'``, ``'bottom_left'`` or
        ``'bottom_right'``), ``opacity`` (0-1, default 0.15) and ``scale``
        (share of the page the image may take outside ``'fill'``).
    page_numbers : bool or dict, optional, default None
        Page numbers: True, or a dict with ``position`` (``'bottom_left'``,
        ``'bottom_center'`` or ``'bottom_right'``), ``font_size`` and
        ``format`` (``'number'`` or ``'page_of_n'``).
    cover : bool, str or dict, optional, default None
        Cover page with ``title``, ``subtitle``, ``authors``, ``date`` and
        its own ``logo``. It has no page number, no watermark, and is not
        counted as a page.
    index : bool or dict, optional, default None
        With a cover, adds a page after it holding the title and an index of
        sections with page numbers. Like the cover it is not numbered or
        counted.

    Returns
    -------
    None
        The PDF is written to ``export_pdf_filepath``; nothing is
        returned. Any files listed in ``gini_lorenz_plots`` are removed
        from disk once the document has been built.
    """
    try:
        _use_agg_if_headless()
        brand = _normalize_pdf_branding(logo, watermark, page_numbers, cover, index)
        toc_on = brand['index'] is not None
        if base_date:
            if isinstance(base_date, int): base_date = str(base_date)
        elements = []
        styles = getSampleStyleSheet()
        centered_style = styles['Normal'].clone('Centered')
        centered_style.alignment = 1
        if report_title is not None:
            title_text = report_title
        else:
            title_text = ((base_date + ' - ') if base_date else '') + "Sales Ratio Study" + ((' - ' + model_name) if model_name != '' else '')

        pending_sections = []       # level-0 index entries waiting for their first flowable

        def as_flowable(element):
            return Image(element, width=700, height=400) if isinstance(element, plt.Figure) else element

        def anchor(level, text, *flowables):
            """Elements for one report block; with an index, tagged with the entries that start on it."""
            flowables = [as_flowable(f) for f in flowables]
            if not toc_on:
                return flowables
            markers = [_TocMarker(0, t) for t in pending_sections]
            pending_sections.clear()
            if text:
                markers.append(_TocMarker(level, _pdf_clean_text(text)))
            return [KeepTogether(markers + flowables)] if markers else flowables

        def graph_block(graph):
            if isinstance(graph, list):
                block = KeepTogether([Paragraph(graph[0], centered_style), ConditionalSpacer(0, 0.2 * inch), as_flowable(graph[1])])
                return anchor(1, graph[0], block)
            return anchor(1, None, ConditionalSpacer(0, 0.5 * inch), graph)

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
            pending_sections.append('Ratio tables')
            if 'Overall' in ratio_tables_dfs:
                overall_ratios = ratio_tables_dfs.pop('Overall')
                if 'Overall' in pdf_grouping_aliases.keys():
                    overall_alias = pdf_grouping_aliases['Overall']
                else:
                    overall_alias = 'Overall'
                heading = "Time Adjusted Sales Ratios - " + overall_alias
                elements.extend(anchor(1, heading, Paragraph(heading, styles['Heading3'])))
                elements.append(_pdf_create_table(overall_ratios))
                elements.append(KeepTogether(ConditionalSpacer(0, 0.2 * inch)))
            for key in ratio_tables_dfs:
                group_ratios = ratio_tables_dfs[key]

                if key in pdf_grouping_aliases.keys():
                    group_alias = pdf_grouping_aliases[key]
                else:
                    group_alias = key

                heading = "Time Adjusted Sales Ratios by " + group_alias
                elements.extend(anchor(1, heading, Paragraph(heading, styles['Heading3'])))
                elements.append(_pdf_create_table(group_ratios))
                elements.append(KeepTogether(ConditionalSpacer(0, 0.2 * inch)))

        for section_title, section_graphs in (('Overall graphs', overall_graphs)
                                               , ('Group graphs', group_graphs)):
            if section_graphs:
                pending_sections.append(section_title)
                for graph in section_graphs:
                    elements.extend(graph_block(graph))

        for section_title, section_graphs in (('Group mosaic graphs', group_mosaic_graphs)
                                               , ('Group distribution mosaic graphs', dist_group_mosaic_graphs)):
            if section_graphs:
                pending_sections.append(section_title)
                for graph in section_graphs:
                    elements.extend(anchor(1, None, ConditionalSpacer(0, 0.5 * inch), graph))

        # lorenz plots
        if gini_lorenz_plots:
            pending_sections.append('Lorenz curves')
            for plot in gini_lorenz_plots:
                elements.extend(anchor(1, None, ConditionalSpacer(0, 0.5 * inch), Image(plot)))

        if ending_comments:
            if isinstance(ending_comments, str):
                ending_comments = [ending_comments]
            pending_sections.append('Comments')
            elements.extend(anchor(1, None, KeepTogether(ConditionalSpacer(0, 1 * inch))))
            for comment in ending_comments:
                if comment == 'spacer':
                    elements.append(KeepTogether(ConditionalSpacer(0, 0.2 * inch)))
                else:
                    elements.append(Paragraph(comment, styles['Normal']))


        # ---- page furniture -------------------------------------------------
        page_w, page_h = LETTER
        left_margin = right_margin = _PAGE_MARGIN_X
        bottom_margin = 40 if brand['page_numbers'] else 20
        top_margin = 50
        logo_cfg = brand['logo']
        logo_reader = ImageReader(logo_cfg['path']) if logo_cfg else None
        logo_top_margin = top_margin
        if logo_cfg:
            _, logo_y, _, _ = _logo_box(logo_cfg, logo_reader, bool(classification_header))
            logo_top_margin = max(top_margin, page_h - logo_y + 8)
        wm_reader = _watermark_reader(brand['watermark']['path'], brand['watermark']['opacity']) if brand['watermark'] else None
        label_func = _make_label_func(classification_header) if classification_header else None
        cover_title = brand['cover']['title'] if brand['cover'] and brand['cover']['title'] else title_text

        def page_furniture(canv, doc, with_logo):
            if wm_reader is not None:
                _draw_watermark(canv, wm_reader, brand['watermark'])
            if with_logo:
                x, y, w, h = _logo_box(logo_cfg, logo_reader, bool(classification_header))
                canv.drawImage(logo_reader, x, y, w, h, mask='auto')
            if label_func:
                label_func(canv, classification_header)

        def on_cover(canv, doc):
            _draw_cover(canv, brand['cover'], title_text)
            if label_func:
                label_func(canv, classification_header)

        def on_front(canv, doc):
            doc.front_pages += 1
            page_furniture(canv, doc, with_logo=bool(logo_cfg and logo_cfg['pages'] == 'all'))

        def on_body_first(canv, doc):
            canv._rs_numbered = True
            page_furniture(canv, doc, with_logo=bool(logo_cfg))

        def on_body_later(canv, doc):
            canv._rs_numbered = True
            page_furniture(canv, doc, with_logo=bool(logo_cfg and logo_cfg['pages'] == 'all'))

        def make_frame(top):
            return Frame(left_margin, bottom_margin, page_w - left_margin - right_margin, page_h - top - bottom_margin, id='normal')

        doc = _ReportDoc(export_pdf_filepath, pagesize=LETTER, topMargin=top_margin, bottomMargin=bottom_margin
                         , rightMargin=right_margin, leftMargin=left_margin)
        later_top = logo_top_margin if (logo_cfg and logo_cfg['pages'] == 'all') else top_margin
        templates = []
        if brand['cover']:
            templates.append(PageTemplate(id='cover', frames=[make_frame(top_margin)], onPage=on_cover, pagesize=LETTER))
        if brand['index']:
            templates.append(PageTemplate(id='front', frames=[make_frame(later_top)], onPage=on_front, pagesize=LETTER))
        templates.append(PageTemplate(id='body_first', frames=[make_frame(logo_top_margin)], onPage=on_body_first, pagesize=LETTER))
        templates.append(PageTemplate(id='body_later', frames=[make_frame(later_top)], onPage=on_body_later, pagesize=LETTER))
        doc.addPageTemplates(templates)
        # The logo and page numbers shrink the frame; scale down any image that no longer fits.
        frame_height = min(t.frames[0]._height for t in templates) - 12
        report_elements = [_shrink_images(as_flowable(e), frame_height) for e in elements]

        story = []
        if brand['cover']:
            story += [Spacer(1, 1), NextPageTemplate('front' if brand['index'] else 'body_first'), PageBreak()]
        if brand['index']:
            toc = TableOfContents()
            toc.dotsMinLevel = 0
            toc.levelStyles = [
                ParagraphStyle('TOC0', fontSize=12, leading=16, leftIndent=0, spaceBefore=8, fontName='Helvetica-Bold'),
                ParagraphStyle('TOC1', fontSize=10, leading=14, leftIndent=18),
            ]
            index_title = brand['index']['title'] or cover_title
            story += [Paragraph(_pdf_clean_text(index_title), styles['Title']), Spacer(1, 0.3 * inch)
                      , Paragraph('Index', styles['Heading2']), toc
                      , NextPageTemplate('body_first'), PageBreak()]
        story += [NextPageTemplate('body_later')] + report_elements

        canvas_class = _numbered_canvas_class(brand['page_numbers']) if brand['page_numbers'] else _rl_canvas.Canvas
        if toc_on:
            doc.multiBuild(story, canvasmaker=canvas_class)
        else:
            doc.build(story, canvasmaker=canvas_class)

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

# Indicators whose ideal value *is* zero: no vertical inequity (VEI, and the
# VEM's severity and prevalence) or no reranking (RRI). The fallback check
# for an indicator without a standard -- "its interval includes 0" -- is a
# finding for a bias or effect size, but for these it is the good outcome,
# so they skip it. The VEI is judged against its band by `VEI_conclusion`.
_ZERO_IS_IDEAL = ('VEI', 'Inequity_severity', 'Inequity_prevalence',
                  'Reranking_Index', 'RRI')

# IAAO 11.2.1 (Uniformity among Strata): a stratum's level of assessment
# should fall within this fraction of the overall level of assessment --
# 5 percent, not 5 percentage points.
_STRATA_MEDIAN_UNIFORMITY_PCT = 0.05

# IAAO 11.2.2 (Uniformity between Classes): no two strata's levels of
# assessment should differ from each other by more than this fraction --
# 10 percent, not 10 percentage points.
_INTERCLASS_MEDIAN_UNIFORMITY_PCT = 0.10

# Statistics that can stand for a stratum's level of assessment in the two
# checks above, in the order tried when the caller does not choose one: the
# Median first (the Standard's preferred measure of level, 7.4), then the
# Weighted Mean, then the Mean.
LEVEL_OF_ASSESSMENT_STATISTICS = ('Median', 'Weighted_Mean', 'Mean')
_LEVEL_OF_ASSESSMENT_ALIASES = {
    'median': 'Median',
    'mean': 'Mean',
    'weighted_mean': 'Weighted_Mean', 'weighted mean': 'Weighted_Mean',
    'weightedmean': 'Weighted_Mean', 'wtd_mean': 'Weighted_Mean',
    'aggregate': 'Weighted_Mean', 'aggregate_ratio': 'Weighted_Mean',
}


def normalize_level_of_assessment(level_of_assessment):
    """
    Canonical spelling of a level-of-assessment statistic.

    Parameters
    ----------
    level_of_assessment : str or None
        ``'Median'``, ``'Mean'`` or ``'Weighted_Mean'``, in any case, with
        ``'weighted mean'`` / ``'wtd_mean'`` / ``'aggregate'`` accepted for
        the last. None means "choose automatically".

    Returns
    -------
    str or None
        One of `LEVEL_OF_ASSESSMENT_STATISTICS`, or None.

    Raises
    ------
    ValueError
        For anything else.
    """
    if level_of_assessment is None:
        return None
    key = str(level_of_assessment).strip().lower()
    if key not in _LEVEL_OF_ASSESSMENT_ALIASES:
        raise ValueError(
            f'level_of_assessment={level_of_assessment!r} is not one of '
            f'{list(LEVEL_OF_ASSESSMENT_STATISTICS)}.')
    return _LEVEL_OF_ASSESSMENT_ALIASES[key]


def _resolve_level_of_assessment(columns, level_of_assessment=None):
    """
    The statistic the between-strata checks compare, given what the table holds.

    Parameters
    ----------
    columns : iterable of str
        Columns of the results table.
    level_of_assessment : str, optional
        The caller's choice. None picks the first of
        `LEVEL_OF_ASSESSMENT_STATISTICS` present -- the Median whenever the
        study computes it.

    Returns
    -------
    str or None
        The column to compare, or None when the chosen statistic (or, for
        None, every candidate) is missing from the table.
    """
    columns = set(columns)
    chosen = normalize_level_of_assessment(level_of_assessment)
    if chosen is not None:
        return chosen if chosen in columns else None
    return next((c for c in LEVEL_OF_ASSESSMENT_STATISTICS if c in columns), None)


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
    overall level of assessment +/- 5 percent (IAAO 11.2.1) rather than
    with the fixed band.

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
            # 5 percent of the overall level, not 5 percentage points (IAAO
            # 11.2.1); the two differ whenever the overall level is not 1.
            return (round(overall_median * (1 - _STRATA_MEDIAN_UNIFORMITY_PCT), 4),
                    round(overall_median * (1 + _STRATA_MEDIAN_UNIFORMITY_PCT), 4))
        return (low, round(high + 0.05, 2))
    return (low, high)


def _usable_ci(ci):
    """
    True for a ``(low, high)`` interval with two finite bounds.

    A blank interval can arrive as ``(nan, nan)`` -- a tuple, so an
    ``isinstance`` test passes it -- and every comparison against NaN is
    False, which would read a NaN gap as a failure.
    """
    if not isinstance(ci, tuple) or len(ci) != 2:
        return False
    low, high = _as_float(ci[0]), _as_float(ci[1])
    return low is not None and high is not None and math.isfinite(low) and math.isfinite(high)


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
                  compare_groups_median_to_overall_median=None,
                  std_non_acceptable_symbol=chr(8869),
                  small_sample_symbol=chr(248),
                  min_count_threshold=15,
                  ci_conf_level=None,
                  comments_header='Comments Ratio Study Standard metrics',
                  VE_bundle_comments=False,
                  add_binaries=True,
                  strata_uniformity_checks=True,
                  level_of_assessment=None):
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
    compare_groups_median_to_overall_median : bool, optional
        Older name for `strata_uniformity_checks`; when given, it wins.
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
    strata_uniformity_checks : bool, default True
        Run the horizontal-equity checks between strata. Both only apply
        when a `group_by` produced a ``'Group'`` column:

        * Stratum vs. overall, IAAO 11.2.1 (uniformity among strata) -- a
          stratum's level of assessment should be within 5 percent (not 5
          percentage points) of the overall level. With both confidence
          intervals available, the gap between their nearest bounds is
          taken as a percent of the overall's bound (see
          :func:`_strata_median_pct_change`); otherwise the two point
          estimates are compared, as a percent of the overall level.
        * Stratum vs. stratum, IAAO 11.2.2 (uniformity between classes) --
          every pair of strata sharing a ``'Level'`` (grouping dimension)
          is flagged when the gap between their nearest confidence bounds
          exceeds 10 percent of the lower stratum's bound (see
          :func:`_interclass_median_pct_change`). The Standard's "probably
          differ" is a statement about the intervals, so a pair without
          both is not compared.

        When the Median is not the level statistic, a stratum Median is
        held to the fixed ``assessment_standards`` band like any other
        indicator.
    level_of_assessment : {'Median', 'Mean', 'Weighted_Mean'}, optional
        Statistic standing for each stratum's level of assessment in those
        checks. Default: the Median when the table has it, else the
        Weighted Mean, else the Mean. A named statistic missing from the
        table switches the checks off.

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

        if compare_groups_median_to_overall_median is not None:
            strata_uniformity_checks = bool(compare_groups_median_to_overall_median)
        level_stat = (_resolve_level_of_assessment(df.columns, level_of_assessment)
                      if strata_uniformity_checks else None)
        overall_level = overall_level_ci = None
        if level_stat is not None:
            overall_rows = df['Level'] == 'Overall'
            if overall_rows.any():
                overall_level = _as_float(df.loc[overall_rows, level_stat].iloc[0])
                if f'{level_stat}_ci' in df.columns:
                    candidate = df.loc[overall_rows, f'{level_stat}_ci'].iloc[0]
                    overall_level_ci = candidate if _usable_ci(candidate) else None
        # The Median's own stratum band follows the overall Median only when
        # the Median is the level statistic; otherwise it is the fixed band.
        median_vs_overall = level_stat == 'Median'

        # `Weighted_Mean` is otherwise never checked; as the level statistic
        # it takes part in the stratum comparison, and only in that.
        cols_to_check = [col for col in results_cols
                         if col not in skip
                         and (col not in _NEVER_CHECKED or col == level_stat)
                         and '_ci' not in col
                         and 'Pvalue' not in col]
        overall_median = None
        if 'Median' in df.columns:
            overall_medians = df.loc[df['Level'] == 'Overall', 'Median']
            if len(overall_medians):
                overall_median = _as_float(overall_medians.iloc[0])

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

                # IAAO 11.2.1 (uniformity among strata): the stratum's level
                # of assessment against the overall level, within 5 percent.
                if col == level_stat and is_group and overall_level is not None:
                    group_level = _as_float(value)
                    if group_level is None:
                        continue
                    group_ci = df.at[idx, f'{col}_ci'] if f'{col}_ci' in df.columns else None
                    if _usable_ci(group_ci) and overall_level_ci is not None:
                        pct_change = _strata_median_pct_change(
                            group_level, group_ci, overall_level, overall_level_ci)
                        basis = f'compared at the nearest bounds of the two {interval_label}s'
                    else:
                        # No interval on one side or the other: the point
                        # estimates, still as a percent of the overall level.
                        pct_change = (abs(group_level - overall_level) / overall_level
                                      if overall_level else None)
                        basis = 'point estimates, without confidence intervals on both sides'
                    if pct_change is not None and pct_change > _STRATA_MEDIAN_UNIFORMITY_PCT:
                        notes.append(
                            f'{col} suggests a review under IAAO 11.2.1 '
                            '(uniformity among strata): '
                            f'{round(pct_change * 100, 2)}% different from '
                            'the overall level of assessment, more than the '
                            f'{round(_STRATA_MEDIAN_UNIFORMITY_PCT * 100)}% '
                            f'limit ({basis})'
                        )
                        flags.append(col)
                    continue
                if col in _NEVER_CHECKED:
                    continue

                band = _standard_band(
                    _standard_for(col, assessment_standards), col,
                    is_group,
                    overall_median=overall_median,
                    compare_groups_median_to_overall_median=median_vs_overall,
                )
                confidence_interval = df.at[idx, f'{col}_ci'] if f'{col}_ci' in df.columns else None

                # Without a standard, the only check available is whether
                # the confidence interval can rule out zero.
                if band is None and col in _ZERO_IS_IDEAL:
                    continue
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
        if 'Group' in df.columns and level_stat is not None:
            for level_value in df.loc[df['Level'] != 'Overall', 'Level'].unique():
                peers = [idx for idx in df.index
                        if df.at[idx, 'Level'] == level_value and idx not in small_samples]
                for i, idx_a in enumerate(peers):
                    median_a = _as_float(df.at[idx_a, level_stat])
                    ci_a = df.at[idx_a, f'{level_stat}_ci'] if f'{level_stat}_ci' in df.columns else None
                    if median_a is None or not _usable_ci(ci_a):
                        continue
                    for idx_b in peers[i + 1:]:
                        median_b = _as_float(df.at[idx_b, level_stat])
                        ci_b = df.at[idx_b, f'{level_stat}_ci'] if f'{level_stat}_ci' in df.columns else None
                        if median_b is None or not _usable_ci(ci_b):
                            continue
                        pct_change = _interclass_median_pct_change(median_a, ci_a, median_b, ci_b)
                        if pct_change is None or pct_change <= _INTERCLASS_MEDIAN_UNIFORMITY_PCT:
                            continue
                        pct_text = round(pct_change * 100, 2)
                        limit_text = round(_INTERCLASS_MEDIAN_UNIFORMITY_PCT * 100)
                        for this_idx, other_idx in ((idx_a, idx_b), (idx_b, idx_a)):
                            comments[this_idx].append(
                                f'{level_stat} (level of assessment) does not pass IAAO 11.2.2 '
                                f'(uniformity between classes): {pct_text}% different '
                                f"from Group {df.at[other_idx, 'Group']!r}, more than "
                                f'the {limit_text}% limit (compared at the nearest '
                                f'bounds of the two {interval_label}s) -- suggests a '
                                'review regarding assessment standards'
                            )
                            flagged[this_idx].append(level_stat)

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
                compare_groups_median_to_overall_median=median_vs_overall,
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
