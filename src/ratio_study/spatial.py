"""Spatial diagnostics for ratio studies: neighbourhood ratio maps and
Moran's I hot/cold-spot analysis.

Every function here expects the sales frame to carry latitude/longitude
columns (default names ``'latitude'``/``'longitude'``), and to be a real
pandas DataFrame -- these functions build ``geopandas.GeoDataFrame``s and
render interactive maps, neither of which narwhals abstracts over.

For a plain numeric Global Moran's I without the mapping/plotting stack,
see :func:`ratio_study.indicators.moran_i`.

Requires the ``spatial`` extra::

    pip install "ratio_study[spatial]"
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import _optional

__all__ = ['ratio_map', 'vem_map', 'global_moran', 'local_moran', 'open_map_in_browser']


# Default map height. lonboard defaults to 500px, which against a full-width
# notebook cell gives a letterbox strip that is hard to read a jurisdiction on.
DEFAULT_MAP_HEIGHT = 700

# Height step for the Taller / Shorter buttons, and the floor they stop at.
_HEIGHT_STEP = 200
_MIN_HEIGHT = 200

# ``radius_scale`` every map's ScatterplotLayer used before ``dot_size`` was
# an argument. Kept as the default so an existing call draws the same dots.
DEFAULT_DOT_SIZE = 40

# Semi-transparent by default. The layers were fully opaque before
# ``opacity`` was an argument, but on any real roll the dots overlap, and a
# solid fill turns a dense area into a flat blob that says nothing about how
# many sales are stacked there. At 0.4 the stacking reads: a patch of two
# hundred sales is visibly darker than a patch of five, and the basemap
# stays legible underneath. Pass ``opacity=1`` for the old solid dots.
DEFAULT_DOT_OPACITY = 0.4


def _as_opacity(opacity):
    """
    Validate a dot opacity and return it as a fraction in ``[0, 1]``.

    Accepts a fraction (``0.4``) or a percentage (``40``), the same rule
    `mvp_sale_price_share` and the confidence levels follow. Checked here
    rather than left to lonboard's trait, whose own out-of-range error
    names the deck.gl attribute and not the argument the caller typed.

    Parameters
    ----------
    opacity : float
        Opacity, in ``[0, 1]`` or ``(1, 100]``.

    Returns
    -------
    float
        The opacity as a fraction in ``[0, 1]``.

    Raises
    ------
    ValueError
        If `opacity` is not a finite number resolving into ``[0, 1]``.
    """
    try:
        value = float(opacity)
    except (TypeError, ValueError):
        raise ValueError(f'opacity must be a number, got {opacity!r}.')
    if not np.isfinite(value):
        raise ValueError(f'opacity must be finite, got {opacity!r}.')
    if 1 < value <= 100:
        value = value / 100.0
    if not 0 <= value <= 1:
        raise ValueError(
            'opacity must be a fraction in [0, 1] (or a percentage in '
            f'(1, 100]), got {opacity!r}.')
    return value


def _extract_map(panel_or_map):
    """
    Return the lonboard Map from either a Map or a panel built by `_map_panel`.

    Parameters
    ----------
    panel_or_map : lonboard.Map or ipywidgets.VBox
        A map, or the panel a mapping function returned.

    Returns
    -------
    lonboard.Map

    Raises
    ------
    TypeError
        If no map can be found.
    """
    if hasattr(panel_or_map, 'to_html'):
        return panel_or_map
    for child in getattr(panel_or_map, 'children', ()):
        if hasattr(child, 'to_html'):
            return child
    raise TypeError(
        'expected a lonboard Map or a panel from ratio_map()/local_moran(), '
        f'got {type(panel_or_map).__name__}'
    )


def open_map_in_browser(panel_or_map, filename=None, title='Ratio study map',
                        open_browser=True):
    """
    Write a map to a standalone HTML file and open it in the default browser.

    The notebook cell is a poor container for a map -- it is a fixed strip in
    a scrolling document, and the notebook's own "maximise output" control
    cannot grow it because lonboard writes the height as an inline style.
    A real browser tab has none of those constraints: the map gets the whole
    window, and the file keeps working after the kernel stops.

    The written file is self-contained (the map data is inlined), so it can be
    emailed or attached to a report.

    Parameters
    ----------
    panel_or_map : lonboard.Map or ipywidgets.VBox
        A map, or the panel returned by :func:`ratio_map` / :func:`local_moran`.
    filename : str or Path, optional
        Where to write. Defaults to a uniquely named file in the system
        temporary directory.
    title : str, default 'Ratio study map'
        Browser tab title for the exported page.
    open_browser : bool, default True
        Open the file in the default browser. Set False to write only --
        useful in headless runs, or when you want the path to hand to
        something else.

    Returns
    -------
    pathlib.Path
        The file written.

    Examples
    --------
    >>> panel = ratio_map(df=sales, column='asr', groupby='neighbourhood')  # doctest: +SKIP
    >>> open_map_in_browser(panel)                                          # doctest: +SKIP

    To choose a specific browser, resolve it through :mod:`webbrowser` and
    pass ``open_browser=False``::

        import webbrowser
        path = open_map_in_browser(panel, open_browser=False)
        webbrowser.get('firefox').open(path.as_uri())
    """
    import tempfile
    import webbrowser

    m = _extract_map(panel_or_map)

    if filename is None:
        handle, name = tempfile.mkstemp(prefix='ratio_study_map_', suffix='.html')
        os.close(handle)
        filename = name
    path = Path(filename)

    m.to_html(path, title=title)
    if open_browser:
        webbrowser.open(path.as_uri())
    return path


def _map_panel(m, legend_box, height, widgets, title='Ratio study map'):
    """
    Wrap a lonboard map and its legend in a resizable panel.

    The controls are real ipywidgets buttons driven by Python callbacks, not
    browser scripting: an earlier version used a Fullscreen API button, but
    embedded notebook webviews reject ``requestFullscreen()`` by Permissions
    Policy, so it did nothing wherever it was most needed. Stepping the
    height through the widget's own trait works in every front end, and
    "Open in browser" sidesteps the notebook container entirely.

    Parameters
    ----------
    m : lonboard.Map
        The map to wrap. Its ``height`` is set here.
    legend_box : ipywidgets.Widget
        Legend laid out beneath the map.
    height : int or str
        Map height in CSS pixels (``700``) or any CSS length (``'80vh'``).
    widgets : module
        The resolved ``ipywidgets`` module, passed in so this helper does
        not repeat the optional-dependency lookup.
    title : str, default 'Ratio study map'
        Browser tab title used by the "Open in browser" button.

    Returns
    -------
    ipywidgets.VBox
        Controls, map, legend and a status line.
    """
    m.height = height
    m.add_class('rs-map')

    status = widgets.HTML('')

    def _step(delta):
        def handler(_button):
            current = m.height
            if isinstance(current, str) and current.endswith('px'):
                m.height = max(_MIN_HEIGHT, int(float(current[:-2])) + delta)
                status.value = ''
            else:
                # A relative length ('85vh') has no pixel value to step from,
                # so say so rather than silently doing nothing.
                status.value = (f'<span style="color:#8a6d3b">Height is set to '
                                f'<code>{current}</code>; pass a pixel height to use '
                                f'these buttons.</span>')
        return handler

    def _open(_button):
        try:
            path = open_map_in_browser(m, title=title)
            status.value = (f'<span style="color:#3c763d">Opened in your browser — '
                            f'<code>{path}</code></span>')
        except Exception as ex:                      # pragma: no cover - env specific
            status.value = (f'<span style="color:#a94442">Could not open a browser: '
                            f'{ex}</span>')

    browser_button = widgets.Button(description='Open in browser', icon='external-link',
                                    tooltip='Write a standalone HTML map and open it in '
                                            'your default browser')
    taller = widgets.Button(description='Taller', icon='plus',
                            tooltip=f'Grow the map by {_HEIGHT_STEP}px')
    shorter = widgets.Button(description='Shorter', icon='minus',
                             tooltip=f'Shrink the map by {_HEIGHT_STEP}px')

    browser_button.on_click(_open)
    taller.on_click(_step(_HEIGHT_STEP))
    shorter.on_click(_step(-_HEIGHT_STEP))

    controls = widgets.HBox([browser_button, taller, shorter])

    panel = widgets.VBox([controls, m, legend_box, status])
    panel.add_class('rs-map-panel')
    return panel


def _fit_view(lon, lat, outlier_k=3.0):
    """
    Centre and zoom that fit the bulk of ``(lon, lat)``, robust to a few
    contaminated coordinates.

    lonboard's own auto-fit takes the exact min/max of every point, so a
    single bad geocode -- ``(0, 0)`` from a failed lookup is the classic
    case -- balloons the view all the way out to the whole globe for
    everyone else's sale. Tukey's IQR fence drops points far outside the
    bulk of the distribution on `lon` and `lat` independently before
    fitting to the rest, which (unlike a fixed percentile clip) still
    catches a single wild point regardless of how many good ones there
    are, so the view stays honest about where the data actually is
    without needing the caller to have already cleaned it.

    Parameters
    ----------
    lon, lat : array-like
        Coordinates, in degrees. Non-finite values are dropped.
    outlier_k : float, default 3.0
        IQR multiplier for the fence: Tukey's standard cutoff for a "far"
        outlier. A point outside ``Q1 - k*IQR`` .. ``Q3 + k*IQR`` on
        either axis is excluded from the fit.

    Returns
    -------
    dict
        ``{'longitude', 'latitude', 'zoom'}``, ready to pass straight to
        ``Map.set_view_state(**view)``.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]

    def _fence(values):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return values.min(), values.max()
        return q1 - outlier_k * iqr, q3 + outlier_k * iqr

    lon_lo, lon_hi = _fence(lon)
    lat_lo, lat_hi = _fence(lat)
    keep = (lon >= lon_lo) & (lon <= lon_hi) & (lat >= lat_lo) & (lat <= lat_hi)
    if keep.any():
        lon, lat = lon[keep], lat[keep]

    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    # A span of 0 (one point left, or every point identical) has no finite
    # zoom; floor it rather than dividing by zero.
    span = max(lon_max - lon_min, lat_max - lat_min, 1e-9)
    zoom = max(1, min(20, int(math.log2(360.0 / span))))
    return {
        'longitude': float((lon_min + lon_max) / 2),
        'latitude': float((lat_min + lat_max) / 2),
        'zoom': zoom,
    }


def _cube_root(x):
    x = abs(x)
    return int(round(x ** (1. / 3)))


def _quantile_bin_edges(values, n_bins=5):
    """
    Quantile-based bin edges over ``values``, deduplicated.

    Replaces fixed ratio cutoffs (e.g. hardcoded 0.85/0.90/...) with
    edges derived from the actual distribution of ratios in the sample,
    so the colour scale always reflects what is really in the data
    rather than an assumption about where "typical" ratios fall.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    return np.unique(edges)


def _bin_index(x, edges):
    """1-based bin index of ``x`` within ``edges`` (clamped to range)."""
    idx = int(np.searchsorted(edges, x, side='right'))
    return min(max(idx, 1), len(edges) - 1)


#  ratio_map's colour scheme
# --------------------------------------------------------------------------
# The two tails are fixed thresholds and fixed colours -- not quantiles of
# whatever happens to be in the sample -- so "this neighbourhood is a low/high
# outlier" means the same thing on every map. Only used by the legacy
# ``colour_code_by='quantile'`` scheme; the three statistical schemes set
# their own cutoffs from the distribution instead.
LOW_OUTLIER_FRACTION = 0.80     # below this fraction of the median ASR: low outlier
HIGH_OUTLIER_FRACTION = 1.20    # above this fraction of the median ASR: high outlier
LOW_OUTLIER_COLOR = (0, 0, 139)     # dark blue
HIGH_OUTLIER_COLOR = (211, 47, 47)  # red


# --------------------------------------------------------------------------
#  Five-band colour coding
# --------------------------------------------------------------------------
# Five bands rather than seven. A reader takes "normal / drifting high or low
# / far out high or low" off a map at a glance; the extra pair of shades the
# seven-bin scale spent on the middle only ever split the central mass finer
# than anyone reads it, and made the two that mattered harder to pick out.
#
# The codes run low -> high so a categorical colormap keyed on them is also a
# legible diverging scale, and the two extremes are the only dark fills, so
# they are the first thing the eye lands on. Blue and orange rather than the
# more common red-green: blue/orange sits on the axis protanopia leaves
# intact, so the map keeps working without colour vision for hue.
BAND_MISSING = 0
BAND_VERY_LOW = 1
BAND_LOW = 2
BAND_CENTRAL = 3
BAND_HIGH = 4
BAND_VERY_HIGH = 5

BAND_COLOURS = {
    BAND_MISSING: (255, 255, 255),     # white -- no ratio to place
    BAND_VERY_LOW: (0, 0, 139),        # dark blue
    BAND_LOW: (158, 202, 225),         # light blue
    BAND_CENTRAL: (189, 189, 189),     # grey
    BAND_HIGH: (253, 190, 133),        # light orange
    BAND_VERY_HIGH: (211, 47, 47),     # red
}


# Reading order for the legend: most under-assessed to most over-assessed,
# with the "no ratio" bucket last because it is rarely present.
_BAND_ORDER = (BAND_VERY_LOW, BAND_LOW, BAND_CENTRAL, BAND_HIGH,
               BAND_VERY_HIGH, BAND_MISSING)

# Bands drawn underneath the rest. Grey is almost always the bulk of the
# sample, so on a dense roll it buries the handful of dots the map exists to
# show -- one grey sale landing on top of a red neighbourhood hides the only
# thing worth looking at. White ("no ratio") sits back there for the same
# reason: it carries no finding and would obscure one.
BACKGROUND_BANDS = (BAND_CENTRAL, BAND_MISSING)

def _style_legend_button(button, rgb):
    """
    Paint a legend button its band's colour, with readable text on top.

    The text colour is derived from the fill rather than listed per band:
    a hardcoded "these ones are dark" tuple has to be remembered every time
    a colour changes, and a swatch recoloured without it goes unreadable.
    The 0.299/0.587/0.114 weights are the standard relative-luminance
    approximation -- green dominates because the eye does.

    Parameters
    ----------
    button : ipywidgets.Button
        The legend entry to style, modified in place.
    rgb : tuple of int
        The band's fill colour, 0-255 per channel.
    """
    button.style.button_color = 'rgb({},{},{})'.format(*rgb)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if luminance < 140:
        button.style.text_color = 'rgb(255,255,255)'


#  local_moran's hot/cold-spot classes
# --------------------------------------------------------------------------
# Code -> (legend label, RGB). One table, because the label and the colour
# used to be written out three times each -- once in a `value_label`
# if-chain, once in the colormap, and once more in each hand-built legend
# button -- so editing one copy silently left the other two behind. Recolour
# a class here and the dots, the tooltip and the legend all follow.
#
# Seven significance classes, not five: these are z-score confidence bands
# (99/95/90 % each way), not a colour-binning choice like `ratio_map`'s
# five, so collapsing them would mean dropping the 90 % bands. Grey for "not
# significant" -- the bulk of any roll, and the thing the reader is meant to
# look past; it was black, which drew the eye to exactly the sales with no
# finding on them.
MORAN_NOT_SIGNIFICANT = 4
MORAN_UNCLASSIFIED = 0

MORAN_CLASSES = {
    MORAN_UNCLASSIFIED: ('Error classifying statistic', (255, 255, 255)),
    1: ('coldspot 99% confidence', (31, 120, 180)),
    2: ('coldspot 95% confidence', (38, 187, 219)),
    3: ('coldspot 90% confidence', (146, 255, 225)),
    MORAN_NOT_SIGNIFICANT: ('not significant', (128, 128, 128)),
    5: ('hotspot 90% confidence', (234, 215, 40)),
    6: ('hotspot 95% confidence', (253, 110, 92)),
    7: ('hotspot 99% confidence', (203, 26, 0)),
}

# Significance bands, widest first: (|z| threshold, max p, cold code, hot
# code). Stated once on the *absolute* z with the sign picked afterwards, so
# the cold and hot sides cannot drift apart. The if/elif chain this replaces
# spelled each side out separately and had lost the minus sign on its very
# first branch (`z <= 2.58` where it meant `z <= -2.58`), which classified
# every significant point with z below +2.58 as a 99 % coldspot and left the
# 90 % and 95 % hotspot classes unreachable whenever p <= 0.01.
MORAN_BANDS = (
    (2.58, 0.01, 1, 7),      # 99 % confidence
    (1.96, 0.05, 2, 6),      # 95 %
    (1.65, 0.10, 3, 5),      # 90 %
)

# Within this |z| and this p, a point is called not significant.
MORAN_NOT_SIGNIFICANT_Z = 1.65
MORAN_NOT_SIGNIFICANT_P = 0.10


def _moran_class(z_value, p_value):
    """
    Classify one Getis-Ord z/p pair into a `MORAN_CLASSES` code.

    Bands are tested widest-first, so a point qualifying at 99 % is not
    also reported at 95 %. `z_value` follows ``G_Local``'s own convention:
    positive means the point sits in a cluster of *high* values, which for
    an ASR is a cluster of over-assessment -- a hotspot.

    A point outside every band that is also not inside the
    not-significant box falls to `MORAN_UNCLASSIFIED`. That is not
    unreachable bookkeeping: the p-value comes from a permutation
    simulation and the z from the normal approximation, so the two can
    disagree -- a large |z| with a middling p, or the reverse.

    Parameters
    ----------
    z_value : float
        Standardised Getis-Ord statistic, positive for a high-value cluster.
    p_value : float
        Two-tailed p-value for that statistic.

    Returns
    -------
    int
        A key of `MORAN_CLASSES`.

    Examples
    --------
    >>> _moran_class(3.0, 0.005) == 7      # hotspot, 99%
    True
    >>> _moran_class(-3.0, 0.005) == 1     # coldspot, 99%
    True
    >>> _moran_class(0.2, 0.8) == MORAN_NOT_SIGNIFICANT
    True
    """
    for z_threshold, max_p, cold_code, hot_code in MORAN_BANDS:
        if abs(z_value) >= z_threshold and p_value <= max_p:
            return hot_code if z_value > 0 else cold_code
    if abs(z_value) <= MORAN_NOT_SIGNIFICANT_Z and p_value >= MORAN_NOT_SIGNIFICANT_P:
        return MORAN_NOT_SIGNIFICANT
    return MORAN_UNCLASSIFIED


COLOUR_CODE_SCHEMES = ('iqr', 'mad', 'standard', 'quantile')

DEFAULT_IQR_BANDS = (1.5, 3.0)
DEFAULT_MAD_BANDS = (3.5, 5.0)
DEFAULT_STANDARD_BANDS = (0.10, 0.25)


def _band_cutoffs(bands, name):
    """Validate a ``(inner, outer)`` pair of band cutoffs and return it sorted."""
    try:
        inner, outer = (float(b) for b in bands)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a pair of numbers, got {bands!r}.')
    if not (np.isfinite(inner) and np.isfinite(outer)):
        raise ValueError(f'{name} must be finite, got {bands!r}.')
    if inner <= 0 or outer <= 0:
        raise ValueError(f'{name} cutoffs must be positive, got {bands!r}.')
    if inner >= outer:
        raise ValueError(
            f'{name} must be (inner, outer) with inner < outer, got {bands!r}.')
    return inner, outer


def _bands_from_deviation(deviation, bands):
    """
    Band codes from a signed deviation measure and its two cutoffs.

    The one place the five-band split is written down: everything inside
    ``+/- inner`` is central, the ring out to ``outer`` is the light
    orange/light blue pair, and past ``outer`` is red/dark blue. Non-finite
    deviations -- a missing ratio, or a measure that could not be scaled --
    fall to `BAND_MISSING` rather than being swept into a coloured band.

    Parameters
    ----------
    deviation : array-like
        Signed distance from the centre, in whatever units the scheme
        measures in (IQR multiples, MADs, or a proportion of the median).
    bands : tuple of float
        ``(inner, outer)`` cutoffs, already validated.

    Returns
    -------
    numpy.ndarray
        1D array of ``BAND_*`` codes.
    """
    inner, outer = bands
    dev = np.asarray(deviation, dtype=float)
    codes = np.full(dev.shape, BAND_CENTRAL, dtype=int)
    codes[dev > inner] = BAND_HIGH
    codes[dev > outer] = BAND_VERY_HIGH
    codes[dev < -inner] = BAND_LOW
    codes[dev < -outer] = BAND_VERY_LOW
    codes[~np.isfinite(dev)] = BAND_MISSING
    return codes


def _classify_iqr(values, bands, log_transform=True, min_cod=0):
    """
    Band codes from :func:`ratio_study.helpers.iqr_filter` run at both cutoffs.

    Reuses the package's own IQR filter rather than taking fresh quartiles
    here, so a point the study would have trimmed at 3 IQRs is exactly the
    point the map paints as a far outlier -- including the log transform,
    which keeps the fences symmetric on a right-skewed ratio distribution.
    The filter only reports in/out, so the side is decided separately, by
    comparing against the median.

    Parameters
    ----------
    values : array-like
        The statistic being mapped (a ratio, or a group's median ratio).
    bands : tuple of float
        ``(inner, outer)`` IQR multiples, e.g. ``(1.5, 3.0)``.
    log_transform : bool, default True
        Passed to ``iqr_filter``: take the fences on ``log(value)``.
    min_cod : float, default 0
        Passed to ``iqr_filter``. Defaults to 0 -- *unlike* the filter's own
        0.06 -- because that gate exists to stop a tight sample being
        stripped of legitimate sales, and colouring a point removes nothing.
        Left at 0.06 here, a low-dispersion sample would come out entirely
        grey with no way to see its relative spread.

    Returns
    -------
    numpy.ndarray
        1D array of ``BAND_*`` codes.
    """
    from .helpers import iqr_filter

    inner, outer = bands
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    codes = np.full(values.shape, BAND_MISSING, dtype=int)
    if not finite.any():
        return codes

    inner_flags = np.asarray(iqr_filter(values, iqr_mult=inner,
                                        log_transform=log_transform,
                                        min_cod=min_cod))
    outer_flags = np.asarray(iqr_filter(values, iqr_mult=outer,
                                        log_transform=log_transform,
                                        min_cod=min_cod))
    above = values > np.median(values[finite])

    codes[finite] = BAND_CENTRAL
    beyond_inner = finite & (inner_flags == 1)
    beyond_outer = finite & (outer_flags == 1)
    codes[beyond_inner & above] = BAND_HIGH
    codes[beyond_inner & ~above] = BAND_LOW
    codes[beyond_outer & above] = BAND_VERY_HIGH
    codes[beyond_outer & ~above] = BAND_VERY_LOW
    return codes


def _classify_mad(values, bands):
    """
    Band codes from the median absolute deviation.

    The MAD is the median of ``|value - median|``, unscaled -- a cutoff of
    1.5 means "one and a half typical deviations from the middle", not a
    normal-theory sigma. A sample with no spread at all (MAD of 0) is all
    central: there is no deviation to band on.

    Parameters
    ----------
    values : array-like
        The statistic being mapped.
    bands : tuple of float
        ``(inner, outer)`` MAD multiples, e.g. ``(3.5, 5.0)``.

    Returns
    -------
    numpy.ndarray
        1D array of ``BAND_*`` codes.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    dev = np.full(values.shape, np.nan)
    if not finite.any():
        return np.full(values.shape, BAND_MISSING, dtype=int)

    median = np.median(values[finite])
    mad = np.median(np.abs(values[finite] - median))
    dev[finite] = 0.0 if mad == 0 else (values[finite] - median) / mad
    return _bands_from_deviation(dev, bands)


def _classify_standard(values, bands):
    """
    Band codes from the proportional distance to the median.

    The fixed-threshold scheme: bands are a share of the median itself, so
    "10 % off the level of assessment" means the same thing on every map
    regardless of how dispersed that particular sample happens to be. This
    is the one scheme comparable across jurisdictions.

    Parameters
    ----------
    values : array-like
        The statistic being mapped.
    bands : tuple of float
        ``(inner, outer)`` proportions of the median, e.g. ``(0.10, 0.25)``.

    Returns
    -------
    numpy.ndarray
        1D array of ``BAND_*`` codes.

    Raises
    ------
    ValueError
        If the median is not positive, leaving nothing to take a proportion of.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, BAND_MISSING, dtype=int)

    median = np.median(values[finite])
    if median <= 0:
        raise ValueError(
            "colour_code_by='standard' bands are a share of the median, but "
            f'the median of the mapped values is {median!r}. Use '
            "colour_code_by='iqr' or 'mad' for a column that is not "
            'positively scaled.')

    dev = np.full(values.shape, np.nan)
    dev[finite] = (values[finite] - median) / median
    return _bands_from_deviation(dev, bands)


def _band_descriptions(scheme, bands, column):
    """
    Legend text for each band under a given scheme.

    Parameters
    ----------
    scheme : {'iqr', 'mad', 'standard'}
        Which classifier produced the codes.
    bands : tuple of float
        The ``(inner, outer)`` cutoffs actually used.
    column : str
        Name of the mapped column, for the central band's wording.

    Returns
    -------
    dict
        ``BAND_*`` code -> description.
    """
    inner, outer = bands
    if scheme == 'standard':
        inner_txt, outer_txt = f'{inner:.0%}', f'{outer:.0%}'
        return {
            BAND_VERY_LOW: f'more than {outer_txt} below the median {column}',
            BAND_LOW: f'{inner_txt} to {outer_txt} below the median',
            BAND_CENTRAL: f'within {inner_txt} of the median',
            BAND_HIGH: f'{inner_txt} to {outer_txt} above the median',
            BAND_VERY_HIGH: f'more than {outer_txt} above the median {column}',
        }
    unit = 'x IQR' if scheme == 'iqr' else 'MAD'
    return {
        BAND_VERY_LOW: f'below -{outer:g} {unit}',
        BAND_LOW: f'-{inner:g} to -{outer:g} {unit}',
        BAND_CENTRAL: f'within +/-{inner:g} {unit} of the median {column}',
        BAND_HIGH: f'+{inner:g} to +{outer:g} {unit}',
        BAND_VERY_HIGH: f'above +{outer:g} {unit}',
    }


def _band_legend(codes, values, column, descriptions, widgets):
    """
    Legend buttons for the bands actually present, with counts and ranges.

    The count and share answer "is that red patch three sales or a third of
    the roll?" without leaving the map, and the observed range of the band
    puts the classifier's abstract cutoffs back into ratio units the reader
    recognises.

    Parameters
    ----------
    codes : numpy.ndarray
        ``BAND_*`` code per mapped point.
    values : array-like
        The mapped statistic, same length as `codes`.
    column : str
        Name of the mapped column, used in the "no ratio" entry.
    descriptions : dict
        Band descriptions from :func:`_band_descriptions`.
    widgets : module
        The resolved ``ipywidgets`` module.

    Returns
    -------
    list of ipywidgets.Button
        Disabled buttons, one per non-empty band, in `_BAND_ORDER`.
    """
    codes = np.asarray(codes)
    values = np.asarray(values, dtype=float)
    total = max(codes.size, 1)
    layout = widgets.Layout(width='auto')

    items = []
    for code in _BAND_ORDER:
        mask = codes == code
        n = int(mask.sum())
        if not n:
            continue
        if code == BAND_MISSING:
            text = f'no {column}: {n:,}'
        else:
            band = values[mask]
            text = (f'{descriptions[code]}: {n:,} ({n / total:.1%}) '
                    f'[{np.nanmin(band):.3f}-{np.nanmax(band):.3f}]')
        button = widgets.Button(disabled=True, description=text, layout=layout)
        _style_legend_button(button, BAND_COLOURS[code])
        items.append(button)
    return items


def _colour_code(values, colour_code_by, column, iqr_bands, iqr_log_transform,
                 iqr_min_cod, mad_bands, standard_bands):
    """
    Dispatch to the requested classifier and pair the codes with legend text.

    Parameters
    ----------
    values : array-like
        The statistic being mapped.
    colour_code_by : {'iqr', 'mad', 'standard'}
        Scheme to apply. ``'quantile'`` is handled by `ratio_map` itself and
        does not come through here.
    column : str
        Name of the mapped column, for the legend wording.
    iqr_bands, mad_bands, standard_bands : tuple of float
        Candidate ``(inner, outer)`` cutoffs; only the requested scheme's
        pair is read and validated.
    iqr_log_transform : bool
        Passed to ``iqr_filter`` for the ``'iqr'`` scheme.
    iqr_min_cod : float
        Passed to ``iqr_filter`` for the ``'iqr'`` scheme.

    Returns
    -------
    tuple
        ``(codes, descriptions)``.
    """
    if colour_code_by == 'iqr':
        bands = _band_cutoffs(iqr_bands, 'iqr_bands')
        codes = _classify_iqr(values, bands, log_transform=iqr_log_transform,
                              min_cod=iqr_min_cod)
    elif colour_code_by == 'mad':
        bands = _band_cutoffs(mad_bands, 'mad_bands')
        codes = _classify_mad(values, bands)
    elif colour_code_by == 'standard':
        bands = _band_cutoffs(standard_bands, 'standard_bands')
        codes = _classify_standard(values, bands)
    else:                                       # pragma: no cover - guarded above
        raise ValueError(f'unsupported colour_code_by={colour_code_by!r}')
    return codes, _band_descriptions(colour_code_by, bands, column)


def _layered_scatter(gdf, categories, background, cmap, dot_size, opacity,
                     lonboard, lonboard_cmap):
    """
    Split the points into a background and a foreground scatter layer.

    deck.gl draws layers in the order the ``Map`` receives them, so the
    background points go in the first layer and end up underneath. Draw
    order *within* one layer is row order, which is an implementation
    detail rather than a guarantee -- hence two layers rather than sorting
    the frame, which is also why this cannot just reorder ``gdf``.

    Both layers get the same ``dot_size`` and ``opacity``, so the split is
    invisible except in what covers what. A layer with no rows is skipped
    entirely: ``from_geopandas`` on an empty frame has no geometry to build
    an arrow table from.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        One row per mapped point.
    categories : array-like
        Category code per row, keyed into `cmap` for the fill colour.
    background : array-like of bool
        True for the rows to draw underneath, same length as `gdf`.
    cmap : dict
        Category code -> RGB tuple.
    dot_size : float
        ``radius_scale`` for both layers.
    opacity : float
        Fill opacity for both layers, 0 to 1.
    lonboard, lonboard_cmap : module
        Resolved optional dependencies, passed in so this helper does not
        repeat the lookup.

    Returns
    -------
    list of lonboard.ScatterplotLayer
        Background layer first, foreground second; either may be absent.
    """
    categories = np.asarray(categories)
    background = np.asarray(background, dtype=bool)

    layers = []
    for mask in (background, ~background):
        if not mask.any():
            continue
        subset = gdf[mask]
        fill_color = lonboard_cmap.apply_categorical_cmap(
            pd.Series(categories[mask], index=subset.index), cmap=cmap)
        layers.append(lonboard.ScatterplotLayer.from_geopandas(
            subset, pickable=True, radius_scale=dot_size, opacity=opacity,
            auto_highlight=True, get_fill_color=fill_color,
            get_line_color='black', line_width_min_pixels=1, stroked=False,
        ))
    return layers


def _graded_colors(n_bins):
    """``n_bins`` RGB int-255 tuples grading blue (low) -> grey -> orange (high).

    For the bins *between* the hardcoded outlier tails. Blue and orange sit
    on the blue-yellow axis, which a protanope (red-blind) reader sees
    normally -- unlike a red-green gradient, which runs along exactly the
    axis protanopia collapses. Every shade here is also lighter than
    `LOW_OUTLIER_COLOR`/`HIGH_OUTLIER_COLOR`, so a graded bin can't be
    mistaken for an outlier tail.
    """
    if n_bins <= 1:
        return [(189, 189, 189)]
    blue = np.array([49, 130, 189])
    grey = np.array([224, 224, 224])
    orange = np.array([230, 145, 40])
    half = (n_bins - 1) / 2.0
    colors = []
    for i in range(n_bins):
        t = (i - half) / half              # -1 (low) .. 0 (middle) .. 1 (high)
        mix = blue if t < 0 else orange
        c = grey + (mix - grey) * abs(t)
        colors.append(tuple(int(round(v)) for v in c))
    return colors


def _outlier_masks(values, low_threshold, high_threshold):
    """Boolean (low, middle, high) masks splitting `values` on the two tails."""
    values = np.asarray(values, dtype=float)
    low = values < low_threshold
    high = values > high_threshold
    return low, ~(low | high), high


def ratio_map(df=None, column=None, groupby=None, n_bins=5,
              lat_col='latitude', long_col='longitude',
              height=DEFAULT_MAP_HEIGHT, *,
              dot_size=DEFAULT_DOT_SIZE,
              opacity=DEFAULT_DOT_OPACITY,
              colour_code_by='iqr',
              iqr_bands=DEFAULT_IQR_BANDS,
              iqr_log_transform=True,
              iqr_min_cod=0,
              mad_bands=DEFAULT_MAD_BANDS,
              standard_bands=DEFAULT_STANDARD_BANDS):
    """
    Display ratios on an interactive map, coloured by how each point's ASR
    compares to the sample's median ASR.

    Five colours, in every scheme: grey for the central mass, light orange /
    light blue for the ring that has drifted off the median, and red /
    dark blue for the far tails. `colour_code_by` chooses what "drifted"
    means -- a dispersion multiple of this sample (``'iqr'``, ``'mad'``) or a
    fixed share of the median (``'standard'``, the one scheme comparable
    between jurisdictions).

    Blue/grey/orange rather than the more common red-yellow-green avoids the
    one axis protanopia (red-blindness) actually collapses; blue and orange
    are read normally. The two extremes are also the only dark fills, so they
    separate from the other three by lightness as well as by hue.

    Grey is always drawn underneath the four coloured bands (see
    `BACKGROUND_BANDS`), in its own layer. Grey is usually most of the
    sample, so left in draw order it buries the handful of dots the map
    exists to show -- one ordinary sale landing over a red neighbourhood
    hides the only finding on screen. Under ``colour_code_by='quantile'``,
    which has no single neutral colour, the graded middle goes underneath
    and the two fixed tails on top instead.

    Parameters
    ----------
    df : DataFrame
        Sales dataframe for analysis. Must contain ``lat_col``/``long_col``.
    column : str
        Ratio column to plot.
    groupby : str, optional
        Neighbourhood column to group ratios by; each group is coloured by
        its median, and the colour coding is applied to those group medians
        rather than to the individual sales. When omitted, every row is its
        own group, so every sale is coloured by its own ratio.
    n_bins : int, default 5
        Total colour categories for ``colour_code_by='quantile'`` only,
        including its two fixed tails. Ignored by the other schemes, which
        are five-band by construction. Down from 7: the extra pair of
        shades only ever split the central mass finer than anyone reads it.
    lat_col, long_col : str, default 'latitude', 'longitude'
        Coordinate columns in ``df``.
    height : int or str, default ``DEFAULT_MAP_HEIGHT`` (700)
        Map height, in CSS pixels when a number is given, or any CSS
        length (``'80vh'``, ``'900px'``). The map is always as wide as the
        cell, so this is what sets the aspect ratio; lonboard's own 500px
        default letterboxes a full-width notebook.
    dot_size : float, default ``DEFAULT_DOT_SIZE`` (40)
        ``radius_scale`` for the dots. This is the value the maps always
        used, so leaving it alone keeps them as they were; raise it for a
        sparse rural roll where the dots vanish at the fitted zoom, lower
        it for a dense urban one where they merge into a single blob.
    opacity : float, default ``DEFAULT_DOT_OPACITY`` (0.4)
        Fill opacity of the dots, from 0 (invisible) to 1 (solid), given
        either as a fraction or as a percentage (``40``). Semi-transparent
        by default so overlapping dots read as overlapping: a patch of two
        hundred sales is visibly darker than a patch of five, and the
        basemap stays legible underneath. Pass ``1`` for the solid dots the
        maps drew before this was an argument.
    colour_code_by : {'iqr', 'mad', 'standard', 'quantile'}, default 'iqr'
        How the five colours are assigned.

        * ``'iqr'`` -- the package's own :func:`ratio_study.helpers.iqr_filter`
          run at both `iqr_bands` cutoffs. Grey inside the inner fence,
          light orange / light blue between the fences, red / dark blue
          outside the outer one. A point the study would trim at 3 IQRs is
          exactly a point this paints as a far outlier.
        * ``'mad'`` -- the same shape, measured in median absolute
          deviations (`mad_bands`) instead. More resistant than the IQR to a
          sample whose quartiles are themselves pulled about.
        * ``'standard'`` -- fixed bands as a share of the median
          (`standard_bands`), so "10 % off the level of assessment" means
          the same thing on every map regardless of this sample's spread.
        * ``'quantile'`` -- the previous scheme: fixed 80 %/120 % tails with
          `n_bins` - 2 adaptive quantile bins between them. Kept for maps
          already built against it.
    iqr_bands : tuple of float, default ``DEFAULT_IQR_BANDS`` (1.5, 3.0)
        ``(inner, outer)`` IQR multiples for ``colour_code_by='iqr'`` --
        Tukey's own "outlier" and "far outlier" fences.
    iqr_log_transform : bool, default True
        Take the IQR fences on ``log(value)``, as ``iqr_filter`` does by
        default: ratios are right-skewed, and raw-scale quartiles push the
        upper fence out far enough to miss genuine high outliers.
    iqr_min_cod : float, default 0
        ``min_cod`` gate passed to ``iqr_filter``. Defaults to 0 rather than
        the filter's own 0.06, because that gate exists to stop a tight
        sample being stripped of legitimate sales and colouring a point
        removes nothing -- left at 0.06, a low-dispersion sample would map
        entirely grey. Set it to 0.06 to match the study's trimming exactly.
    mad_bands : tuple of float, default ``DEFAULT_MAD_BANDS`` (3.5, 5.0)
        ``(inner, outer)`` MAD multiples for ``colour_code_by='mad'``. Wider
        than `iqr_bands` because the MAD is a much smaller unit than the
        IQR: on a roughly normal sample the IQR is about 2.7 MADs, so 1.5
        MADs would band a fifth of an ordinary roll as drifting.
    standard_bands : tuple of float, default ``DEFAULT_STANDARD_BANDS`` (0.10, 0.25)
        ``(inner, outer)`` proportions of the median for
        ``colour_code_by='standard'`` -- so by default grey within 10 % of
        the median, light orange / light blue from 10 % to 25 % out, and
        red / dark blue beyond 25 %.

    Returns
    -------
    ipywidgets.VBox
        Controls, map, legend and a status line. **Open in browser** writes a
        standalone HTML map and opens it in a real browser tab, which is the
        way to get a full-window map; **Taller** / **Shorter** step the height
        by 200px in place.

    Examples
    --------
    >>> ratio_map(df=sales, column='asr', groupby='neighbourhood')     # doctest: +SKIP
    >>> ratio_map(df=sales, column='asr', colour_code_by='standard',
    ...           dot_size=25)                                        # doctest: +SKIP

    See Also
    --------
    local_moran : hot/cold-spot map, with the same controls.
    open_map_in_browser : the export the button calls, usable from a script.
    """
    gpd = _optional.require('geopandas', 'spatial', 'ratio_map')
    widgets = _optional.require('ipywidgets', 'spatial', 'ratio_map')
    lonboard = _optional.require('lonboard', 'spatial', 'ratio_map')
    lonboard_cmap = _optional.require('lonboard.colormap', 'spatial', 'ratio_map')

    try:
        if colour_code_by not in COLOUR_CODE_SCHEMES:
            raise ValueError(
                f'colour_code_by={colour_code_by!r} is not one of '
                f'{COLOUR_CODE_SCHEMES}.')
        opacity = _as_opacity(opacity)
        if column not in df.columns or (groupby is not None and groupby not in df.columns):
            print('Please enter valid column name')
            return

        gdf = gpd.GeoDataFrame(
            df, crs='EPSG:4326',
            geometry=gpd.points_from_xy(x=df[long_col], y=df[lat_col], crs='EPSG:4326'),
        )

        if groupby:
            group_label = groupby
            med_ratio = gdf[[groupby, column, 'geometry']].dissolve(by=[groupby], aggfunc='median')
            med_ratio = med_ratio.drop(columns='geometry').rename(columns={column: 'median'})
            mean_ratio = gdf[[groupby, column, 'geometry']].dissolve(by=[groupby], aggfunc='mean')
            mean_ratio = mean_ratio.drop(columns='geometry').rename(columns={column: 'mean'})
            min_ratio = gdf[[groupby, column, 'geometry']].dissolve(by=[groupby], aggfunc='min')
            min_ratio = min_ratio.drop(columns='geometry').rename(columns={column: 'min'})
            max_ratio = gdf[[groupby, column, 'geometry']].dissolve(by=[groupby], aggfunc='max')
            max_ratio = max_ratio.drop(columns='geometry').rename(columns={column: 'max'})
            sale_count = gdf[[groupby, column, 'geometry']].dissolve(by=[groupby], aggfunc='count')
            sale_count = sale_count.rename(columns={column: 'sale_count'})

            points = pd.concat([med_ratio, mean_ratio, min_ratio, max_ratio, sale_count], axis=1)
            points = points.reset_index(drop=False)
            points_gdf = gpd.GeoDataFrame(points, crs='EPSG:4326', geometry='geometry')
        else:
            # No grouping column: every sale is its own group, so every
            # point gets its own colour instead of a neighbourhood blend.
            group_label = 'sale'
            points = pd.DataFrame({'sale': gdf.index.astype(str)})
            for stat in ('median', 'mean', 'min', 'max'):
                points[stat] = gdf[column].to_numpy()
            points['sale_count'] = 1
            points_gdf = gpd.GeoDataFrame(points, crs='EPSG:4326', geometry=gdf.geometry.to_numpy())

        if colour_code_by == 'quantile':
            layers, legend_items = _quantile_layers(
                points_gdf, column, n_bins, dot_size, opacity, widgets,
                lonboard, lonboard_cmap)
        else:
            values = points_gdf['median'].to_numpy(dtype=float)
            codes, descriptions = _colour_code(
                values, colour_code_by, column,
                iqr_bands=iqr_bands, iqr_log_transform=iqr_log_transform,
                iqr_min_cod=iqr_min_cod, mad_bands=mad_bands,
                standard_bands=standard_bands)

            # The band name rides along as a column so the tooltip says which
            # band a dot is in -- the legend alone means matching a shade by
            # eye, which is exactly what the colour scheme is trying to avoid.
            band_gdf = points_gdf.copy()
            band_gdf['colour_band'] = [descriptions.get(code, 'no ratio')
                                       for code in codes]
            layers = _layered_scatter(
                band_gdf, codes, np.isin(codes, BACKGROUND_BANDS),
                BAND_COLOURS, dot_size, opacity, lonboard, lonboard_cmap)
            legend_items = _band_legend(codes, values, column, descriptions,
                                        widgets)

        m = lonboard.Map(layers)
        m.basemap_style = 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json'
        m.set_view_state(**_fit_view(df[long_col], df[lat_col]))

        box = widgets.HBox(legend_items)
        return _map_panel(m, box, height, widgets,
                          title=f'Ratio map - {column} by {group_label}')

    except Exception as ex:
        raise ex


def _quantile_layers(points_gdf, column, n_bins, dot_size, opacity, widgets,
                     lonboard, lonboard_cmap):
    """
    Layers and legend for the legacy ``colour_code_by='quantile'`` scheme.

    Fixed 80 %/120 % tails in dark blue and red, with ``n_bins - 2`` adaptive
    quantile bins of the subrange between them graded blue -> grey -> orange.
    The tails used to be drawn as triangles in a separate ``PolygonLayer``;
    they are circles in the same scatter layer as everything else now, so a
    dot's size means the same thing everywhere on the map.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        One row per mapped point, carrying a ``'median'`` column.
    column : str
        Name of the mapped column, for the legend wording.
    n_bins : int
        Total colour categories, including the two tails.
    dot_size : float
        ``radius_scale`` for the dots.
    opacity : float
        Fill opacity of the dots, 0 to 1.
    widgets, lonboard, lonboard_cmap : module
        Resolved optional dependencies, passed in so this helper does not
        repeat the lookup.

    Returns
    -------
    tuple
        ``(layers, legend_items)``.
    """
    overall_median = float(np.median(points_gdf['median']))
    low_threshold = LOW_OUTLIER_FRACTION * overall_median
    high_threshold = HIGH_OUTLIER_FRACTION * overall_median
    low, middle, high = _outlier_masks(points_gdf['median'], low_threshold,
                                       high_threshold)

    legend_items = []
    layout = widgets.Layout(width='auto')

    # `category` is shared across the tails and the graded bins so the whole
    # map can go in one scatter layer: 0 for the low tail, 1..actual_bins for
    # the graded middle, and actual_bins + 1 for the high tail.
    categories = np.zeros(len(points_gdf), dtype=int)
    color_dict = {}
    actual_bins = 0
    edges = np.array([])

    if middle.any():
        edges = _quantile_bin_edges(points_gdf.loc[middle, 'median'],
                                    n_bins=max(n_bins - 2, 1))
        # A single edge means every value in the middle is identical --
        # there is nothing to bin, so it is one bin, not zero.
        actual_bins = max(len(edges) - 1, 1)
        mid_colors = _graded_colors(actual_bins)
        color_dict.update({i + 1: mid_colors[i] for i in range(actual_bins)})
        mid_values = points_gdf['median'].to_numpy(dtype=float)
        if len(edges) > 1:
            categories[np.asarray(middle)] = [
                _bin_index(x, edges) for x in mid_values[np.asarray(middle)]]
        else:
            categories[np.asarray(middle)] = 1

    high_code = actual_bins + 1
    color_dict[0] = LOW_OUTLIER_COLOR
    color_dict[high_code] = HIGH_OUTLIER_COLOR
    categories[np.asarray(low)] = 0
    categories[np.asarray(high)] = high_code

    if low.any():
        button = widgets.Button(
            disabled=True, layout=layout,
            description=f'low outlier: median {column} < {low_threshold:.3f} '
                        f'({LOW_OUTLIER_FRACTION:.0%} of the overall median)')
        button.style.button_color = 'rgb({},{},{})'.format(*LOW_OUTLIER_COLOR)
        button.style.text_color = 'rgb(255,255,255)'
        legend_items.append(button)

    for i in range(1, actual_bins + 1):
        if len(edges) == 1:
            label = f'Median {column} = {edges[0]:.3f}'
        else:
            lower, upper = edges[i - 1], edges[i]
            if i == 1:
                label = f'Median {column} < {upper:.3f}'
            elif i == actual_bins:
                label = f'Median {column} >= {lower:.3f}'
            else:
                label = f'Median {column} in [{lower:.3f}, {upper:.3f})'
        button = widgets.Button(disabled=True, description=label, layout=layout)
        button.style.button_color = 'rgb({},{},{})'.format(*color_dict[i])
        legend_items.append(button)

    if high.any():
        button = widgets.Button(
            disabled=True, layout=layout,
            description=f'high outlier: median {column} > {high_threshold:.3f} '
                        f'({HIGH_OUTLIER_FRACTION:.0%} of the overall median)')
        button.style.button_color = 'rgb({},{},{})'.format(*HIGH_OUTLIER_COLOR)
        button.style.text_color = 'rgb(255,255,255)'
        legend_items.append(button)

    scatter_gdf = points_gdf.copy()
    scatter_gdf['category'] = categories
    # The graded middle goes underneath and the two fixed tails on top. This
    # scheme has no single neutral colour to hold back -- its middle is a
    # blue-grey-orange gradient -- so the split is by role instead: the
    # tails are the finding, and they were drawn over everything else back
    # when they were triangles in their own layer.
    layers = _layered_scatter(
        scatter_gdf, categories,
        ~np.isin(categories, (0, high_code)),
        color_dict, dot_size, opacity, lonboard, lonboard_cmap)
    return layers, legend_items


def vem_map(df=None, assessment_header=None, market_value_header=None,
            asr_col=None, lat_col='latitude', long_col='longitude',
            quantiles=None, window_size=None, conf_level=0.9, ci_method='binom',
            acceptable_inequity_level=0.1, strict=False,
            height=DEFAULT_MAP_HEIGHT, dot_size=DEFAULT_DOT_SIZE,
            opacity=DEFAULT_DOT_OPACITY):
    """
    Map each sale by whether its local (moving) median ASR is inside the band.

    The VEM answers *where along the value range* assessment level drifts;
    this answers *where on the ground* those sales are. Every sale is coloured
    by the verdict of the quantile group it belongs to:

    * **green** -- the local median sits within ``acceptable_inequity_level``
      of the jurisdiction-wide median
    * **red** -- the local median is above the band (over-assessed stretch)
    * **blue** -- the local median is below the band (under-assessed stretch)

    Because the grouping is by *value*, not by location, a clean geographic
    pattern in the colours is a real finding: it means a price band that is
    mis-assessed also happens to be concentrated somewhere, which is a
    modelling problem rather than a scattering of individual errors.

    Parameters
    ----------
    df : DataFrame
        Sales frame. Must carry ``lat_col`` / ``long_col``.
    assessment_header, market_value_header : str
        Assessed value and market value proxy. Not needed if `asr_col`
        already holds the ratio, but `market_value_header` is always
        required -- the VEM groups on it.
    asr_col : str, optional
        Existing ratio column. Computed from the two headers when omitted.
    lat_col, long_col : str, default 'latitude', 'longitude'
        Coordinate columns.
    quantiles, window_size, conf_level, ci_method, acceptable_inequity_level
        Passed straight to :func:`ratio_study.vem`; see it for the meaning.
    strict : bool, default False
        Colour by ``ci_out_of_bounds`` rather than ``out_of_bounds`` -- a
        stretch then has to have its whole local-median interval outside the
        band before it is flagged, not just the point estimate. This is the
        same test the VEM verdict itself uses, so ``strict=True`` maps
        exactly what the indicator reports.
    height : int or str, default ``DEFAULT_MAP_HEIGHT`` (700)
        Map height. See :func:`ratio_map`.
    dot_size : float, default ``DEFAULT_DOT_SIZE`` (40)
        ``radius_scale`` for the dots. See :func:`ratio_map`.
    opacity : float, default ``DEFAULT_DOT_OPACITY`` (0.4)
        Fill opacity of the dots, 0 to 1. See :func:`ratio_map`.

    Returns
    -------
    ipywidgets.VBox
        Controls, map, legend and a status line, as :func:`ratio_map`.

    See Also
    --------
    ratio_study.vem : the indicator behind the colours.
    ratio_study.vem_plot : the same verdict along the value range.
    """
    from .indicators import vem as _vem

    gpd = _optional.require('geopandas', 'spatial', 'vem_map')
    widgets = _optional.require('ipywidgets', 'spatial', 'vem_map')
    lonboard = _optional.require('lonboard', 'spatial', 'vem_map')
    lonboard_cmap = _optional.require('lonboard.colormap', 'spatial', 'vem_map')

    try:
        if market_value_header is None or market_value_header not in df.columns:
            raise ValueError('vem_map needs `market_value_header` to be a column of `df`.')
        for coord in (lat_col, long_col):
            if coord not in df.columns:
                raise ValueError(f'vem_map needs a {coord!r} column to place the sales.')

        work = df.copy()
        ratio_col = asr_col or 'asr'
        if ratio_col not in work.columns:
            if not assessment_header:
                raise ValueError(
                    'vem_map needs either `asr_col` naming an existing ratio column, '
                    'or `assessment_header` so the ratio can be computed.')
            work[ratio_col] = work[assessment_header] / work[market_value_header]

        result = _vem(work, market_value_proxy=market_value_header, asr_col=ratio_col,
                      quantiles=quantiles, window_size=window_size,
                      conf_level=conf_level, ci_method=ci_method,
                      acceptable_inequity_level=acceptable_inequity_level)

        flags = result['dataframe']
        column = 'ci_out_of_bounds' if strict else 'out_of_bounds'
        if column not in flags.columns:
            raise ValueError(
                f"the VEM produced no {column!r} column; "
                'pass `conf_level` and `acceptable_inequity_level` to get one.')

        # vem() keeps the caller's index, so the per-sale verdict joins straight
        # back on. Sales it dropped (missing ratio or value) simply do not map.
        work = work.loc[flags.index].copy()
        work['vem_flag'] = flags[column].astype(int).to_numpy()
        work['local_median'] = (flags['moving_median'] + result['asr_center']).to_numpy()
        work['vem_verdict'] = work['vem_flag'].map(
            {-1: 'below the band', 0: 'within the band', 1: 'above the band'})

        keep = [c for c in (market_value_header, assessment_header, ratio_col,
                            'local_median', 'vem_verdict', 'vem_flag') if c and c in work.columns]
        gdf = gpd.GeoDataFrame(
            work[keep], crs='EPSG:4326',
            geometry=gpd.points_from_xy(x=work[long_col], y=work[lat_col], crs='EPSG:4326'),
        )

        colours = {0: (46, 160, 67), 1: (203, 26, 0), -1: (31, 120, 180)}
        fill_color = lonboard_cmap.apply_categorical_cmap(gdf['vem_flag'], cmap=colours)

        layer = lonboard.ScatterplotLayer.from_geopandas(
            gdf, pickable=True, radius_scale=dot_size,
            opacity=_as_opacity(opacity), auto_highlight=True,
            get_fill_color=fill_color, get_line_color='black',
            line_width_min_pixels=1, stroked=False,
        )
        m = lonboard.Map([layer])
        m.basemap_style = 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json'
        m.set_view_state(**_fit_view(work[long_col], work[lat_col]))

        # Counts on the legend, so the reader sees at once whether "red" is
        # three sales or a third of the roll.
        counts = work['vem_flag'].value_counts()
        total = len(work)
        layout = widgets.Layout(width='auto')
        legend = []
        for flag, label, rgb in ((0, 'within the band', colours[0]),
                                 (1, 'above the band (over-assessed)', colours[1]),
                                 (-1, 'below the band (under-assessed)', colours[-1])):
            n = int(counts.get(flag, 0))
            button = widgets.Button(disabled=True, layout=layout,
                                    description=f'{label}: {n:,} ({n / total:.1%})')
            button.style.button_color = 'rgb({},{},{})'.format(*rgb)
            if flag != 0:
                button.style.text_color = 'rgb(255,255,255)'
            legend.append(button)

        band = f'{acceptable_inequity_level:.0%}'
        basis = 'confirmed by the interval' if strict else 'point estimate'
        return _map_panel(m, widgets.HBox(legend), height, widgets,
                          title=f'VEM map - local median vs +/-{band} band ({basis})')
    except Exception as ex:
        raise RuntimeError('error in ratio study: vem_map', ex)


MORAN_SIGNIFICANCE_LEVEL = 0.05


def _moran_message(moran_i, z_rand, p_rand, alpha=MORAN_SIGNIFICANCE_LEVEL):
    """
    Plain-language reading of a Global Moran's I.

    Magnitude and significance are reported separately, because they answer
    different questions: |I| says how strong the clustering is, the p-value
    (under the randomisation hypothesis) says whether it can be told from
    chance. On thousands of sales a tiny I can be significant, so a small I
    is never described as "no autocorrelation" when the test says otherwise.

    Parameters
    ----------
    moran_i, z_rand, p_rand : float
        Moran's I with its randomisation-hypothesis z-score and p-value.
    alpha : float, default 0.05
        Significance level the p-value is judged against; named in the text.

    Returns
    -------
    str
        The reading, ending with a recommended action.
    """
    significant = bool(p_rand < alpha)
    size = abs(moran_i)
    if size < 0.2:
        magnitude = 'negligible'
    elif size < 0.5:
        magnitude = 'low'
    elif size < 0.8:
        magnitude = 'moderate'
    else:
        magnitude = 'high'
    direction = '' if magnitude == 'negligible' else ('positive ' if moran_i > 0 else 'negative ')
    verdict = ('statistically significant' if significant else 'not statistically significant')
    level = f'{alpha:.0%}'

    if magnitude in ('moderate', 'high'):
        action = 'Action required: investigate the spatial pattern before relying on the study.'
    elif significant and magnitude == 'low' and moran_i > 0:
        action = 'Worth monitoring: nearby sales tend to share similar ratios.'
    else:
        action = 'No action required.'
    return (f"{action} A Global Moran's I of {round(moran_i, 2)} indicates {magnitude} "
            f"{direction}spatial autocorrelation. Under the randomisation hypothesis the Z value "
            f"({round(z_rand, 2)}) and P value ({round(p_rand, 4)}) are {verdict} at the {level} level.")


def global_moran(df=None, ratio=None, lat_col='latitude', long_col='longitude'):
    """
    Calculate the Global Moran statistic; returns statistics and a
    graphical plot of Global Moran's I.

    Parameters
    ----------
    df : DataFrame
        Sales DataFrame to plot. Must contain ``lat_col``/``long_col``.
    ratio : str
        Ratio to test for global spatial auto-correlation.
    lat_col, long_col : str, default 'latitude', 'longitude'
        Coordinate columns in ``df``.
    """
    scipy_stats = _optional.require('scipy.stats', 'spatial', 'global_moran')
    gpd = _optional.require('geopandas', 'spatial', 'global_moran')
    weights = _optional.require('libpysal.weights', 'spatial', 'global_moran')
    esda = _optional.require('esda', 'spatial', 'global_moran')
    plot_moran = _optional.require('splot.esda', 'spatial', 'global_moran').plot_moran

    try:
        if ratio in df.columns:

            gdf = gpd.GeoDataFrame(
                df, crs='EPSG:4326',
                geometry=gpd.points_from_xy(x=df[long_col], y=df[lat_col], crs='EPSG:4326'),
            )

            # ------------------ KNN WEIGHTS ----------------------------------
            k = _cube_root(len(gdf.index))
            kW = weights.KNN.from_dataframe(gdf, k=k, p=2)
            kW.transform = 'r'

            # ------------------ Global Moran ----------------------------------
            moran_stat = esda.Moran(gdf[ratio], kW, transformation='r')

            moran_message = _moran_message(moran_stat.I, moran_stat.z_rand, moran_stat.p_rand)

            plot_moran(moran_stat, zstandard=True, figsize=(10, 4))

            print(f'{moran_message} \n\nI: {moran_stat.I} \nEI: {moran_stat.EI} \np_norm: {moran_stat.p_norm} \nz_norm: {moran_stat.z_norm} \nVI_norm: {moran_stat.VI_norm} \nseI_norm: {moran_stat.seI_norm} \np_rand: {moran_stat.p_rand} \nz_rand: {moran_stat.z_rand} \nVI_rand: {moran_stat.VI_rand} \nseI_rand: {moran_stat.seI_rand} \nEI_sim: {moran_stat.EI_sim} \np_sim: {moran_stat.p_sim} \nz_sim: {moran_stat.z_sim} \np_z_sim {moran_stat.p_z_sim} \npermutations: {moran_stat.permutations}\np_z_norm (one-sided): {scipy_stats.norm.sf(abs(moran_stat.z_norm))}')
            plt.show()
            return None
        else:
            print('Please enter a valid column name.')
            return None
    except Exception as ex:
        raise ex


def local_moran(df=None, ratio=None, lat_col='latitude', long_col='longitude',
                height=DEFAULT_MAP_HEIGHT, dot_size=DEFAULT_DOT_SIZE,
                opacity=DEFAULT_DOT_OPACITY):
    """
    Calculate the local Moran/Getis-Ord statistic and visually plot
    cold and hot spots for the sales file.

    Parameters
    ----------
    df : DataFrame
        Sales DataFrame to plot. Must contain ``ratio``, ``lat_col`` and
        ``long_col``. Any of ``roll_num``, ``inst_num``, ``saledate``,
        ``saletype``, ``sale_amt``, ``r_enbhd_condo`` or ``r_hnbhd_condo``
        that are present are carried through to the map tooltip.
    ratio : str
        Ratio to test for local spatial auto-correlation.
    lat_col, long_col : str, default 'latitude', 'longitude'
        Coordinate columns in ``df``.
    height : int or str, default ``DEFAULT_MAP_HEIGHT`` (700)
        Map height, in CSS pixels when a number is given, or any CSS
        length (``'80vh'``, ``'900px'``). See :func:`ratio_map`.
    dot_size : float, default ``DEFAULT_DOT_SIZE`` (40)
        ``radius_scale`` for the dots. See :func:`ratio_map`.
    opacity : float, default ``DEFAULT_DOT_OPACITY`` (0.4)
        Fill opacity of the dots, 0 to 1. See :func:`ratio_map`.

    Returns
    -------
    ipywidgets.VBox
        Controls, map, legend and a status line, with **Open in browser** and
        **Taller** / **Shorter**. See :func:`ratio_map`.
    """
    gpd = _optional.require('geopandas', 'spatial', 'local_moran')
    weights = _optional.require('libpysal.weights', 'spatial', 'local_moran')
    G_Local = _optional.require('esda.getisord', 'spatial', 'local_moran').G_Local
    widgets = _optional.require('ipywidgets', 'spatial', 'local_moran')
    lonboard = _optional.require('lonboard', 'spatial', 'local_moran')
    lonboard_cmap = _optional.require('lonboard.colormap', 'spatial', 'local_moran')

    try:
        if ratio in df.columns:
            gdf = gpd.GeoDataFrame(
                df, crs='EPSG:4326',
                geometry=gpd.points_from_xy(x=df[long_col], y=df[lat_col], crs='EPSG:4326'),
            )

            # # ------------------ LOCAL G STAT ----------------------------------
            k = _cube_root(len(gdf.index))

            GL_kW = weights.KNN.from_dataframe(gdf, k=k, p=1)
            GL_kW.transform = 'b'

            gL = G_Local(np.array(gdf[f'{ratio}'], dtype=np.float64), GL_kW, star=True, transform='B')
            # `z_sim` as ``G_Local`` reports it: positive means the point
            # sits in a cluster of high values, which for an ASR is a
            # cluster of over-assessment -- a hotspot. This used to be
            # multiplied by -1, which inverted that, so a planted cluster of
            # high ratios came back labelled "coldspot" and vice versa.
            gdf['Z_Value'] = gL.z_sim
            # `p_z_sim` is one-tailed, so doubling it gives the two-tailed
            # p-value the bands are written against. It cannot exceed 1:
            # a one-tailed p tops out at 0.5.
            gdf['P_Value'] = gL.p_z_sim * 2

            gdf['category'] = gdf.apply(lambda x: _moran_class(x.Z_Value, x.P_Value), axis=1)
            gdf['flag'] = gdf['category'].map(
                {code: label for code, (label, _) in MORAN_CLASSES.items()})

            cat_color_map = {code: rgb for code, (_, rgb) in MORAN_CLASSES.items()}
            fill_color = lonboard_cmap.apply_categorical_cmap(gdf['category'], cmap=cat_color_map)

            keep_cols = [c for c in ['roll_num', 'inst_num', 'saledate', 'saletype', 'sale_amt',
                                      'r_enbhd_condo', 'r_hnbhd_condo'] if c in gdf.columns]
            gdf = gdf[keep_cols + ['Z_Value', 'P_Value', 'category', 'flag', 'geometry']]

            # No `get_line_color` / `line_width_min_pixels`: `stroked=False`
            # means no outline is drawn, so both were inert.
            layer = lonboard.ScatterplotLayer.from_geopandas(
                gdf, pickable=True, radius_scale=dot_size,
                opacity=_as_opacity(opacity), auto_highlight=True,
                get_fill_color=fill_color, stroked=False,
            )

            m = lonboard.Map([layer])
            m.basemap_style = 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json'
            m.set_view_state(**_fit_view(df[long_col], df[lat_col]))

            layout = widgets.Layout(width='auto')

            # Straight off `MORAN_CLASSES`, in code order (coldest to
            # hottest, with the unclassified bucket first), so a swatch can
            # never disagree with the dots it is labelling.
            items = []
            for code, (label, rgb) in MORAN_CLASSES.items():
                button = widgets.Button(disabled=True, description=label,
                                        layout=layout)
                _style_legend_button(button, rgb)
                items.append(button)

            box = widgets.HBox(items)

            return _map_panel(m, box, height, widgets,
                              title=f'Hot and cold spots - {ratio}')
        else:
            print('Please pick valid column name. Columns roll_num, inst_num, saledate, saletype, sale_amt, '
                  'r_enbhd_condo, r_hnbhd_condo required in dataframe')
            return None
    except Exception as ex:
        raise ex


# Usage examples:
# from ratio_study import spatial
# from ratio_study.indicators import moran_i
#
# sales = ratio_study.load_sample_dataset()
# sales['asr'] = sales['assessment_value_new'] / sales['time_adjusted_sale_price']
#
# spatial.ratio_map(df=sales, column='asr', groupby='neighbourhood')
# spatial.local_moran(df=sales, ratio='asr')
# spatial.global_moran(df=sales, ratio='asr')
# moran_i(sales, 'asr')
