# Spatial

Ratio maps, VEM maps and Moran’s I.

*Module:* `ratio_study.spatial`

## Contents

- [`global_moran`](#global-moran) — Calculate the Global Moran statistic; returns statistics and a
- [`local_moran`](#local-moran) — Calculate the local Moran/Getis-Ord statistic and visually plot
- [`open_map_in_browser`](#open-map-in-browser) — Write a map to a standalone HTML file and open it in the default browser.
- [`ratio_map`](#ratio-map) — Display ratios on an interactive map, coloured by how each point's ASR
- [`vem_map`](#vem-map) — Map each sale by whether its local (moving) median ASR is inside the band.

---

## `global_moran`

```python
global_moran(df=None, ratio=None, lat_col='latitude', long_col='longitude')
```

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

---

## `local_moran`

```python
local_moran(df=None, ratio=None, lat_col='latitude', long_col='longitude', height=700,
            dot_size=40, opacity=0.4)
```

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

---

## `open_map_in_browser`

```python
open_map_in_browser(panel_or_map, filename=None, title='Ratio study map',
                    open_browser=True)
```

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

---

## `ratio_map`

```python
ratio_map(df=None, column=None, groupby=None, n_bins=5, lat_col='latitude',
          long_col='longitude', height=700, *, dot_size=40, opacity=0.4,
          colour_code_by='iqr', iqr_bands=(1.5, 3.0), iqr_log_transform=True,
          iqr_min_cod=0, mad_bands=(3.5, 5.0), standard_bands=(0.1, 0.25))
```

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

---

## `vem_map`

```python
vem_map(df=None, assessment_header=None, market_value_header=None, asr_col=None,
        lat_col='latitude', long_col='longitude', quantiles=None, window_size=None,
        conf_level=0.9, ci_method='binom', acceptable_inequity_level=0.1, strict=False,
        height=700, dot_size=40, opacity=0.4)
```

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
