"""Tests for spatial diagnostics: the adaptive ratio_map colour scale and
the moran_i indicator.

``ratio_study.spatial`` only imports numpy/pandas/matplotlib at module
level -- geopandas/libpysal/esda/lonboard/ipywidgets are required lazily,
inside each function, via ``_optional.require``. So importing the module
never needs the ``spatial`` extra; only calling its functions does.
"""
import numpy as np
import pandas as pd
import pytest

from ratio_study import spatial
from ratio_study import indicators as I

libpysal = pytest.importorskip("libpysal")
esda = pytest.importorskip("esda")


# --------------------------------------------------------------------------
# Distribution-adaptive colour scale
# --------------------------------------------------------------------------

def test_quantile_bin_edges_span_the_actual_distribution():
    values = np.linspace(0.7, 1.3, 100)
    edges = spatial._quantile_bin_edges(values, n_bins=7)

    assert edges[0] == pytest.approx(values.min())
    assert edges[-1] == pytest.approx(values.max())
    assert len(edges) == 8


def test_quantile_bin_edges_dedupe_when_distribution_is_degenerate():
    # All-identical values collapse every quantile to the same edge.
    edges = spatial._quantile_bin_edges(np.ones(20), n_bins=7)
    assert len(edges) == 1


def test_bin_index_is_monotonic_across_edges():
    edges = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    indices = [spatial._bin_index(x, edges) for x in [0.5, 0.85, 0.95, 1.05, 1.15, 5.0]]
    assert indices == sorted(indices)
    assert indices[0] == 1
    assert indices[-1] == len(edges) - 1


def test_fit_view_centres_on_a_tight_cluster_and_zooms_in():
    lon = [-75.00, -75.01, -75.02, -75.03]
    lat = [40.00, 40.01, 40.02, 40.03]
    view = spatial._fit_view(lon, lat)
    assert view['zoom'] > 8                      # much tighter than the old fixed zoom=8
    assert -75.03 <= view['longitude'] <= -75.00
    assert 40.00 <= view['latitude'] <= 40.03


def test_fit_view_is_not_wrecked_by_a_single_bad_coordinate():
    # (0, 0) is the classic failed-geocode value. lonboard's own exact
    # min/max fit would zoom this out to the whole globe for one bad row.
    lon = [-75.00, -75.01, -75.02, -75.03, -75.02, -75.01, -75.00, -75.015,
          -75.005, -75.025, 0.0]
    lat = [40.00, 40.01, 40.02, 40.03, 40.02, 40.01, 40.00, 40.015,
          40.005, 40.025, 0.0]
    view = spatial._fit_view(lon, lat)
    assert view['zoom'] > 8
    assert -75.03 <= view['longitude'] <= -75.00


def test_fit_view_drops_non_finite_coordinates():
    view_clean = spatial._fit_view([-75.0, -75.01], [40.0, 40.01])
    view_with_nan = spatial._fit_view([-75.0, -75.01, np.nan], [40.0, 40.01, np.nan])
    assert view_clean == view_with_nan


def test_graded_colors_returns_the_requested_count_and_distinct_ends():
    colors = spatial._graded_colors(5)
    assert len(colors) == 5
    assert colors[0] != colors[-1]
    for rgb in colors:
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)


def test_graded_colors_run_blue_to_grey_to_orange():
    # No green, and no red -- those are reserved for the hardcoded outlier
    # tails, and a red-green gradient is exactly the axis protanopia erases.
    low, mid, high = spatial._graded_colors(3)
    assert low[2] > low[0]           # blue channel leads on the low end
    assert high[0] > high[2]         # red/orange channel leads on the high end
    assert mid == (224, 224, 224)    # grey in the middle


def test_graded_colors_handles_a_single_bin():
    assert spatial._graded_colors(1) == [(189, 189, 189)]


def test_graded_colors_are_lighter_than_the_hardcoded_outlier_colours():
    # So a graded bin is never mistaken for a hardcoded outlier tail.
    for rgb in spatial._graded_colors(5):
        assert sum(rgb) > sum(spatial.LOW_OUTLIER_COLOR)
        assert sum(rgb) > sum(spatial.HIGH_OUTLIER_COLOR)


def test_outlier_masks_split_low_middle_and_high():
    low, middle, high = spatial._outlier_masks([0.5, 0.95, 1.0, 1.05, 2.0], 0.9, 1.1)
    assert low.tolist() == [True, False, False, False, False]
    assert high.tolist() == [False, False, False, False, True]
    assert middle.tolist() == [False, True, True, True, False]


# --------------------------------------------------------------------------
# Five-band colour coding
# --------------------------------------------------------------------------
# A tight central mass with four planted outliers -- two mild, two far --
# so every scheme has all five bands to fill.

def _banded_values():
    rng = np.random.default_rng(7)
    return np.concatenate([rng.normal(1.0, 0.05, 300),
                           [1.18, 1.75, 0.82, 0.35]])


def _codes(scheme, values, **kwargs):
    options = {'iqr_bands': spatial.DEFAULT_IQR_BANDS,
               'iqr_log_transform': True,
               'iqr_min_cod': 0,
               'mad_bands': spatial.DEFAULT_MAD_BANDS,
               'standard_bands': spatial.DEFAULT_STANDARD_BANDS}
    options.update(kwargs)
    codes, _ = spatial._colour_code(values, scheme, 'asr', **options)
    return codes


@pytest.mark.parametrize('scheme', ['iqr', 'mad', 'standard'])
def test_every_scheme_puts_the_planted_outliers_in_the_right_tails(scheme):
    values = _banded_values()
    codes = _codes(scheme, values)
    # The last four rows are, in order: mildly high, far high, mildly low,
    # far low. Every scheme has to agree on the side, and on far being
    # further out than mild.
    mild_high, far_high, mild_low, far_low = codes[-4:]
    assert far_high == spatial.BAND_VERY_HIGH
    assert far_low == spatial.BAND_VERY_LOW
    assert mild_high in (spatial.BAND_HIGH, spatial.BAND_VERY_HIGH)
    assert mild_low in (spatial.BAND_LOW, spatial.BAND_VERY_LOW)
    # And the bulk stays grey.
    assert (codes[:300] == spatial.BAND_CENTRAL).mean() > 0.6


@pytest.mark.parametrize('scheme', ['iqr', 'mad', 'standard'])
def test_no_scheme_emits_more_than_five_bands(scheme):
    codes = _codes(scheme, _banded_values())
    assert set(np.unique(codes)) <= {spatial.BAND_VERY_LOW, spatial.BAND_LOW,
                                     spatial.BAND_CENTRAL, spatial.BAND_HIGH,
                                     spatial.BAND_VERY_HIGH}


@pytest.mark.parametrize('scheme', ['iqr', 'mad', 'standard'])
def test_a_missing_value_gets_its_own_band_rather_than_a_colour(scheme):
    values = np.concatenate([_banded_values(), [np.nan]])
    codes = _codes(scheme, values)
    assert codes[-1] == spatial.BAND_MISSING


def test_standard_bands_are_a_fixed_share_of_the_median():
    # Exactly the spec: grey inside 10%, light orange/blue out to 25%,
    # red/dark blue past it. Median of this sample is 1.0.
    values = np.array([1.0, 1.05, 1.15, 1.30, 0.95, 0.85, 0.70, 1.0, 1.0])
    codes = _codes('standard', values)
    assert codes.tolist() == [
        spatial.BAND_CENTRAL,     # 1.00
        spatial.BAND_CENTRAL,     # +5%
        spatial.BAND_HIGH,        # +15%
        spatial.BAND_VERY_HIGH,   # +30%
        spatial.BAND_CENTRAL,     # -5%
        spatial.BAND_LOW,         # -15%
        spatial.BAND_VERY_LOW,    # -30%
        spatial.BAND_CENTRAL,
        spatial.BAND_CENTRAL,
    ]


def test_standard_scheme_refuses_a_non_positive_median():
    with pytest.raises(ValueError, match='share of the median'):
        _codes('standard', np.array([-1.0, 0.0, 1.0]))


def test_mad_bands_count_median_absolute_deviations():
    # Explicit cutoffs, not `DEFAULT_MAD_BANDS`: this pins the arithmetic --
    # deviations measured in MADs -- and must not fail when the default is
    # retuned. Median 1.0, MAD 0.1, so (1.5, 3.0) lands at +/-0.15, +/-0.30.
    values = np.array([1.0, 0.9, 1.1, 0.9, 1.1, 1.2, 1.4, 0.8, 0.6])
    codes = _codes('mad', values, mad_bands=(1.5, 3.0))
    assert codes[values == 1.2].tolist() == [spatial.BAND_HIGH]
    assert codes[values == 1.4].tolist() == [spatial.BAND_VERY_HIGH]
    assert codes[values == 0.8].tolist() == [spatial.BAND_LOW]
    assert codes[values == 0.6].tolist() == [spatial.BAND_VERY_LOW]


def test_a_sample_with_no_spread_is_entirely_central():
    for scheme in ('iqr', 'mad', 'standard'):
        codes = _codes(scheme, np.ones(20))
        assert (codes == spatial.BAND_CENTRAL).all(), scheme


def test_iqr_scheme_agrees_with_the_packages_own_filter():
    from ratio_study.helpers import iqr_filter

    values = _banded_values()
    codes = _codes('iqr', values)
    flagged_at_3 = np.asarray(iqr_filter(values, iqr_mult=3.0, min_cod=0)) == 1
    far = np.isin(codes, [spatial.BAND_VERY_LOW, spatial.BAND_VERY_HIGH])
    assert far.tolist() == flagged_at_3.tolist()


def test_iqr_min_cod_defaults_to_zero_so_a_tight_sample_still_bands():
    # iqr_filter's own 0.06 gate would flag nothing here and paint the whole
    # map grey. Colouring a point removes nothing, so the map does not want
    # that gate -- but it stays available.
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(1.0, 0.01, 200), [1.09, 0.91]])
    assert set(np.unique(_codes('iqr', values))) != {spatial.BAND_CENTRAL}
    gated = _codes('iqr', values, iqr_min_cod=0.06)
    assert set(np.unique(gated)) == {spatial.BAND_CENTRAL}


@pytest.mark.parametrize('bands, name', [
    ((3.0, 1.5), 'iqr_bands'),
    ((0.0, 2.0), 'iqr_bands'),
    ((1.0, 1.0), 'iqr_bands'),
])
def test_band_cutoffs_reject_a_bad_pair(bands, name):
    with pytest.raises(ValueError, match=name):
        spatial._band_cutoffs(bands, name)


def test_band_legend_reports_counts_and_the_observed_range():
    widgets = pytest.importorskip('ipywidgets')
    values = np.array([1.0, 1.0, 1.30, 0.70])
    codes = _codes('standard', values)
    items = spatial._band_legend(codes, values, 'asr',
                                 spatial._band_descriptions(
                                     'standard', spatial.DEFAULT_STANDARD_BANDS,
                                     'asr'),
                                 widgets)
    text = ' | '.join(b.description for b in items)
    assert '2 (50.0%)' in text                    # the central pair
    assert '[1.300-1.300]' in text                # the high tail's range
    # Only the bands present, and never more than five.
    assert 1 <= len(items) <= 5


# --------------------------------------------------------------------------
# moran_i
# --------------------------------------------------------------------------

def _grid_df(n_side=6):
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    lon = xs.ravel().astype(float)
    lat = ys.ravel().astype(float)
    # A checkerboard pattern is strongly negatively autocorrelated;
    # a smooth gradient is strongly positively autocorrelated.
    checkerboard = ((xs + ys) % 2).ravel().astype(float)
    gradient = (xs + ys).ravel().astype(float)
    return pd.DataFrame({
        'latitude': lat,
        'longitude': lon,
        'checkerboard': checkerboard,
        'gradient': gradient,
    })


def test_moran_i_requires_lat_and_long_columns():
    df = pd.DataFrame({'asr': [1.0, 1.1, 0.9], 'longitude': [0, 1, 2]})
    with pytest.raises(ValueError, match='latitude'):
        I.moran_i(df, 'asr')


def test_moran_i_returns_expected_keys():
    df = _grid_df()
    result = I.moran_i(df, 'gradient', k=4)
    for key in ('I', 'EI', 'p_norm', 'z_norm', 'p_rand', 'z_rand', 'p_sim', 'z_sim', 'n'):
        assert key in result
    assert result['n'] == len(df)


def test_moran_i_detects_positive_autocorrelation_in_a_smooth_gradient():
    df = _grid_df()
    result = I.moran_i(df, 'gradient', k=4, permutations=199)
    assert result['I'] > 0.5


def test_moran_i_detects_negative_autocorrelation_in_a_checkerboard():
    df = _grid_df()
    result = I.moran_i(df, 'checkerboard', k=4, permutations=199)
    assert result['I'] < 0


def test_moran_i_respects_custom_lat_long_column_names():
    df = _grid_df().rename(columns={'latitude': 'y', 'longitude': 'x'})
    result = I.moran_i(df, 'gradient', lat_col='y', long_col='x', k=4, permutations=199)
    assert result['I'] > 0.5


# --------------------------------------------------------------------------
# Map panel: sizing, and getting out of the notebook
# --------------------------------------------------------------------------
# lonboard defaults a Map to 500px, which against a full-width notebook cell
# is a letterbox strip, and the notebook's own "maximise output" control
# cannot grow it because lonboard writes the height as an inline style.
#
# An earlier attempt added a Fullscreen API button. It was removed: embedded
# notebook webviews reject requestFullscreen() by Permissions Policy, so the
# button did nothing in exactly the front ends that needed it. The controls
# are now plain ipywidgets buttons on Python callbacks, plus an export that
# opens the map in a real browser where full screen genuinely works.

lonboard = pytest.importorskip("lonboard")
widgets = pytest.importorskip("ipywidgets")


def _panel(height=spatial.DEFAULT_MAP_HEIGHT):
    return spatial._map_panel(lonboard.Map([]), widgets.HBox([]), height, widgets)


def _controls(panel):
    return {b.description: b for b in panel.children[0].children}


def test_map_panel_lays_out_controls_map_legend_and_status():
    panel = _panel()
    assert [type(c).__name__ for c in panel.children] == ['HBox', 'Map', 'HBox', 'HTML']
    assert 'rs-map' in panel.children[1]._dom_classes


def test_map_panel_offers_exactly_the_three_working_controls():
    assert list(_controls(_panel())) == ['Open in browser', 'Taller', 'Shorter']


def test_no_fullscreen_control_survives():
    # Regression guard: the API is blocked in embedded webviews, so a
    # full-screen button here is a button that does nothing.
    panel = _panel()
    assert not any('full' in d.lower() for d in _controls(panel))
    assert 'requestFullscreen' not in panel.children[3].value


def test_map_panel_applies_the_requested_height():
    assert _panel(900).children[1].height == '900px'
    assert _panel('85vh').children[1].height == '85vh'


def test_map_panel_default_is_taller_than_the_lonboard_default():
    assert spatial.DEFAULT_MAP_HEIGHT > 500
    assert _panel().children[1].height == f'{spatial.DEFAULT_MAP_HEIGHT}px'


def test_taller_and_shorter_step_the_height_trait():
    panel = _panel(700)
    m, ctl = panel.children[1], _controls(panel)
    ctl['Taller'].click()
    assert m.height == f'{700 + spatial._HEIGHT_STEP}px'
    ctl['Shorter'].click()
    assert m.height == '700px'


def test_shorter_stops_at_the_floor_instead_of_going_negative():
    panel = _panel(300)
    m, ctl = panel.children[1], _controls(panel)
    for _ in range(10):
        ctl['Shorter'].click()
    assert m.height == f'{spatial._MIN_HEIGHT}px'


def test_stepping_a_relative_height_explains_itself_rather_than_no_op():
    panel = _panel('85vh')
    _controls(panel)['Taller'].click()
    assert panel.children[1].height == '85vh'          # unchanged
    assert '85vh' in panel.children[3].value           # ... and the reason is shown


# --- export to a browser ---------------------------------------------------

def test_open_map_in_browser_writes_a_standalone_page(tmp_path):
    out = tmp_path / 'map.html'
    path = spatial.open_map_in_browser(_panel(), filename=out, title='My ratio map',
                                       open_browser=False)
    assert path == out and out.exists()
    html = out.read_text()
    assert '<html' in html[:400].lower()
    assert 'My ratio map' in html                      # the browser tab name


def test_open_map_in_browser_accepts_a_bare_map(tmp_path):
    out = tmp_path / 'bare.html'
    spatial.open_map_in_browser(lonboard.Map([]), filename=out, open_browser=False)
    assert out.exists()


def test_open_map_in_browser_rejects_something_that_is_not_a_map():
    with pytest.raises(TypeError, match='lonboard Map'):
        spatial.open_map_in_browser(widgets.HBox([]), open_browser=False)


def test_open_map_in_browser_launches_the_browser_by_default(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr('webbrowser.open', lambda url: opened.append(url) or True)
    path = spatial.open_map_in_browser(_panel(), filename=tmp_path / 'm.html')
    assert opened == [path.as_uri()]


def test_the_browser_button_exports_and_reports_where(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr('webbrowser.open', lambda url: opened.append(url) or True)
    panel = _panel()
    _controls(panel)['Open in browser'].click()
    assert len(opened) == 1
    assert 'browser' in panel.children[3].value.lower()


def test_the_browser_button_reports_failure_instead_of_dying(monkeypatch):
    def boom(url):
        raise RuntimeError('no display')
    monkeypatch.setattr('webbrowser.open', boom)
    panel = _panel()
    _controls(panel)['Open in browser'].click()        # must not raise
    assert 'no display' in panel.children[3].value


# --------------------------------------------------------------------------
# vem_map: the VEM verdict, on the ground
# --------------------------------------------------------------------------

import ratio_study as rs

def _vem_sales():
    df = rs.load_sample_dataset()
    df['asr'] = df['assessment_value_new'] / df['time_adjusted_sale_price']
    return df


def _legend(panel):
    return [b.description for b in panel.children[2].children]


def test_vem_map_returns_the_standard_panel():
    panel = spatial.vem_map(df=_vem_sales(), assessment_header='assessment_value_new',
                            market_value_header='time_adjusted_sale_price', quantiles=10)
    assert [type(c).__name__ for c in panel.children] == ['HBox', 'Map', 'HBox', 'HTML']
    assert 'rs-map-panel' in panel._dom_classes


def test_vem_map_legend_covers_the_three_verdicts_and_counts_them():
    legend = _legend(spatial.vem_map(
        df=_vem_sales(), assessment_header='assessment_value_new',
        market_value_header='time_adjusted_sale_price',
        quantiles=20, acceptable_inequity_level=0.02))
    assert len(legend) == 3
    assert any('within the band' in t for t in legend)
    assert any('over-assessed' in t for t in legend)
    assert any('under-assessed' in t for t in legend)
    # Counts must add to the mapped sales, or the reader is being misled.
    counts = [int(t.split(':')[1].split('(')[0].strip().replace(',', '')) for t in legend]
    assert sum(counts) == len(_vem_sales())


def test_a_tighter_band_flags_more_sales():
    def flagged(level):
        legend = _legend(spatial.vem_map(
            df=_vem_sales(), assessment_header='assessment_value_new',
            market_value_header='time_adjusted_sale_price',
            quantiles=20, acceptable_inequity_level=level))
        return int(legend[0].split(':')[1].split('(')[0].strip().replace(',', ''))
    assert flagged(0.02) < flagged(0.10)     # fewer inside the band when it narrows


def test_strict_mode_flags_no_more_than_the_point_estimate_does():
    # ci_out_of_bounds needs the whole interval outside, so it is a subset.
    def within(strict):
        legend = _legend(spatial.vem_map(
            df=_vem_sales(), assessment_header='assessment_value_new',
            market_value_header='time_adjusted_sale_price', quantiles=20,
            acceptable_inequity_level=0.02, strict=strict))
        return int(legend[0].split(':')[1].split('(')[0].strip().replace(',', ''))
    assert within(True) >= within(False)


def test_vem_map_accepts_an_existing_ratio_column():
    panel = spatial.vem_map(df=_vem_sales(), market_value_header='time_adjusted_sale_price',
                            asr_col='asr', quantiles=10)
    assert _legend(panel)


def test_vem_map_takes_the_dot_size_and_opacity_arguments():
    panel = spatial.vem_map(df=_vem_sales(), market_value_header='time_adjusted_sale_price',
                            asr_col='asr', quantiles=10, dot_size=12, opacity=0.25)
    layer = panel.children[1].layers[0]
    assert layer.radius_scale == 12
    assert layer.opacity == pytest.approx(0.25)

    default = spatial.vem_map(df=_vem_sales(), market_value_header='time_adjusted_sale_price',
                              asr_col='asr', quantiles=10)
    assert default.children[1].layers[0].radius_scale == spatial.DEFAULT_DOT_SIZE
    assert default.children[1].layers[0].opacity == pytest.approx(
        spatial.DEFAULT_DOT_OPACITY)


# --------------------------------------------------------------------------
# local_moran: hot/cold-spot classification
# --------------------------------------------------------------------------

def test_each_of_the_seven_classes_is_reachable():
    # Constructed (z, p) pairs, one per class, so no band can go dead the
    # way the 90%/95% hotspots did under the old if/elif chain.
    cases = {
        7: (3.00, 0.005),    # hotspot 99%
        6: (2.20, 0.030),    # hotspot 95%
        5: (1.80, 0.080),    # hotspot 90%
        4: (0.20, 0.800),    # not significant
        3: (-1.80, 0.080),   # coldspot 90%
        2: (-2.20, 0.030),   # coldspot 95%
        1: (-3.00, 0.005),   # coldspot 99%
    }
    for expected, (z, pval) in cases.items():
        assert spatial._moran_class(z, pval) == expected, (z, pval)
    # Every class except the unclassified catch-all is covered above.
    assert set(cases) | {spatial.MORAN_UNCLASSIFIED} == set(spatial.MORAN_CLASSES)


@pytest.mark.parametrize('z, pval', [
    (3.00, 0.005), (2.20, 0.030), (1.80, 0.080), (5.00, 0.001),
])
def test_the_two_sides_are_exact_mirrors(z, pval):
    # The bands are stated on |z| with the sign applied afterwards, so a
    # dropped minus cannot make one side disagree with the other.
    hot = spatial._moran_class(z, pval)
    cold = spatial._moran_class(-z, pval)
    band = next(b for b in spatial.MORAN_BANDS if abs(z) >= b[0] and pval <= b[1])
    assert (cold, hot) == (band[2], band[3])


def test_a_significant_positive_z_below_2_58_is_a_hotspot_not_a_coldspot():
    """The bug this guards.

    The old first branch read ``z_value <= 2.58 and p_value <= 0.01``,
    missing a minus sign, so every significant point with z anywhere below
    +2.58 was returned as class 1 -- "coldspot 99% confidence" -- including
    genuine hotspots.
    """
    for z in (1.70, 1.96, 2.20, 2.57):
        code = spatial._moran_class(z, 0.005)
        label = spatial.MORAN_CLASSES[code][0]
        assert 'hotspot' in label, (z, label)


def test_the_90_and_95_hotspot_classes_survive_a_tiny_p_value():
    # Both were unreachable whenever p <= 0.01, because the mis-signed
    # first branch swallowed them.
    assert spatial._moran_class(1.80, 0.001) == 5
    assert spatial._moran_class(2.20, 0.001) == 6


def test_the_widest_band_wins():
    # Tested widest-first, so a point good at 99% is not also reported at
    # 95% or 90%.
    assert spatial._moran_class(3.0, 0.001) == 7
    assert spatial._moran_class(-3.0, 0.001) == 1


def test_z_and_p_disagreeing_lands_in_the_unclassified_bucket():
    # The p-value is simulated and the z is a normal approximation, so the
    # two can disagree. A large |z| with a middling p is in no band, and is
    # not inside the not-significant box either.
    assert spatial._moran_class(2.40, 0.40) == spatial.MORAN_UNCLASSIFIED
    assert spatial._moran_class(-2.40, 0.40) == spatial.MORAN_UNCLASSIFIED


def test_every_band_code_has_a_class_entry():
    for _z, _p, cold_code, hot_code in spatial.MORAN_BANDS:
        assert cold_code in spatial.MORAN_CLASSES
        assert hot_code in spatial.MORAN_CLASSES


def _moran_cluster_df(n_side=12):
    """A grid with one planted high-ratio block and one planted low one."""
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    asr = np.full(n_side * n_side, 1.0)
    high = (xs.ravel() >= n_side - 4) & (ys.ravel() >= n_side - 4)
    low = (xs.ravel() <= 3) & (ys.ravel() <= 3)
    asr[high] = 2.0
    asr[low] = 0.2
    df = pd.DataFrame({
        'latitude': ys.ravel().astype(float),
        'longitude': xs.ravel().astype(float),
        'asr': asr,
    })
    return df, high, low


def test_a_cluster_of_high_ratios_is_a_hotspot():
    """The sign convention, end to end.

    ``G_Local``'s z is positive for a cluster of high values; a cluster of
    high ASRs is over-assessment, which is what "hotspot" has to mean.
    ``Z_Value`` used to be multiplied by -1, which inverted it, so a
    planted block of high ratios came back labelled "coldspot".
    """
    df, high, low = _moran_cluster_df()
    table = spatial.local_moran(df=df, ratio='asr').children[1].layers[0].table
    flags = np.array(table.column('flag').to_pylist())
    z = np.array(table.column('Z_Value').to_pylist())

    assert z[high].mean() > 0
    assert all('hotspot' in flag for flag in flags[high])

    assert z[low].mean() < 0
    assert all('coldspot' in flag for flag in flags[low])


def test_the_mapped_category_matches_the_classifier():
    df, _high, _low = _moran_cluster_df()
    table = spatial.local_moran(df=df, ratio='asr').children[1].layers[0].table
    codes = table.column('category').to_pylist()
    z = table.column('Z_Value').to_pylist()
    pvals = table.column('P_Value').to_pylist()
    assert codes == [spatial._moran_class(a, b) for a, b in zip(z, pvals)]


def test_the_doubled_p_value_stays_a_probability():
    # `p_z_sim` is one-tailed, so it tops out at 0.5 and doubling it cannot
    # push the reported P_Value past 1.
    df, _high, _low = _moran_cluster_df()
    table = spatial.local_moran(df=df, ratio='asr').children[1].layers[0].table
    pvals = np.array(table.column('P_Value').to_pylist())
    assert ((pvals >= 0) & (pvals <= 1)).all()


def _moran_df(n_side=9):
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    return pd.DataFrame({
        'latitude': ys.ravel().astype(float),
        'longitude': xs.ravel().astype(float),
        'asr': (xs + ys).ravel().astype(float),
    })


def _moran_layer_colours(panel):
    """category code -> the RGB actually handed to the layer for it."""
    layer = panel.children[1].layers[0]
    codes = np.array(layer.table.column('category').to_pylist())
    fills = np.array(layer.get_fill_color.to_pylist())
    return {int(code): tuple(fills[np.where(codes == code)[0][0]])
            for code in sorted(set(codes.tolist()))}


def test_non_significant_dots_are_grey():
    # They were black, which drew the eye to exactly the sales with no
    # finding on them. The regression this guards: the grey was once put on
    # MORAN_UNCLASSIFIED (code 0, effectively never populated) instead of
    # MORAN_NOT_SIGNIFICANT, so nothing on screen changed.
    label, rgb = spatial.MORAN_CLASSES[spatial.MORAN_NOT_SIGNIFICANT]
    assert label == 'not significant'
    assert rgb == (128, 128, 128)

    colours = _moran_layer_colours(spatial.local_moran(df=_moran_df(), ratio='asr'))
    assert colours[spatial.MORAN_NOT_SIGNIFICANT] == (128, 128, 128)


def test_every_moran_dot_gets_its_tables_colour():
    panel = spatial.local_moran(df=_moran_df(), ratio='asr')
    for code, rgb in _moran_layer_colours(panel).items():
        assert rgb == spatial.MORAN_CLASSES[code][1], code


def test_the_moran_legend_matches_the_dots_it_labels():
    # The label and the colour used to be written out three separate times;
    # this is what caught them drifting apart.
    panel = spatial.local_moran(df=_moran_df(), ratio='asr')
    legend = panel.children[2].children
    assert len(legend) == len(spatial.MORAN_CLASSES)
    for button, (label, rgb) in zip(legend, spatial.MORAN_CLASSES.values()):
        assert button.description == label
        assert button.style.button_color == 'rgb({},{},{})'.format(*rgb)


def test_the_moran_flag_column_matches_the_class_table():
    panel = spatial.local_moran(df=_moran_df(), ratio='asr')
    layer = panel.children[1].layers[0]
    codes = layer.table.column('category').to_pylist()
    flags = layer.table.column('flag').to_pylist()
    assert flags == [spatial.MORAN_CLASSES[code][0] for code in codes]


@pytest.mark.parametrize('rgb, expect_white', [
    ((0, 0, 139), True),         # dark blue
    ((211, 47, 47), True),       # red
    ((128, 128, 128), True),     # mid grey -- the new "not significant"
    ((189, 189, 189), False),    # light grey
    ((146, 255, 225), False),    # pale mint
    ((255, 255, 255), False),    # white
])
def test_legend_text_flips_to_white_on_a_dark_fill(rgb, expect_white):
    widgets = pytest.importorskip('ipywidgets')
    button = widgets.Button(disabled=True, description='x')
    spatial._style_legend_button(button, rgb)
    assert button.style.button_color == 'rgb({},{},{})'.format(*rgb)
    assert (button.style.text_color == 'rgb(255,255,255)') is expect_white


def test_local_moran_takes_the_dot_size_and_opacity_arguments():
    # local_moran had no coverage at all; this pins the two presentation
    # arguments it now shares with the other maps, and that it still
    # returns the standard panel.
    df = _grid_df(n_side=8).rename(columns={'gradient': 'asr'})
    panel = spatial.local_moran(df=df, ratio='asr', dot_size=12, opacity=0.25)
    assert [type(c).__name__ for c in panel.children] == ['HBox', 'Map', 'HBox', 'HTML']
    layer = panel.children[1].layers[0]
    assert layer.radius_scale == 12
    assert layer.opacity == pytest.approx(0.25)

    default = spatial.local_moran(df=df, ratio='asr')
    assert default.children[1].layers[0].radius_scale == spatial.DEFAULT_DOT_SIZE
    assert default.children[1].layers[0].opacity == pytest.approx(
        spatial.DEFAULT_DOT_OPACITY)


def test_vem_map_takes_the_height_argument():
    panel = spatial.vem_map(df=_vem_sales(), market_value_header='time_adjusted_sale_price',
                            asr_col='asr', quantiles=10, height=820)
    assert panel.children[1].height == '820px'


@pytest.mark.parametrize('drop, missing', [
    ('latitude', 'latitude'),
    ('longitude', 'longitude'),
])
def test_vem_map_needs_coordinates(drop, missing):
    df = _vem_sales().drop(columns=[drop])
    with pytest.raises(RuntimeError, match=missing):
        spatial.vem_map(df=df, market_value_header='time_adjusted_sale_price', asr_col='asr')


def test_vem_map_needs_a_market_value_column():
    with pytest.raises(RuntimeError, match='market_value_header'):
        spatial.vem_map(df=_vem_sales(), asr_col='asr')


def test_vem_map_needs_a_way_to_get_the_ratio():
    df = _vem_sales().drop(columns=['asr'])
    with pytest.raises(RuntimeError, match='assessment_header'):
        spatial.vem_map(df=df, market_value_header='time_adjusted_sale_price')


def test_the_vem_frame_keeps_the_callers_index():
    # vem_map joins the per-sale verdict back onto the source frame, which
    # only works because vem() no longer resets the index.
    df = _vem_sales().sample(frac=1, random_state=0)
    frame = rs.vem(df, market_value_proxy='time_adjusted_sale_price',
                   asr_col='asr', quantiles=10)['dataframe']
    assert set(frame.index) == set(df.index)


# --------------------------------------------------------------------------
# ratio_map: hardcoded outlier tails, graded middle, optional groupby
# --------------------------------------------------------------------------
# Neighbourhood medians engineered so the split is unambiguous: overall
# median of medians is 1.00 (B); A sits below 90% of it, C above 110%.

def _ratio_map_df():
    return pd.DataFrame({
        'latitude':  [40.00, 40.01, 40.02, 40.03, 40.04, 40.05],
        'longitude': [-75.00, -75.01, -75.02, -75.03, -75.04, -75.05],
        'nbhd': ['A', 'A', 'B', 'B', 'C', 'C'],
        'asr': [0.70, 0.70, 1.00, 1.00, 1.50, 1.50],
    })


def test_ratio_map_returns_the_standard_panel():
    panel = spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd')
    assert [type(c).__name__ for c in panel.children] == ['HBox', 'Map', 'HBox', 'HTML']
    assert 'rs-map-panel' in panel._dom_classes


def test_ratio_map_draws_every_point_as_a_circle():
    # The outlier tails used to be triangles in their own PolygonLayer.
    # Circles everywhere now, so `dot_size` means the same thing for every
    # point on the map -- including across the background/foreground split.
    for scheme in spatial.COLOUR_CODE_SCHEMES:
        panel = spatial.ratio_map(df=_ratio_map_df(), column='asr',
                                  groupby='nbhd', colour_code_by=scheme)
        layer_types = {type(layer).__name__ for layer in panel.children[1].layers}
        assert layer_types == {'ScatterplotLayer'}, scheme


def test_quantile_scheme_keeps_the_fixed_outlier_tails_in_its_legend():
    panel = spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd',
                              colour_code_by='quantile')
    legend = [b.description for b in panel.children[2].children]
    assert any('low outlier' in t for t in legend)
    assert any('high outlier' in t for t in legend)
    assert sum('outlier' in t for t in legend) == 2


def test_grey_is_drawn_underneath_the_coloured_bands():
    # deck.gl draws layers in the order the Map receives them, so the
    # background points have to be in the *first* layer. Grey is usually
    # most of the sample; left in draw order it buries the few dots the
    # map exists to show.
    rng = np.random.default_rng(11)
    ratios = np.concatenate([rng.normal(1.0, 0.03, 200),
                             [1.6, 1.7, 0.4, 0.3]])
    df = pd.DataFrame({
        'latitude': 40.0 + np.arange(len(ratios)) * 0.001,
        'longitude': -75.0 - np.arange(len(ratios)) * 0.001,
        'asr': ratios,
    })
    for scheme in ('iqr', 'mad', 'standard'):
        codes = _codes(scheme, ratios)
        background = int(np.isin(codes, spatial.BACKGROUND_BANDS).sum())
        layers = spatial.ratio_map(df=df, column='asr',
                                   colour_code_by=scheme).children[1].layers
        assert len(layers) == 2, scheme
        assert len(layers[0].table) == background, scheme
        assert len(layers[1].table) == len(ratios) - background, scheme


def test_the_quantile_scheme_draws_its_fixed_tails_on_top():
    # No single neutral colour in this scheme -- its middle is a gradient --
    # so the split is by role: the graded middle underneath, the 80%/120%
    # tails over it, as they were when they were triangles in their own
    # layer.
    legend = [b.description for b in spatial.ratio_map(
        df=_ratio_map_df(), column='asr', groupby='nbhd',
        colour_code_by='quantile').children[2].children]
    tails = sum('outlier' in t for t in legend)
    layers = spatial.ratio_map(df=_ratio_map_df(), column='asr',
                               groupby='nbhd',
                               colour_code_by='quantile').children[1].layers
    assert tails == 2                       # A is low, C is high
    assert len(layers) == 2
    # One neighbourhood in the graded middle (B), two in the tails.
    assert len(layers[0].table) == 1
    assert len(layers[1].table) == 2


def test_a_map_with_no_coloured_bands_is_a_single_background_layer():
    # `from_geopandas` cannot build a layer from an empty frame, so the
    # split has to skip a side rather than emit a zero-row layer.
    df = pd.DataFrame({
        'latitude': 40.0 + np.arange(30) * 0.001,
        'longitude': -75.0 - np.arange(30) * 0.001,
        'asr': np.ones(30),
    })
    layers = spatial.ratio_map(df=df, column='asr').children[1].layers
    assert len(layers) == 1
    assert len(layers[0].table) == 30


def test_a_map_with_no_grey_at_all_is_a_single_foreground_layer():
    ratios = np.array([0.5] * 15 + [1.6] * 15)
    df = pd.DataFrame({
        'latitude': 40.0 + np.arange(len(ratios)) * 0.001,
        'longitude': -75.0 - np.arange(len(ratios)) * 0.001,
        'asr': ratios,
    })
    layers = spatial.ratio_map(df=df, column='asr',
                               colour_code_by='standard').children[1].layers
    assert len(layers) == 1
    assert len(layers[0].table) == len(ratios)


def test_ratio_map_dot_size_reaches_every_layer():
    # Both the background and the foreground layer, or the split would show
    # up as two different dot sizes on the same map.
    for scheme in spatial.COLOUR_CODE_SCHEMES:
        panel = spatial.ratio_map(df=_ratio_map_df(), column='asr',
                                  groupby='nbhd', colour_code_by=scheme,
                                  dot_size=17)
        sizes = {layer.radius_scale for layer in panel.children[1].layers}
        assert sizes == {17}, scheme

    default = spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd')
    assert {layer.radius_scale for layer in default.children[1].layers} == {
        spatial.DEFAULT_DOT_SIZE}


def test_ratio_map_opacity_reaches_every_layer():
    for scheme in spatial.COLOUR_CODE_SCHEMES:
        panel = spatial.ratio_map(df=_ratio_map_df(), column='asr',
                                  groupby='nbhd', colour_code_by=scheme,
                                  opacity=0.45)
        opacities = [layer.opacity for layer in panel.children[1].layers]
        assert opacities == [pytest.approx(0.45)] * len(opacities), scheme

    default = spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd')
    defaults = [layer.opacity for layer in default.children[1].layers]
    assert defaults == [pytest.approx(spatial.DEFAULT_DOT_OPACITY)] * len(defaults)


def test_opacity_may_be_given_as_a_percentage():
    panel = spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd',
                              opacity=40)
    assert panel.children[1].layers[0].opacity == pytest.approx(0.4)


@pytest.mark.parametrize('bad', [-0.1, 101, 'half', None, np.nan])
def test_ratio_map_rejects_an_impossible_opacity(bad):
    with pytest.raises(ValueError, match='opacity'):
        spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd',
                          opacity=bad)


def test_ratio_map_rejects_an_unknown_colour_code_by():
    with pytest.raises(ValueError, match='colour_code_by'):
        spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd',
                          colour_code_by='rainbow')


def test_ratio_map_legend_never_exceeds_five_bands():
    # A realistic sample -- a tight central mass with a mild and a far
    # outlier on each side -- so every scheme has bands to fill, and none
    # of them may spend more than five colours on it.
    rng = np.random.default_rng(11)
    ratios = np.concatenate([rng.normal(1.0, 0.05, 120),
                             [1.18, 1.75, 0.82, 0.35]])
    df = pd.DataFrame({
        'latitude': 40.0 + np.arange(len(ratios)) * 0.001,
        'longitude': -75.0 - np.arange(len(ratios)) * 0.001,
        'asr': ratios,
    })
    grey = 'rgb({},{},{})'.format(*spatial.BAND_COLOURS[spatial.BAND_CENTRAL])
    for scheme in ('iqr', 'mad', 'standard'):
        panel = spatial.ratio_map(df=df, column='asr', colour_code_by=scheme)
        colours = [b.style.button_color for b in panel.children[2].children]
        assert grey in colours, scheme
        assert 1 < len(colours) <= 5, scheme
        assert len(set(colours)) == len(colours), scheme


def test_ratio_map_fits_the_view_to_the_data_instead_of_a_fixed_zoom():
    # The neighbourhoods above sit a few hundredths of a degree apart --
    # much tighter than the old hardcoded zoom=8, which was sized for
    # nothing in particular.
    zoom = spatial.ratio_map(df=_ratio_map_df(), column='asr',
                             groupby='nbhd').children[1].view_state.zoom
    assert zoom > 8


def test_ratio_map_view_survives_one_bad_geocode():
    # A single (0, 0) row -- a classic failed-geocode value -- must not
    # drag the whole map's view out to the globe for everyone else's sale.
    df = pd.concat([_ratio_map_df(), pd.DataFrame({
        'latitude': [0.0], 'longitude': [0.0], 'nbhd': ['A'], 'asr': [0.70],
    })], ignore_index=True)
    zoom = spatial.ratio_map(df=df, column='asr', groupby='nbhd').children[1].view_state.zoom
    assert zoom > 8


def test_ratio_map_groupby_is_optional_and_colours_every_row_on_its_own():
    df = _ratio_map_df().drop(columns=['nbhd'])
    panel = spatial.ratio_map(df=df, column='asr')
    # One point per sale, not per group, now that there is no group to
    # dissolve into -- and every sale mapped, across however many layers
    # the background/foreground split needed.
    assert sum(len(layer.table) for layer in panel.children[1].layers) == len(df)


def test_ratio_map_rejects_an_invalid_column(capsys):
    result = spatial.ratio_map(df=_ratio_map_df(), column='nope', groupby='nbhd')
    assert result is None
    assert 'valid column name' in capsys.readouterr().out


def test_ratio_map_takes_the_height_argument():
    panel = spatial.ratio_map(df=_ratio_map_df(), column='asr', groupby='nbhd', height=820)
    assert panel.children[1].height == '820px'


# --------------------------------------------------------------------------
# Global Moran's I: the plain-language reading
# --------------------------------------------------------------------------

def test_moran_message_never_calls_a_significant_result_insignificant():
    """I = 0.006 with p = 0.019 used to read "no spatial autocorrelation ...
    statistically insignificant" -- at 5 % it is significant, and the size is
    what is negligible."""
    from ratio_study.spatial import _moran_message

    text = _moran_message(0.0058, 2.34, 0.0192)
    assert "negligible" in text
    assert "not statistically significant" not in text
    assert "statistically significant at the 5% level" in text


def test_moran_message_reports_a_null_result_as_not_significant():
    from ratio_study.spatial import _moran_message

    text = _moran_message(0.004, 0.5, 0.62)
    assert "not statistically significant at the 5% level" in text
    assert text.startswith("No action required")


def test_moran_message_flags_a_strong_pattern_for_action():
    from ratio_study.spatial import _moran_message

    assert _moran_message(0.6, 9.0, 0.001).startswith("Action required")
    assert "negative" in _moran_message(-0.3, -4.0, 0.001)
