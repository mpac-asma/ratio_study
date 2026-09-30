"""Tests for the Vertical Equity Panel (gauges + VEM/VEI charts)."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ratio_study.graphs import (
    _format_proxy_cod_labels,
    _format_proxy_labels,
    _gauge_angle,
    _gauge_status,
    _unwrap_vei_result,
    _ve_coerce_float,
    _ve_display_center,
    _ve_smart_limits,
    _vei_chart_title,
    _vem_chart_title,
    _vem_smoothing_label,
    _ve_gauge_specs,
    _vem_axes,
    ve_scat_median_plot,
    vertical_equity_panel,
)
from ratio_study import indicators as I
from ratio_study.indicators import vem

# Dial geometry the gauges are built on: green 45 deg on top, red 135 deg
# each side, grey 45 deg at the bottom.
TOP = 90.0
PROG_EDGE = 67.5
REG_EDGE = 112.5
BOTTOM = 270.0
BAND = dict(lo=-0.10, hi=0.10, outer_lo=-0.30, outer_hi=0.30)


@pytest.fixture
def ve_df():
    """1,200 sales with a mild, deliberately regressive ratio pattern."""
    rng = np.random.default_rng(3)
    price = rng.lognormal(12.5, 0.55, 1200)
    ratio = 1.05 * (price / price.mean()) ** -0.08 * rng.lognormal(0, 0.10, 1200)
    return pd.DataFrame({"sale": price, "assess": price * ratio})


# --------------------------------------------------------------------------
# Angle mapping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value, status, expected", [
    (0.0, "acceptable", TOP),                 # neutral points straight up
    (0.10, "acceptable", PROG_EDGE),          # band edge lands on the wedge edge
    (-0.10, "acceptable", REG_EDGE),
    (0.05, "acceptable", 78.75),              # linear inside the green wedge
    (0.10, "progressive", PROG_EDGE),        # red wedge starts at the band edge
    (0.30, "progressive", PROG_EDGE - 135),  # full scale reaches its outer end
    (-0.30, "regressive", REG_EDGE + 135),
    (0.0, "inconclusive", BOTTOM),           # grey wedge, magnitude ignored
    (0.0, "unavailable", BOTTOM),
])
def test_gauge_angle_landmarks(value, status, expected):
    assert _gauge_angle(value, status=status, **BAND) == pytest.approx(expected)


@pytest.mark.parametrize("value", [3.0, 1e6, -3.0, -1e6])
def test_gauge_angle_clamps_beyond_full_scale(value):
    """Extreme values stop at the red wedge edge instead of wrapping round."""
    status = "progressive" if value > 0 else "regressive"
    angle = _gauge_angle(value, status=status, **BAND)
    expected = PROG_EDGE - 135 if value > 0 else REG_EDGE + 135
    assert angle == pytest.approx(expected)


def test_gauge_angle_handles_asymmetric_band():
    """PRD's (0.98, 1.03) band is asymmetric once reflected to +0.02/-0.03."""
    kw = dict(lo=-0.03, hi=0.02, outer_lo=-0.09, outer_hi=0.06)
    assert _gauge_angle(0.0, status="acceptable", **kw) == pytest.approx(TOP)
    assert _gauge_angle(0.02, status="acceptable", **kw) == pytest.approx(PROG_EDGE)
    assert _gauge_angle(-0.03, status="acceptable", **kw) == pytest.approx(REG_EDGE)


def test_gauge_angle_never_leaves_its_region():
    """A value outside its status region is clamped, not drawn elsewhere."""
    assert _gauge_angle(99.0, status="acceptable", **BAND) == pytest.approx(PROG_EDGE)
    assert _gauge_angle(-99.0, status="acceptable", **BAND) == pytest.approx(REG_EDGE)


# --------------------------------------------------------------------------
# Status classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value, conclusive, expected", [
    (0.0, True, "acceptable"),
    (0.10, True, "acceptable"),          # inclusive at the band edge
    (0.11, True, "progressive"),
    (-0.11, True, "regressive"),
    (0.02, False, "acceptable"),         # inside the band: no evidence of bias
    (0.25, False, "inconclusive"),      # outside, but not significant
    (None, True, "unavailable"),
])
def test_gauge_status(value, conclusive, expected):
    assert _gauge_status(value, -0.10, 0.10, conclusive=conclusive) == expected


# --------------------------------------------------------------------------
# Value coercion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw, value, flagged", [
    ("0.043*", 0.043, True),        # PRB non-significance marker
    ("-0.12*", -0.12, True),
    ("0.043", 0.043, False),
    (0.05, 0.05, False),
    (None, None, False),
    (float("nan"), None, False),
])
def test_ve_coerce_float(raw, value, flagged):
    got_value, got_flag = _ve_coerce_float(raw)
    assert got_flag is flagged
    if value is None:
        assert got_value is None
    else:
        assert got_value == pytest.approx(value)


# --------------------------------------------------------------------------
# Gauge specifications
# --------------------------------------------------------------------------

def test_prd_is_reflected_so_right_means_progressive():
    """PRD below 1 is progressivity, so its dial runs the other way."""
    progressive = _ve_gauge_specs(prd_value=0.90)[1]
    regressive = _ve_gauge_specs(prd_value=1.10)[1]
    assert progressive["status"] == "progressive"
    assert regressive["status"] == "regressive"
    # Progressive sits in the right half of the dial, regressive in the left.
    assert progressive["angle"] < PROG_EDGE
    assert regressive["angle"] > REG_EDGE


def test_non_significant_prb_outside_band_is_inconclusive():
    spec = _ve_gauge_specs(prb_value="-0.25*")[0]
    assert spec["status"] == "inconclusive"
    assert spec["angle"] == pytest.approx(BOTTOM)
    # The point estimate is still reported, not hidden.
    assert "-0.250" in spec["value_text"]


def test_prb_flag_argument_matches_the_string_marker():
    by_marker = _ve_gauge_specs(prb_value="-0.25*")[0]
    by_flag = _ve_gauge_specs(prb_value=-0.25, prb_not_significant=True)[0]
    assert by_marker["status"] == by_flag["status"] == "inconclusive"


def test_vei_needle_follows_the_point_estimate_and_discloses_the_verdict():
    """A large VEI stays in the red wedge even if the test says Acceptable.

    ``vei_analysis`` can conclude 'Acceptable' for a large point estimate
    -- when its confidence interval still reaches the threshold -- so the
    panel shows where the estimate falls and reports the differing verdict
    underneath rather than moving the needle into the green.
    """
    spec = _ve_gauge_specs(vei_metrics={
        "vei": -38.3, "conclusion": "Acceptable", "acceptable_range": 10,
    })[2]
    assert spec["status"] == "regressive"
    assert spec["angle"] > REG_EDGE
    assert spec["detail"] == ("VEI test: Acceptable",)
    assert spec["value_text"] == "-38.3%"


def test_vei_agreeing_verdict_adds_no_note():
    spec = _ve_gauge_specs(vei_metrics={
        "vei": -3.0, "conclusion": "Acceptable", "acceptable_range": 10,
    })[2]
    assert spec["status"] == "acceptable"
    assert spec["detail"] == ()


def test_vei_small_sample_is_inconclusive_not_neutral():
    """vei_analysis returns 0 when it bails out; 0 must not read as neutral."""
    spec = _ve_gauge_specs(vei_metrics={
        "vei": 0, "conclusion": "Sample size too small", "acceptable_range": 10,
    })[2]
    assert spec["status"] == "inconclusive"
    assert spec["value_text"] == "n/a"


def test_vem_takes_its_sign_from_the_direction():
    """Inequity_severity is unsigned, so the conclusion supplies the sign."""
    regressive = _ve_gauge_specs(vem_result={
        "Inequity_conclusion": "Regressivity",
        "Inequity_severity": 20.0, "Inequity_prevalence": 50.0,
    })[3]
    progressive = _ve_gauge_specs(vem_result={
        "Inequity_conclusion": "Progressivity",
        "Inequity_severity": 20.0, "Inequity_prevalence": 50.0,
    })[3]
    assert regressive["value_text"] == "-20.0%"
    assert progressive["value_text"] == "+20.0%"
    assert regressive["angle"] > REG_EDGE
    assert progressive["angle"] < PROG_EDGE
    assert regressive["detail"] == ("prevalence 50.0%",)


def test_vem_inconclusive_is_shown_unsigned():
    """No direction was established, so no sign should be implied."""
    spec = _ve_gauge_specs(vem_result={
        "Inequity_conclusion": "Inconclusive",
        "Inequity_severity": 18.0, "Inequity_prevalence": 40.0,
    })[3]
    assert spec["status"] == "inconclusive"
    assert spec["value_text"] == "18.0%"


def test_vem_acceptable_points_straight_up():
    spec = _ve_gauge_specs(vem_result={
        "Inequity_conclusion": "no significant vertical inequity",
        "Inequity_severity": 0.0, "Inequity_prevalence": 0.0,
    })[3]
    assert spec["status"] == "acceptable"
    assert spec["angle"] == pytest.approx(TOP)


def test_specs_degrade_to_unavailable_without_inputs():
    specs = _ve_gauge_specs()
    assert [s["label"] for s in specs] == ["PRB", "PRD", "VEI", "VEM"]
    assert all(s["status"] == "unavailable" for s in specs)
    assert all(s["value_text"] == "n/a" for s in specs)


def test_custom_assessment_standards_move_the_band_ticks():
    specs = _ve_gauge_specs(prb_value=0.04, prd_value=1.0,
                            assessment_standards={"PRB": (-0.03, 0.03)})
    assert specs[0]["status"] == "progressive"      # 0.04 now breaches 0.03
    assert specs[0]["tick_hi"] == "+0.03"


# --------------------------------------------------------------------------
# VEI result unwrapping
# --------------------------------------------------------------------------

def test_unwrap_vei_result_accepts_both_shapes():
    flat = {"vei": 1.0, "conclusion": "Acceptable", "pg_stats": [{"a": 1}]}
    assert _unwrap_vei_result(flat)["vei"] == 1.0

    wrapped = {"vei_metrics": {"vei": 2.0, "conclusion": "Acceptable"},
               "group_stats": [{"a": 1}]}
    got = _unwrap_vei_result(wrapped)
    assert got["vei"] == 2.0 and got["pg_stats"] == [{"a": 1}]

    as_df = {"vei_metrics": pd.DataFrame([{"vei": 3.0, "conclusion": "Acceptable"}]),
             "group_stats": [{"a": 1}]}
    assert _unwrap_vei_result(as_df)["vei"] == 3.0

    assert _unwrap_vei_result(None) == {}


def test_format_proxy_labels():
    assert _format_proxy_labels([1234567, 45678, 999]) == ["$1.23M", "$46K", "$999"]
    assert _format_proxy_labels([45678], short=False) == ["$45,678"]


def test_format_proxy_cod_labels_puts_group_cod_under_the_proxy():
    pg_stats = [{"med_proxy": 290000, "cod": 0.15},
                {"med_proxy": 1234567, "cod": 0.0824}]
    assert _format_proxy_cod_labels(pg_stats) == ["$290K\n15.0%",
                                                  "$1.23M\n8.2%"]
    assert _format_proxy_cod_labels(pg_stats[:1], short=False) == ["$290,000\n15.0%"]


def test_format_proxy_cod_labels_falls_back_to_the_asr_vector():
    # Results produced before ``vei_analysis`` stored a per-group CoD, and
    # any hand-built pg_stats, still label from the ratios themselves.
    ratios = np.array([0.8, 1.0, 1.0, 1.2])
    labels = _format_proxy_cod_labels([{"med_proxy": 100000,
                                        "asr_vector": ratios}])
    assert labels == ["$100K\n10.0%"]

    # Nothing to compute from: the proxy amount stands alone.
    assert _format_proxy_cod_labels([{"med_proxy": 100000}]) == ["$100K"]


def test_vei_analysis_reports_a_cod_per_percentile_group(ve_df):
    asr = (ve_df["assess"] / ve_df["sale"]).to_numpy()
    proxy = ve_df["sale"].to_numpy()
    groups = I.vei_analysis(asr, proxy)["pg_stats"]
    for group in groups:
        assert group["cod"] == pytest.approx(I.cod(group["asr_vector"],
                                                   group["median_ratio"]))


# --------------------------------------------------------------------------
# End-to-end panel
# --------------------------------------------------------------------------

def test_panel_returns_figure_with_expected_axes(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        # four gauges, the secondary-indicator strip, then the VEI and VEM charts
        assert len(fig.axes) == 7
        assert [ax.get_label() for ax in fig.axes].count('secondary_indicators') == 1
    finally:
        plt.close(fig)


def test_panel_is_laid_out_in_letter_proportions(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        width, height = fig.get_size_inches()
        assert (width, height) == pytest.approx((8.5, 11.0))
    finally:
        plt.close(fig)


def test_panel_stacks_vei_above_vem(ve_df):
    """Both charts span the full width, VEI on top of VEM."""
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        vei_box = fig.axes[-2].get_position()
        vem_box = fig.axes[-1].get_position()
        assert vem_box.y1 < vei_box.y0                    # stacked, not side by side
        assert vei_box.x0 == pytest.approx(vem_box.x0)
        assert vei_box.width == pytest.approx(vem_box.width)
        assert vei_box.width > 0.8                        # full panel width
    finally:
        plt.close(fig)


def test_panel_returns_reportlab_image(ve_df):
    caption, image = vertical_equity_panel(
        ve_df, "assess", "sale", return_object=True, subtitle="unit test",
    )
    assert caption == "Vertical Equity Panel – unit test"
    # Sized to fit the report page (7 inch text width, letter portrait) --
    # a portrait panel is capped by the text height, not the width.
    assert image.drawWidth <= 504.0
    assert image.drawHeight <= 650
    # Letter proportions survive the trip through ReportLab.
    assert image.drawHeight / image.drawWidth == pytest.approx(11.0 / 8.5, rel=0.02)


def test_panel_saves_png_and_appends_extension(ve_df, tmp_path):
    target = tmp_path / "panel"
    path = vertical_equity_panel(ve_df, "assess", "sale", graph_filepath=str(target))
    assert path == str(target) + ".png"
    assert (tmp_path / "panel.png").stat().st_size > 0


def test_panel_survives_a_sample_too_small_for_vei_and_vem(ve_df):
    """Below the VEI (20) and VEM (30) minimums the panel still renders."""
    fig = vertical_equity_panel(ve_df.head(18), "assess", "sale", show=False)
    try:
        assert len(fig.axes) == 7
    finally:
        plt.close(fig)


def test_panel_drops_nan_and_non_positive_rows(ve_df):
    dirty = ve_df.head(400).copy()
    dirty.loc[dirty.index[:5], "assess"] = np.nan
    dirty.loc[dirty.index[5:10], "sale"] = 0
    fig = vertical_equity_panel(dirty, "assess", "sale", show=False)
    plt.close(fig)


def test_panel_rejects_unknown_columns(ve_df):
    with pytest.raises(RuntimeError):
        vertical_equity_panel(ve_df, "not_a_column", "sale", show=False)


def test_panel_vem_chart_is_centred_on_the_median_asr(ve_df):
    """The VEM axes read in ASR units, not as offsets from zero."""
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        vem_ax = fig.axes[-1]         # gauges, strip, VEI, then VEM last
        center = float(np.median(ve_df["assess"] / ve_df["sale"]))
        low, high = vem_ax.get_ylim()
        assert vem_ax.get_ylabel() == "ASR"
        assert low < center < high
        assert vem_ax.get_title().startswith("Vertical Equity: ")
        assert vem_ax.get_xlabel() == "Value (log scale)"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
def test_panel_accepts_other_backends(ve_df, backend):
    frame = pytest.importorskip(backend)
    if backend == "polars":
        data = frame.from_pandas(ve_df)
    else:
        data = frame.Table.from_pandas(ve_df)
    fig = vertical_equity_panel(data, "assess", "sale", show=False)
    plt.close(fig)


# --------------------------------------------------------------------------
# VEM display scale: indices on the centred ASR, charts in ASR units
# --------------------------------------------------------------------------

@pytest.fixture
def vem_result(ve_df):
    """A VEM run on the regressive fixture, with a CI band."""
    return vem(
        pd.DataFrame({"sale": ve_df["sale"],
                      "asr": ve_df["assess"] / ve_df["sale"]}),
        market_value_proxy="sale", window_size=120, conf_level=0.9,
    )


def test_vem_reports_the_median_it_centred_on(ve_df, vem_result):
    center = float(np.median(ve_df["assess"] / ve_df["sale"]))
    assert vem_result["asr_center"] == pytest.approx(center)

    frame = vem_result["dataframe"]
    # The working ASR stays centred -- that is what every index is built on.
    assert frame["asr"].to_numpy() == pytest.approx(
        frame["asr_raw"].to_numpy() - center)
    assert float(np.median(frame["asr"])) == pytest.approx(0.0)


def test_vem_indices_are_computed_on_the_centred_scale(vem_result):
    """Severity/prevalence read the centred moving median, not the shifted one."""
    frame = vem_result["dataframe"]
    # Both indices read the *confirmed* rows -- moving median and its whole
    # CI band outside the band -- so the two numbers describe the same
    # windows and can be quoted together.
    confirmed = frame["ci_out_of_bounds"] != 0
    assert vem_result["Inequity_severity"] == pytest.approx(
        round(float(frame.loc[confirmed, "moving_median"].abs().mean()) * 100, 4))
    assert vem_result["Inequity_prevalence"] == pytest.approx(
        round(float(confirmed.mean()) * 100, 4))
    assert vem_result["Inequity_conclusion"] == "Regressivity"


def test_vem_plot_is_drawn_in_asr_units(vem_result):
    frame = vem_result["dataframe"]
    center = vem_result["asr_center"]
    fig = ve_scat_median_plot(frame, "t", None, acceptable_inequity_level=0.1,
                              zoom_y=True, conf_level=0.9, center=center,
                              smart_zoom=False)

    traces = {trace.name: np.asarray(trace.y, dtype=float) for trace in fig.data}
    assert traces["Moving Median ASR"] == pytest.approx(
        frame["moving_median"].to_numpy() + center)
    assert traces["Upper 90% CI for median"] == pytest.approx(
        frame["median_ci_upper"].to_numpy() + center)

    # Scatter points are the untouched ratios, split by the bounds flag.
    scattered = np.concatenate([traces["ASR (acceptable local median)"],
                                traces["ASR (excessive local median)"]])
    assert sorted(scattered) == pytest.approx(sorted(frame["asr_raw"].to_numpy()))

    # Reference lines: the median ASR, and the acceptable band around it.
    assert sorted(shape.y0 for shape in fig.layout.shapes) == pytest.approx(
        [center - 0.1, center, center + 0.1])
    assert fig.layout.yaxis.title.text == "ASR"
    assert fig.layout.yaxis.range == pytest.approx((center - 0.3, center + 0.3))


def test_vem_axes_is_drawn_in_asr_units(vem_result):
    frame = vem_result["dataframe"]
    center = vem_result["asr_center"]
    fig, ax = plt.subplots()
    try:
        _vem_axes(ax, frame, acceptable_inequity_level=0.1,
                  conf_level=0.9, center=center, smart_zoom=False)
        moving_median = next(line for line in ax.lines
                             if line.get_label().startswith("Moving median"))
        assert moving_median.get_ydata() == pytest.approx(
            frame["moving_median"].to_numpy() + center)
        assert sorted(line.get_ydata()[0] for line in ax.lines
                      if line.get_label().startswith(("Median ASR",
                                                      "Acceptable inequity",
                                                      "_nolegend_"))
                      ) == pytest.approx([center - 0.1, center, center + 0.1])
        assert ax.get_ylabel() == "ASR"
        assert ax.get_ylim() == pytest.approx((center - 0.3, center + 0.3))
    finally:
        plt.close(fig)


def test_display_center_is_recovered_from_the_frame(vem_result):
    """Callers that only hand over the frame still get an ASR-unit chart."""
    frame = vem_result["dataframe"]
    assert _ve_display_center(frame) == pytest.approx(vem_result["asr_center"])
    # No 'asr_raw' to fall back on: the frame is already centred, so the
    # chart stays on the centred scale rather than inventing an offset.
    assert _ve_display_center(frame.drop(columns="asr_raw")) == pytest.approx(0.0)
    assert _ve_display_center(frame, center=np.nan) == pytest.approx(
        vem_result["asr_center"])

# --------------------------------------------------------------------------
# VEM smart zoom
# --------------------------------------------------------------------------

def test_smart_zoom_trims_both_axes_to_the_bulk_of_the_points(vem_result):
    frame = vem_result["dataframe"]
    center = vem_result["asr_center"]
    xlim, ylim = _ve_smart_limits(frame, center, acceptable_inequity_level=0.1)

    mv = frame["mv"].to_numpy()
    asr = frame["asr_raw"].to_numpy()
    # Inside the full data range: the value tails are cut off.
    assert mv.min() < xlim[0] < xlim[1] < mv.max()
    assert asr.min() < ylim[0] < ylim[1] < asr.max()
    # Most points survive the trim, only the tails are dropped.
    shown = ((mv >= xlim[0]) & (mv <= xlim[1])
             & (asr >= ylim[0]) & (asr <= ylim[1]))
    assert shown.mean() > 0.9
    # The x-axis is cut much harder than the y-axis, as the value
    # distribution is far more skewed than the ratios.
    assert (xlim[1] - xlim[0]) / (mv.max() - mv.min()) < 0.6
    assert (ylim[1] - ylim[0]) / (asr.max() - asr.min()) > 0.6


def test_smart_zoom_never_hides_the_moving_median_or_its_bands(vem_result):
    """Zooming may drop scatter points, never the statistic being read."""
    frame = vem_result["dataframe"]
    center = vem_result["asr_center"]
    xlim, ylim = _ve_smart_limits(frame, center, acceptable_inequity_level=0.1)

    inside = (frame["mv"] >= xlim[0]) & (frame["mv"] <= xlim[1])
    for column in ("moving_median", "median_ci_lower", "median_ci_upper"):
        shifted = frame.loc[inside, column].to_numpy() + center
        assert ylim[0] <= shifted.min() and shifted.max() <= ylim[1]
    # ... including the reference lines the eye reads them against.
    assert ylim[0] <= center - 0.1 and center + 0.1 <= ylim[1]


def test_smart_zoom_is_on_by_default_in_both_charts(vem_result):
    frame = vem_result["dataframe"]
    center = vem_result["asr_center"]
    _, smart_ylim = _ve_smart_limits(frame, center, acceptable_inequity_level=0.1)

    fig = ve_scat_median_plot(frame, "t", None, acceptable_inequity_level=0.1,
                              conf_level=0.9, center=center)
    assert fig.layout.yaxis.range == pytest.approx(smart_ylim)
    # A log x-axis takes its range in log10 units.
    assert fig.layout.xaxis.range[0] > 1
    assert fig.layout.xaxis.title.text == "Value (log scale)"

    fig_mpl, ax = plt.subplots()
    try:
        _vem_axes(ax, frame, acceptable_inequity_level=0.1,
                  conf_level=0.9, center=center)
        assert ax.get_ylim() == pytest.approx(smart_ylim)
        assert ax.get_xlim()[0] > frame["mv"].min()
        assert ax.get_xlim()[1] < frame["mv"].max()
    finally:
        plt.close(fig_mpl)


def test_an_explicit_zoom_y_still_wins_over_smart_zoom(vem_result):
    frame = vem_result["dataframe"]
    center = vem_result["asr_center"]
    fig, ax = plt.subplots()
    try:
        _vem_axes(ax, frame, zoom_y=(0.5, 1.5), center=center)
        assert ax.get_ylim() == pytest.approx((0.5, 1.5))
    finally:
        plt.close(fig)


def test_smart_zoom_off_falls_back_to_the_fixed_window(vem_result):
    frame = vem_result["dataframe"]
    center = vem_result["asr_center"]
    fig, ax = plt.subplots()
    try:
        _vem_axes(ax, frame, zoom_y=True, center=center, smart_zoom=False)
        assert ax.get_ylim() == pytest.approx((center - 0.3, center + 0.3))
        assert ax.get_xlim()[0] < frame["mv"].min()
    finally:
        plt.close(fig)


def test_smart_zoom_copes_with_a_linear_x_axis_and_flat_ratios():
    """No usable spread must not blow up the limit maths."""
    flat = pd.DataFrame({
        "mv": np.linspace(100.0, 1000.0, 40),
        "asr_raw": np.full(40, 1.0),
        "asr": np.zeros(40),
        "moving_median": np.zeros(40),
        "out_of_bounds": np.zeros(40, dtype=int),
    })
    xlim, ylim = _ve_smart_limits(flat, 1.0, acceptable_inequity_level=0.1,
                                  x_scale="linear")
    assert xlim[0] < xlim[1]
    assert ylim[0] < 0.9 and ylim[1] > 1.1

# --------------------------------------------------------------------------
# Gauge row layout and include_vem
# --------------------------------------------------------------------------

def _gauge_axes(fig):
    """The gauge subplots, in draw order.

    They carry no axis labels -- and neither does the secondary-indicator
    strip, which is why that one is excluded by name.
    """
    return [ax for ax in fig.axes
            if not ax.get_ylabel() and ax.get_label() != 'secondary_indicators']


def test_gauge_row_is_centred_on_the_page(ve_df):
    """The charts below need a wide left margin; the gauges must not inherit it."""
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        boxes = [ax.get_position() for ax in _gauge_axes(fig)]
        assert boxes[0].x0 == pytest.approx(1 - boxes[-1].x1, abs=0.01)
    finally:
        plt.close(fig)


def test_gauges_sit_above_the_charts_with_room_for_the_title(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        gauge_top = max(ax.get_position().y1 for ax in _gauge_axes(fig))
        chart_top = fig.axes[-2].get_position().y1
        assert gauge_top < 0.93                      # clear of the suptitle
        assert gauge_top - chart_top > 0.2           # gauges are not squeezed
    finally:
        plt.close(fig)


def test_include_vem_false_drops_the_vem_gauge_and_recentres_the_rest(ve_df):
    with_vem = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    without_vem = vertical_equity_panel(ve_df, "assess", "sale", show=False,
                                        include_vem=False)
    try:
        four = [ax.get_position() for ax in _gauge_axes(with_vem)]
        three = [ax.get_position() for ax in _gauge_axes(without_vem)]
        assert len(four) == 4 and len(three) == 3
        assert len(without_vem.axes) == len(with_vem.axes) - 1
        # Same dial size, still centred.
        assert three[0].width == pytest.approx(four[0].width, rel=0.02)
        assert three[0].x0 == pytest.approx(1 - three[-1].x1, abs=0.01)
        assert three[0].x0 > four[0].x0
    finally:
        plt.close(with_vem)
        plt.close(without_vem)


def test_include_vem_false_strips_the_verdict_from_the_vem_chart(ve_df):
    """The chart stays, but nothing on it reports a VEM verdict."""
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False,
                                include_vem=False)
    try:
        vem_ax = fig.axes[-1]
        labels = [text.get_text() for text in vem_ax.get_legend().get_texts()]
        assert not any("local median)" in label for label in labels)
        assert not any("Acceptable inequity" in label for label in labels)
        # The moving median and its band are what the chart is for.
        assert any(label.startswith("Moving median") for label in labels)
        assert any("CI for local median" in label for label in labels)
        # One plain scatter, not the green/red split.
        assert len(vem_ax.collections) == 2          # scatter + CI fill_between
    finally:
        plt.close(fig)


def test_vem_reports_the_smoothing_it_used(ve_df, vem_result):
    assert vem_result["window_size"] == 120
    assert vem_result["quantiles"] is None

    by_quantile = vem(
        pd.DataFrame({"sale": ve_df["sale"],
                      "asr": ve_df["assess"] / ve_df["sale"]}),
        market_value_proxy="sale", quantiles=10, conf_level=0.9,
    )
    assert by_quantile["quantiles"] == 10
    assert by_quantile["window_size"] is None

# --------------------------------------------------------------------------
# CoD read-out and x-axis ticks
# --------------------------------------------------------------------------

def test_cod_is_reported_under_the_vei_gauge():
    spec = _ve_gauge_specs(vei_metrics={"vei": -3.0, "conclusion": "Acceptable",
                                        "acceptable_range": 10},
                           cod_value=0.0834)[2]
    assert spec["detail"] == ("COD: 8.3%",)


def test_cod_joins_a_differing_vei_verdict_as_a_second_line():
    spec = _ve_gauge_specs(vei_metrics={"vei": -38.3, "conclusion": "Acceptable",
                                        "acceptable_range": 10},
                           cod_value=0.0834)[2]
    assert spec["detail"] == ("VEI test: Acceptable", "COD: 8.3%")


def test_no_cod_no_line():
    spec = _ve_gauge_specs(vei_metrics={"vei": -3.0, "conclusion": "Acceptable",
                                        "acceptable_range": 10})[2]
    assert spec["detail"] == ()


def test_panel_computes_the_cod_it_shows(ve_df):
    """The read-out must match the CoD of the ratios the panel was given."""
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        asr = (ve_df["assess"] / ve_df["sale"]).to_numpy()
        expected = f"COD: {I.cod(asr) * 100:.1f}%"
        vei_ax = _gauge_axes(fig)[2]
        assert expected in [text.get_text() for text in vei_ax.texts]
    finally:
        plt.close(fig)


def test_a_supplied_cod_is_used_as_is(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False,
                                cod_value=0.123)
    try:
        vei_ax = _gauge_axes(fig)[2]
        assert "COD: 12.3%" in [text.get_text() for text in vei_ax.texts]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("include_vem", [True, False])
def test_vem_chart_keeps_several_value_reference_points(ve_df, include_vem):
    """A single decade of value range must not collapse to two ticks."""
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False,
                                include_vem=include_vem)
    try:
        vem_ax = fig.axes[-1]
        low, high = vem_ax.get_xlim()
        visible = [tick for tick in vem_ax.get_xticks() if low <= tick <= high]
        assert 4 <= len(visible) <= 8
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# Chart titles
# --------------------------------------------------------------------------

def test_the_vei_title_carries_the_indicator_and_its_verdict():
    assert _vei_chart_title({"vei": -35.0, "conclusion": "Regressivity"}) == (
        "Vertical Equity Indicator: -35.0%, Regressivity")
    assert _vei_chart_title({"vei": 4.2, "conclusion": "Acceptable"}) == (
        "Vertical Equity Indicator: +4.2%, Acceptable")


def test_the_vei_title_drops_what_it_does_not_have():
    # vei_analysis reports 0 when it bails out on sample size, which would
    # read as a real result.
    assert _vei_chart_title({"vei": 0, "conclusion": "Sample size too small"}) == (
        "Vertical Equity Indicator: Sample size too small")
    assert _vei_chart_title({"vei": -3.0}) == "Vertical Equity Indicator: -3.0%"
    assert _vei_chart_title({}) == "Vertical Equity Indicator"
    assert _vei_chart_title(None) == "Vertical Equity Indicator"


def test_the_vem_title_carries_the_verdict_with_severity_and_prevalence():
    result = {"Inequity_conclusion": "Regressivity", "Inequity_severity": 15.76,
              "Inequity_prevalence": 44.0, "window_size": 120}
    assert _vem_chart_title(result) == (
        "Vertical Equity: Regressivity, severity: 15.8%, prevalence: 44.0%")


def test_the_vem_title_describes_the_chart_when_it_reports_no_verdict():
    """include_vem=False shows no VEM verdict, so the title says what is drawn.

    The smoothing is never in the title: it describes the moving median
    line, so it rides on that legend entry instead.
    """
    result = {"Inequity_conclusion": "Regressivity", "Inequity_severity": 15.76,
              "Inequity_prevalence": 44.0, "window_size": 120}
    assert _vem_chart_title(result, include_verdict=False) == "Moving Median of ASR"
    assert _vem_chart_title({"quantiles": 10}, include_verdict=False) == (
        "Moving Median of ASR")


def test_the_vem_title_falls_back_when_there_is_no_conclusion():
    assert _vem_chart_title({"window_size": 120}) == "Moving Median of ASR"
    assert _vem_chart_title({}) == "Moving Median of ASR"
    assert _vem_chart_title(None) == "Moving Median of ASR"


def test_the_smoothing_label_names_the_window_or_the_quantiles():
    assert _vem_smoothing_label({"window_size": 120}) == "120"
    assert _vem_smoothing_label({"quantiles": 10}) == "10 quantiles"
    assert _vem_smoothing_label({}) == ""
    assert _vem_smoothing_label(None) == ""


def test_the_panel_titles_both_charts_from_the_results(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        assert fig.axes[-2].get_title().startswith("Vertical Equity Indicator: ")
        assert fig.axes[-1].get_title().startswith("Vertical Equity: ")
        # The smoothing moves to the legend, so it is still on the chart.
        labels = [text.get_text() for text in fig.axes[-1].get_legend().get_texts()]
        assert any("(window " in label for label in labels)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("include_vem", [True, False])
def test_the_smoothing_is_always_in_the_legend_never_the_title(ve_df, include_vem):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False,
                                include_vem=include_vem)
    try:
        assert "(window " not in fig.axes[-1].get_title()
        labels = [text.get_text() for text in fig.axes[-1].get_legend().get_texts()]
        assert any(label.startswith("Moving median") and "(window " in label
                   for label in labels)
    finally:
        plt.close(fig)


def test_without_the_vem_verdict_the_title_describes_the_chart(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False,
                                include_vem=False)
    try:
        assert fig.axes[-1].get_title() == "Moving Median of ASR"
    finally:
        plt.close(fig)


def test_an_acceptable_vem_title_carries_the_conclusion_alone():
    """Severity and prevalence are nil when there is no inequity to report."""
    result = {"Inequity_conclusion": "acceptable",
              "Inequity_severity": 0.0, "Inequity_prevalence": 0.0,
              "window_size": 120}
    assert _vem_chart_title(result) == "Vertical Equity: acceptable"
    # An inconclusive or breaching result still reports both.
    assert "severity: " in _vem_chart_title(
        {**result, "Inequity_conclusion": "Inconclusive"})


# --------------------------------------------------------------------------
# VEM legend
# --------------------------------------------------------------------------

def test_the_vem_legend_is_drawn_smaller_than_the_tick_labels(vem_result):
    """Six entries at tick size covered a quarter of the plot."""
    fig, ax = plt.subplots()
    try:
        _vem_axes(ax, vem_result["dataframe"], conf_level=0.9,
                  center=vem_result["asr_center"], tick_fontsize=10)
        sizes = [text.get_fontsize() for text in ax.get_legend().get_texts()]
        assert sizes == [pytest.approx(6.0)] * len(sizes)
    finally:
        plt.close(fig)


def test_the_vem_legend_size_can_be_overridden(vem_result):
    fig, ax = plt.subplots()
    try:
        _vem_axes(ax, vem_result["dataframe"], conf_level=0.9,
                  center=vem_result["asr_center"], legend_fontsize=9)
        sizes = [text.get_fontsize() for text in ax.get_legend().get_texts()]
        assert sizes == [pytest.approx(9.0)] * len(sizes)
    finally:
        plt.close(fig)


def test_the_vem_legend_stays_clear_of_most_of_the_plot(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    try:
        fig.canvas.draw()
        vem_ax = fig.axes[-1]
        legend = vem_ax.get_legend().get_window_extent()
        axes_box = vem_ax.get_window_extent()
        share = (legend.width * legend.height) / (axes_box.width * axes_box.height)
        assert share < 0.10
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# mvp_sale_price_share
# --------------------------------------------------------------------------

def _gauge_texts(fig):
    return [t.get_text() for t in fig.findobj(match=lambda o: isinstance(o, plt.Text))]


def test_the_panel_default_share_draws_the_historical_proxy(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    explicit = vertical_equity_panel(ve_df, "assess", "sale",
                                     mvp_sale_price_share=0.5, show=False)
    try:
        assert _gauge_texts(fig) == _gauge_texts(explicit)
    finally:
        plt.close(fig)
        plt.close(explicit)


def test_the_panel_share_moves_the_prb_and_vei_gauges(ve_df):
    fig = vertical_equity_panel(ve_df, "assess", "sale", show=False)
    leaned = vertical_equity_panel(ve_df, "assess", "sale",
                                   mvp_sale_price_share=0.1, show=False)
    try:
        # The proxy feeds the PRB regression and the VEI grouping, so the
        # dials have to disagree once it is reweighted.
        assert _gauge_texts(fig) != _gauge_texts(leaned)
    finally:
        plt.close(fig)
        plt.close(leaned)


def test_the_panel_share_is_ignored_when_a_proxy_column_is_given(ve_df):
    data = ve_df.copy()
    data["proxy"] = data["sale"]
    fig = vertical_equity_panel(data, "assess", "sale",
                                mkt_value_proxy_header="proxy",
                                prb_mvp_header="proxy", show=False)
    leaned = vertical_equity_panel(data, "assess", "sale",
                                   mkt_value_proxy_header="proxy",
                                   prb_mvp_header="proxy",
                                   mvp_sale_price_share=0.1, show=False)
    try:
        assert _gauge_texts(fig) == _gauge_texts(leaned)
    finally:
        plt.close(fig)
        plt.close(leaned)
