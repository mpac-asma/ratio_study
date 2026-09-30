"""Tests for the chart display options and the VEI bucket-selection diagnostic.

Covers: the secondary-indicator strip on the panel, `compare_all_buckets`,
the Appendix E.5 ranking note, scatter/facet zoom, and boxplot value labels.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import ratio_study as rs
from ratio_study import graphs

MV = 'time_adjusted_sale_price'
AV = 'assessment_value_new'


@pytest.fixture(scope='module')
def sales():
    return rs.load_sample_dataset()


# --- VEI: compare_all_buckets ---------------------------------------------

def test_the_default_compares_the_first_and_last_groups(sales, capsys):
    result = rs.vei_analysis(sales[AV] / sales[MV], sales[MV])
    assert result['group_compared_low'] == 1
    assert result['group_compared_high'] == result['n_groups']
    assert result['compare_all_buckets'] is False


def test_compare_all_buckets_reports_the_extreme_groups(sales, capsys):
    result = rs.vei_analysis(sales[AV] / sales[MV], sales[MV], compare_all_buckets=True)
    out = capsys.readouterr().out
    assert 'compare_all_buckets is ON' in out
    assert 'Lowest median ratio' in out and 'Highest median ratio' in out
    assert 'diagnostic, not the IAAO standard acceptance test' in out
    assert result['compare_all_buckets'] is True


def test_it_picks_the_widest_gap_when_an_inner_group_is_the_extreme():
    # Ratios engineered so the *middle* band holds the highest median: the
    # default endpoints understate the inequity, compare_all_buckets does not.
    rng = np.random.default_rng(0)
    price = np.linspace(100_000, 900_000, 600)
    asr = np.where((price > 350_000) & (price < 550_000), 1.30, 1.00)
    asr = asr + rng.normal(0, 0.01, price.size)

    default = rs.vei_analysis(asr, price, print_settings=False)
    widest = rs.vei_analysis(asr, price, compare_all_buckets=True, print_settings=False)

    medians = [g['median_ratio'] for g in widest['pg_stats']]
    assert 1 < int(np.argmax(medians)) + 1 < len(medians)      # an inner group leads
    assert abs(widest['vei']) > abs(default['vei'])            # and it finds more


def test_value_order_is_preserved_so_the_sign_still_means_regressive():
    rng = np.random.default_rng(1)
    price = np.linspace(100_000, 900_000, 600)
    asr = 1.25 - 0.4 * (price - price.min()) / np.ptp(price) + rng.normal(0, 0.01, price.size)
    widest = rs.vei_analysis(asr, price, compare_all_buckets=True, print_settings=False)
    assert widest['group_compared_low'] < widest['group_compared_high']
    assert widest['vei'] < 0 and widest['direction'] == 'Regressivity'


def test_two_groups_make_the_option_a_no_op(sales):
    # 20-50 sales means halves; there is no third pair to prefer.
    small = sales.head(40)
    a = rs.vei_analysis(small[AV] / small[MV], small[MV], print_settings=False)
    b = rs.vei_analysis(small[AV] / small[MV], small[MV],
                        compare_all_buckets=True, print_settings=False)
    assert a['vei'] == b['vei']


def test_print_settings_false_keeps_it_quiet(sales, capsys):
    rs.vei_analysis(sales[AV] / sales[MV], sales[MV], print_settings=False)
    assert capsys.readouterr().out == ''


def test_ratios_reports_the_ranking_method_once(sales, capsys):
    rs.ratios(sales, assessment_header=AV, market_value_header=MV,
              indicators=['Median', 'VEI'], vei_percentile_method='R6')
    out = capsys.readouterr().out
    # `ratios` prints this once from its own consolidated note rather than
    # from inside `vei_analysis`, which runs once per group and once per
    # bootstrap resample and would otherwise repeat it.
    assert out.count('VEI quantile method') == 1
    assert 'R6' in out


def test_ratios_flags_the_diagnostic_mode(sales, capsys):
    rs.ratios(sales, assessment_header=AV, market_value_header=MV,
              indicators=['Median', 'VEI'], vei_compare_all_buckets=True)
    assert 'compare_all_buckets ON' in capsys.readouterr().out


# --- VEI chart: the Appendix E.5 note -------------------------------------

def _note_texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts if 'ranking' in t.get_text()]


def test_the_standalone_boxplot_records_the_ranking_method(sales):
    result = rs.vei_analysis(sales[AV] / sales[MV], sales[MV], print_settings=False)
    fig = rs.vei_boxplot(sales[AV] / sales[MV], sales[MV], result)
    try:
        notes = _note_texts(fig)
        assert notes and 'R7 ranking' in notes[0]
    finally:
        plt.close(fig)


def test_the_panel_records_it_too(sales):
    fig = rs.vertical_equity_panel(sales, assessment_header=AV,
                                   market_value_header=MV, show=False)
    try:
        assert any('ranking' in n for n in _note_texts(fig))
    finally:
        plt.close(fig)


# --- the panel's secondary-indicator strip --------------------------------

def test_the_strip_carries_the_five_appendix_f_tests(sales):
    fig = rs.vertical_equity_panel(sales, assessment_header=AV,
                                   market_value_header=MV, show=False)
    try:
        strip = next(ax for ax in fig.axes if ax.get_label() == 'secondary_indicators')
        labels = [t.get_text() for t in strip.texts]
        for expected in ('MKI', 'RRI', 'Mann-Whitney', 'Kruskal-Wallis', 'Spearman'):
            assert expected in labels
    finally:
        plt.close(fig)


def test_the_gini_indices_are_not_graded():
    # Appendix F gives MKI and RRI no compliance range, so they must not be
    # coloured as pass/fail the way the gauges are.
    rng = np.random.default_rng(2)
    price = np.linspace(100_000, 900_000, 400)
    asr = 1.0 + rng.normal(0, 0.05, price.size)
    vei = rs.vei_analysis(asr, price, print_settings=False)
    items = graphs._secondary_indicator_specs(
        asr, price, assessment=asr * price, market_value=price, vei_metrics=vei)
    graded = {i['label']: i['flag'] for i in items}
    assert graded['MKI'] is None and graded['RRI'] is None
    assert graded['Kruskal-Wallis'] in (True, False)


def test_a_strong_gradient_trips_the_hypothesis_tests():
    rng = np.random.default_rng(3)
    price = np.linspace(100_000, 900_000, 500)
    asr = 1.3 - 0.5 * (price - price.min()) / np.ptp(price) + rng.normal(0, 0.02, price.size)
    vei = rs.vei_analysis(asr, price, print_settings=False)
    items = {i['label']: i for i in graphs._secondary_indicator_specs(
        asr, price, assessment=asr * price, market_value=price, vei_metrics=vei)}
    assert items['Mann-Whitney']['flag'] is True
    assert items['Spearman']['flag'] is True
    assert 'regressive' in items['Spearman']['note']


def test_the_strip_degrades_to_na_without_the_value_columns():
    rng = np.random.default_rng(4)
    price = np.linspace(100_000, 900_000, 300)
    asr = 1.0 + rng.normal(0, 0.05, price.size)
    items = {i['label']: i for i in graphs._secondary_indicator_specs(asr, price)}
    assert items['MKI']['value'] == 'n/a' and items['RRI']['value'] == 'n/a'
    assert items['Spearman']['value'] != 'n/a'       # this one needs no value columns


# --- scatter / facet: zoom and bands --------------------------------------

def test_the_scatter_no_longer_draws_acceptance_bands_by_default(sales):
    fig = rs.asr_scatter(sales, assessment_header=AV, market_value_header=MV,
                         x=MV, show=False)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert not any('Acceptable level' in t for t in labels)
        assert any('Median' in t for t in labels)     # the median line stays
    finally:
        plt.close(fig)


def test_zoom_trims_both_axes_to_the_bulk(sales):
    zoomed = rs.asr_scatter(sales, assessment_header=AV, market_value_header=MV,
                            x=MV, log_x=True, zoom=True, show=False)
    full = rs.asr_scatter(sales, assessment_header=AV, market_value_header=MV,
                          x=MV, log_x=True, zoom=False, show=False)
    try:
        zx, zy = zoomed.axes[0].get_xlim(), zoomed.axes[0].get_ylim()
        fx, fy = full.axes[0].get_xlim(), full.axes[0].get_ylim()
        assert (zx[1] - zx[0]) < (fx[1] - fx[0])
        assert (zy[1] - zy[0]) < (fy[1] - fy[0])
    finally:
        plt.close(zoomed); plt.close(full)


def test_zoom_never_hides_the_median_line(sales):
    fig = rs.asr_scatter(sales, assessment_header=AV, market_value_header=MV,
                         x=MV, zoom=True, show=False)
    try:
        median = float((sales[AV] / sales[MV]).median())
        low, high = fig.axes[0].get_ylim()
        assert low < median < high
    finally:
        plt.close(fig)


def test_the_marginal_histogram_has_more_buckets_than_before(sales):
    fig = rs.asr_scatter(sales, assessment_header=AV, market_value_header=MV,
                         x=MV, marginals=True, show=False)
    try:
        # axes[1] is the top marginal; each bar is one bucket.
        top = fig.axes[1]
        assert len(top.patches) > 60
    finally:
        plt.close(fig)


def _facet_panels(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]


def test_the_facet_zoom_centres_each_panel_but_keeps_one_scale(sales):
    # Panels centred on a frame-wide median push a low-assessed stratum
    # against the bottom of its panel. Centring each one separately fixes
    # that; holding the span equal is what keeps them comparable, which is
    # the point of a grid.
    fig = rs.asr_facet(sales, assessment_header=AV, market_value_header=MV,
                       x=MV, col='quality_class', max_facets=4, zoom=True,
                       tile_min_records=0, show=False)
    try:
        limits = [ax.get_ylim() for ax in _facet_panels(fig)]
        spans = {round(high - low, 6) for low, high in limits}
        assert len(spans) == 1
        assert len({round(low, 6) for low, _ in limits}) > 1
    finally:
        plt.close(fig)


def test_the_facet_can_still_share_one_window(sales):
    fig = rs.asr_facet(sales, assessment_header=AV, market_value_header=MV,
                       x=MV, col='quality_class', max_facets=4, zoom='overall',
                       tile_min_records=0, show=False)
    try:
        assert len({ax.get_ylim() for ax in _facet_panels(fig)}) == 1
    finally:
        plt.close(fig)


def test_share_y_off_lets_each_facet_panel_size_itself(sales):
    fig = rs.asr_facet(sales, assessment_header=AV, market_value_header=MV,
                       x=MV, col='quality_class', max_facets=4, share_y=False,
                       tile_min_records=0, show=False)
    try:
        spans = {round(high - low, 6) for low, high in
                 (ax.get_ylim() for ax in _facet_panels(fig))}
        assert len(spans) > 1
    finally:
        plt.close(fig)


def test_the_facet_skips_panels_with_too_few_records(sales, capsys):
    # Appendix C: a handful of sales cannot support a conclusion. A panel
    # drawn from four of them still gets a trend line and reads like one.
    counts = sales['quality_class'].value_counts()
    threshold = int(counts.max())          # only the largest class survives
    fig = rs.asr_facet(sales, assessment_header=AV, market_value_header=MV,
                       x=MV, col='quality_class', tile_min_records=threshold,
                       show=False)
    try:
        assert len(_facet_panels(fig)) == 1
        out = capsys.readouterr().out
        assert 'not displayed' in out
        for value, n in counts.items():    # every dropped panel is named
            if n < threshold:
                assert f'quality_class={value}' in out
    finally:
        plt.close(fig)


def test_zero_draws_every_facet_panel_however_thin(sales, capsys):
    fig = rs.asr_facet(sales, assessment_header=AV, market_value_header=MV,
                       x=MV, col='quality_class', max_facets=99,
                       tile_min_records=0, show=False)
    try:
        assert len(_facet_panels(fig)) == sales['quality_class'].nunique()
        assert 'not displayed' not in capsys.readouterr().out
    finally:
        plt.close(fig)


def test_row_and_col_facets_pack_without_blank_gaps():
    # `row` x `col` facets on the full cross product; most combinations
    # here have no data ('fair' and 'good' each use only one 'cond').
    # Packing the real combinations, at the same width the two-dimensional
    # grid would have used, should still fill the first row completely --
    # 'fair's single panel slides into 'average's leftover slot instead of
    # opening a mostly-blank row of its own.
    rows = [{'quality': 'average', 'cond': cond, 'mv': 100_000 + cond * 1_000, 'asr': 1.0}
            for cond in [3, 4, 5, 6, 7, 8]]
    rows.append({'quality': 'fair', 'cond': 9, 'mv': 150_000, 'asr': 1.05})
    rows.append({'quality': 'good', 'cond': 5, 'mv': 160_000, 'asr': 0.95})
    df = pd.DataFrame(rows)

    fig = rs.asr_facet(df, market_value_header='mv', asr_col='asr', x='mv',
                       col='cond', row='quality', tile_min_records=0,
                       lowess=False, zoom=False, show=False)
    try:
        # Wrapping the real combinations creates exactly one panel each --
        # no cross-product blanks, not even hidden trailing ones -- unlike
        # the 3 x 7 = 21-cell grid the old row x col crossing produced.
        assert len(fig.axes) == 8
        assert len(_facet_panels(fig)) == 8
        assert all(ax.has_data() for ax in fig.axes)

        assert [ax.get_title() for ax in fig.axes] == [
            'average | 3', 'average | 4', 'average | 5', 'average | 6',
            'average | 7', 'average | 8', 'fair | 9', 'good | 5',
        ]
    finally:
        plt.close(fig)


def test_a_threshold_no_panel_can_meet_is_an_error(sales):
    with pytest.raises(ValueError, match='tile_min_records'):
        rs.asr_facet(sales, assessment_header=AV, market_value_header=MV,
                     x=MV, col='quality_class', tile_min_records=len(sales) + 1,
                     show=False)


def test_the_facet_drops_the_bands_too(sales):
    fig = rs.asr_facet(sales, assessment_header=AV, market_value_header=MV,
                       x=MV, col='quality_class', max_facets=4, show=False)
    try:
        assert not any('acceptable level' in t.get_text()
                       for t in fig.texts)
    finally:
        plt.close(fig)


# --- boxplot value labels --------------------------------------------------

def _annotations(fig):
    return [t.get_text() for t in fig.axes[0].texts]


def test_values_are_off_by_default(sales):
    fig = rs.asr_boxplot(sales, assessment_header=AV, market_value_header=MV,
                         by='property_type', min_count=30, show=False)
    try:
        assert not _annotations(fig)
    finally:
        plt.close(fig)


def test_show_values_labels_every_box_with_its_median(sales):
    fig = rs.asr_boxplot(sales, assessment_header=AV, market_value_header=MV,
                         by='property_type', min_count=30, show_values=True, show=False)
    try:
        frame = sales.assign(_r=sales[AV] / sales[MV])
        counts = frame['property_type'].value_counts()
        expected = int((counts >= 30).sum())
        labels = _annotations(fig)
        assert len(labels) == expected
        medians = frame.groupby('property_type')['_r'].median()
        for group in counts[counts >= 30].index:
            assert f'{medians[group]:.3f}' in labels
    finally:
        plt.close(fig)


def test_the_label_format_is_configurable(sales):
    fig = rs.asr_boxplot(sales, assessment_header=AV, market_value_header=MV,
                         by='property_type', min_count=30, show_values=True,
                         value_fmt='{:.1%}', show=False)
    try:
        assert all(label.endswith('%') for label in _annotations(fig))
    finally:
        plt.close(fig)


# --- results table ---------------------------------------------------------

def test_vei_pg_stats_never_reaches_either_output_frame(sales):
    for group_by in (None, 'property_type'):
        result = rs.ratios(sales, assessment_header=AV, market_value_header=MV,
                           group_by=group_by, return_dict=True, print_ci_settings=False)
        assert 'VEI_pg_stats' not in result['df_all_ratios'].columns
        assert 'VEI_pg_stats' not in result['df_all_ratios_formatted'].columns


def test_the_group_detail_is_still_available_in_the_results_dict(sales):
    result = rs.ratios(sales, assessment_header=AV, market_value_header=MV,
                       return_dict=True, print_ci_settings=False)
    assert result['dict_population_ratios']['ratios']['VEI_pg_stats']
