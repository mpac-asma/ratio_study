"""Tests for `bootstrap_resample_pct` and `print_ci_settings`."""
import numpy as np
import pandas as pd
import pytest

import ratio_study as rs
from ratio_study.confint import (normalize_ci_args, describe_ci_settings,
                                 BOOTSTRAP_MIN_RESAMPLE_SIZE)

MV = 'time_adjusted_sale_price'
AV = 'assessment_value_new'


@pytest.fixture(scope='module')
def sales():
    return rs.load_sample_dataset()


def _ci_width(text):
    low, high = str(text).strip('()').split(',')
    return float(high) - float(low)


# --- resolving the percentage ---------------------------------------------

def test_the_default_is_the_whole_sample():
    resolved, _ = normalize_ci_args()
    assert resolved['bootstrap_resample_pct'] == 1.0


@pytest.mark.parametrize('given, expected', [(0.2, 0.2), (20, 0.2), (1, 1.0), (100, 1.0)])
def test_a_fraction_or_a_percentage_both_work(given, expected):
    resolved, _ = normalize_ci_args(bootstrap_args={'resample_pct': given})
    assert resolved['bootstrap_resample_pct'] == pytest.approx(expected)


@pytest.mark.parametrize('alias', ['resample_pct', 'sample_pct', 'pct', 'resample_percent'])
def test_the_short_forms_are_accepted(alias):
    resolved, _ = normalize_ci_args(bootstrap_args={alias: 0.3})
    assert resolved['bootstrap_resample_pct'] == pytest.approx(0.3)


@pytest.mark.parametrize('bad', [0, -0.5, 'x', None])
def test_a_nonsense_percentage_falls_back_and_says_so(bad):
    resolved, messages = normalize_ci_args(bootstrap_args={'resample_pct': bad})
    assert resolved['bootstrap_resample_pct'] == 1.0
    assert any('bootstrap_resample_pct' in m for m in messages)


# --- what it does to the interval -----------------------------------------

def test_a_smaller_resample_widens_the_interval(sales):
    # An m-out-of-n bootstrap answers "how sure would I be on a smaller
    # sample?", so the interval must get wider as m falls.
    widths = []
    for pct in (1.0, 0.25, 0.05):
        row = rs.ratios(sales, assessment_header=AV, market_value_header=MV,
                        indicators=['Median'], ci_mode='bootstrap', ci_seed=11,
                        print_ci_settings=False,
                        bootstrap_args={'resamplings': 200, 'resample_pct': pct}).iloc[0]
        widths.append(_ci_width(row['Median_ci']))
    assert widths[0] < widths[1] < widths[2]


def test_the_point_estimate_is_untouched(sales):
    estimates = {
        rs.ratios(sales, assessment_header=AV, market_value_header=MV,
                  indicators=['Median'], ci_seed=11, print_ci_settings=False,
                  bootstrap_args={'resamplings': 100, 'resample_pct': pct}).iloc[0]['Median']
        for pct in (1.0, 0.5, 0.1)
    }
    assert len(estimates) == 1        # resampling never moves the sample statistic


def test_the_floor_keeps_a_resample_usable(sales):
    # n=50 at 20% would draw 10; the floor lifts it to 20. 40% also draws 20,
    # so with one seed the two runs have to agree exactly.
    small = sales.head(50)
    runs = [
        rs.ratios(small, assessment_header=AV, market_value_header=MV,
                  indicators=['Median'], ci_seed=3, print_ci_settings=False,
                  bootstrap_args={'resamplings': 150, 'resample_pct': pct}).iloc[0]['Median_ci']
        for pct in (0.2, 0.4)
    ]
    assert runs[0] == runs[1]
    assert BOOTSTRAP_MIN_RESAMPLE_SIZE == 20


def test_the_floor_never_exceeds_the_sample(sales):
    # 12 sales is below the floor; the draw has to fall back to n, not 20.
    tiny = sales.head(12)
    result = rs.ratios(tiny, assessment_header=AV, market_value_header=MV,
                       indicators=['Median'], ci_seed=3, print_ci_settings=False,
                       bootstrap_args={'resamplings': 50, 'resample_pct': 0.1,
                                       'min_count_threshold': 4})
    assert len(result) == 1           # ran without raising


def test_ratios_report_accepts_the_option(sales, tmp_path):
    rs.ratios_report(sales, assessment_header=AV, market_value_header=MV,
                     export_directory=tmp_path, export_in_compacted_file='',
                     print_ci_settings=False, return_dict=False,
                     bootstrap_args={'resamplings': 100, 'resample_pct': 0.3})
    assert list(tmp_path.glob('*.pdf'))


# --- the settings summary --------------------------------------------------

def test_the_summary_lists_every_setting_that_governs_an_interval():
    resolved, _ = normalize_ci_args(ci_mode='bootstrap')
    text = describe_ci_settings(resolved, count=1000)
    for label in ('Mode', 'Level', 'Indicators', 'Overall intervals',
                  'Per-group intervals', 'Minimum sales per row',
                  'Method', 'Resamplings', 'Resample size', 'Seed'):
        assert label in text


def test_values_the_caller_set_are_marked_and_defaults_are_not():
    resolved, _ = normalize_ci_args(ci_mode='bootstrap', ci_conf_level=95,
                                     bootstrap_args={'method': 'percentile'})
    lines = {line.split(':')[0].strip(): line for line in
             describe_ci_settings(resolved, count=1000).splitlines() if ':' in line}
    assert '<-- set' in lines['Level']
    assert '<-- set' in lines['Method']
    assert '<-- set' not in lines['Resamplings']       # untouched default


def test_the_summary_reports_the_actual_draw_including_the_floor():
    resolved, _ = normalize_ci_args(ci_mode='bootstrap', bootstrap_args={'resample_pct': 0.2})
    assert '20% of 1,000 = 200 sales per resample' in describe_ci_settings(resolved, count=1000)
    # 20% of 50 is 10, which the floor lifts to 20 -- and the summary says so.
    assert 'floor applied' in describe_ci_settings(resolved, count=50)


def test_distribution_mode_hides_the_bootstrap_only_settings():
    resolved, _ = normalize_ci_args(ci_mode='distribution')
    text = describe_ci_settings(resolved, count=100)
    assert 'Appendix D' in text
    assert 'Resamplings' not in text and 'Seed' not in text


def test_the_summary_says_so_when_intervals_are_off():
    resolved, _ = normalize_ci_args(confidence_intervals_aggs=False)
    assert describe_ci_settings(resolved) == 'Confidence intervals: off.'


def test_an_unseeded_run_is_flagged_as_irreproducible():
    resolved, _ = normalize_ci_args(ci_mode='bootstrap')
    assert 'vary slightly between runs' in describe_ci_settings(resolved, count=100)


# --- printing from ratios() ------------------------------------------------

def test_ratios_prints_the_settings_by_default(sales, capsys):
    rs.ratios(sales, assessment_header=AV, market_value_header=MV, indicators=['Median'])
    assert 'Confidence interval settings' in capsys.readouterr().out


def test_printing_can_be_switched_off(sales, capsys):
    rs.ratios(sales, assessment_header=AV, market_value_header=MV,
              indicators=['Median'], print_ci_settings=False)
    assert 'Confidence interval settings' not in capsys.readouterr().out


def test_the_printed_count_is_the_cleaned_sample_not_the_input(sales, capsys):
    # Rows with a non-positive value are dropped before the bootstrap sees
    # them, so the reported draw has to be based on what survives.
    dirty = sales.copy()
    dirty.loc[dirty.index[:100], AV] = 0
    rs.ratios(dirty, assessment_header=AV, market_value_header=MV,
              indicators=['Median'], ci_mode='bootstrap')
    assert f'{len(sales) - 100:,}' in capsys.readouterr().out


# --- bootstrap intervals for MKI, RRI, VEI and VEM ------------------------

def test_vei_and_vem_are_accepted_for_the_bootstrap():
    # They used to be refused with "no bootstrap confidence interval is
    # available for ['VEI', 'VEM']; they were dropped".
    resolved, messages = normalize_ci_args(
        ci_mode='bootstrap', ci_indicators=['VEI', 'VEM', 'MKI', 'RRI'],
        study_indicators=['VEI', 'VEM', 'MKI', 'RRI'])
    assert resolved['bootstrap_indicators'] == ['VEI', 'VEM', 'MKI', 'RRI']
    assert not any('no bootstrap confidence interval' in m for m in messages)


def test_indicators_map_to_the_columns_that_carry_their_interval():
    from ratio_study.confint import bootstrap_ci_result_columns
    assert bootstrap_ci_result_columns(['Median', 'MKI', 'RRI', 'VEI', 'VEM']) == [
        'Median', 'Modified_Kakwani_Index', 'Reranking_Index', 'VEI',
        'Inequity_severity', 'Inequity_prevalence']


def test_mki_rri_vei_and_vem_get_bootstrap_intervals(sales):
    # MKI and RRI were bootstrapped but filed under their long names, so the
    # table, which asked for 'MKI_ci' / 'RRI_ci', silently showed neither.
    wanted = ['Median', 'VEI', 'VEM', 'MKI', 'RRI']
    with _no_dropped_ci_warning():
        out = rs.ratios(sales.head(400), assessment_header=AV, market_value_header=MV,
                        indicators=wanted, ci_indicators=wanted,
                        ci_mode='bootstrap', ci_conf_level=90, ci_seed=7,
                        print_ci_settings=False,
                        bootstrap_args={'resamplings': 40})
    row = out.iloc[0]
    for col in ('Median_ci', 'VEI_ci', 'MKI_ci', 'RRI_ci',
                'Inequity_severity_ci', 'Inequity_prevalence_ci'):
        assert col in out.columns, col
        low, high = (float(x.strip(' %')) for x in str(row[col]).strip('()').split(','))
        assert low <= high, col
    # Each interval sits next to its indicator, not at the end of the table.
    cols = list(out.columns)
    assert cols.index('VEI_ci') == cols.index('VEI') + 1
    assert cols.index('RRI_ci') == cols.index('RRI') + 1
    assert cols.index('MKI_ci') == cols.index('MKI') + 1


def test_the_bootstrap_skips_labels_and_unrequested_columns():
    # A resample returns 'VEI_conclusion' / 'Inequity_conclusion' strings and
    # by-products such as 'Weighted_Mean'; only what was asked for is
    # intervalled.
    from ratio_study.ratio_study import _bootstrap_analysis
    df = rs.load_sample_dataset().head(300)
    out = _bootstrap_analysis(df, assessment_header=AV, market_value_header=MV,
                              weights_header=None, confint_level=0.9,
                              bootstrap_method='percentile',
                              bootstrap_resamplings=30,
                              population_ratios={}, ci_only=True,
                              req_significance=0.05, indicators=['VEI', 'VEM', 'MKI'],
                              seed=3)
    assert set(out) == {'VEI_ci', 'Inequity_severity_ci',
                        'Inequity_prevalence_ci', 'Modified_Kakwani_Index_ci'}


class _no_dropped_ci_warning:
    """Fail if `ratios` warns that a requested interval was dropped."""
    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        warnings.simplefilter('always')
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        dropped = [str(w.message) for w in self._log
                   if 'confidence interval is available' in str(w.message)]
        assert not dropped, dropped
        return False


def test_an_interval_through_zero_is_not_a_finding_for_vei_vem_or_rri(sales):
    # Zero is the ideal VEI, VEM severity / prevalence and RRI. Once these
    # got bootstrap intervals, the generic "could be zero" fallback marked
    # them as findings on a perfectly equitable sample.
    wanted = ['Median', 'VEI', 'VEM', 'RRI']
    out = rs.ratios(sales.head(400), assessment_header=AV, market_value_header=MV,
                    descriptive=True, indicators=wanted, ci_indicators=wanted,
                    ci_mode='bootstrap', ci_seed=7, print_ci_settings=False,
                    bootstrap_args={'resamplings': 40})
    comments = ' '.join(out.filter(like='Comments').astype(str).iloc[0])
    assert 'could be zero' not in comments
    for col in ('VEI', 'Inequity_severity', 'Inequity_prevalence', 'RRI'):
        assert chr(8869) not in str(out.iloc[0][col]), col
