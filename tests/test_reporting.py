"""Tests for the reporting layer: ``_add_comments`` and ``compare_ratios``.

``_add_comments`` decides what a reader is told about every indicator --
which values are flagged, which get a comment, and (since `add_binaries`)
which are reported acceptable -- so the assertions here pin those
decisions rather than the exact wording around them. ``compare_ratios``
has to read those same formatted values back as numbers.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ratio_study.reporting import (
    _add_comments,
    compare_ratios,
    _acceptance_binaries,
    _small_sample_comment,
    _split_marker,
    _standard_band,
    _standard_for,
)

COMMENTS = 'Comments Ratio Study Standard metrics'
NOT_ACCEPTABLE = chr(8869)
SMALL_SAMPLE = chr(248)

STANDARDS = {'Median': (0.9, 1.1), 'PRB': (-0.05, 0.05), 'PRD': (0.98, 1.03),
             'CoD': (0.05, 0.15), 'Mkt_desc': ''}
FORMATS = {'Count': '{:,.0f}', 'CoD': '{:.2%}', 'VEI': '{:.2f}',
           'All others': '{:,.4f}'}
COLS = ['Level', 'Group', 'Count', 'Median', 'CoD', 'PRD', 'PRB',
        'VEI', 'VEI_conclusion']


def row(**overrides):
    """A acceptable overall row, with the given fields overridden."""
    base = dict(Level='Overall', Group='', Count=500, Median=1.00, CoD=0.09,
                PRD=1.01, PRB=0.01, VEI=-2.0, VEI_conclusion='Acceptable')
    base.update(overrides)
    return base


def comment(df, index=0):
    return df.loc[index, COMMENTS]


def run(rows, cols=COLS, **kwargs):
    kwargs.setdefault('skip', ['Level', 'Group'])
    return _add_comments(pd.DataFrame(rows), 0.05, STANDARDS, cols, FORMATS,
                         **kwargs)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value, number, marked", [
    ('-0.08*', -0.08, True),                 # PRB non-significance marker
    (f'0.043{NOT_ACCEPTABLE}', 0.043, True),
    ('0.043', 0.043, False),
    (0.043, 0.043, False),
    (5, 5.0, False),
    ('.5', 0.5, False),
    ('8.34%', 0.0834, False),                # a percent sign is not a marker
    ('-8.34%', -0.0834, False),              # ... and the sign survives it
    (f'13.50%{NOT_ACCEPTABLE}', 0.135, True),
    ('n/a', None, True),
    (None, None, False),
    (np.nan, None, False),
])
def test_split_marker(value, number, marked):
    got_number, got_marked = _split_marker(value)
    assert got_marked is marked
    if number is None:
        assert got_number is None
    else:
        assert got_number == pytest.approx(number)


def test_standard_for_maps_the_cod_variants():
    assert _standard_for('CoD_rsd', STANDARDS) == (0.05, 0.15)
    assert _standard_for('CoDw', STANDARDS) == (0.05, 0.15)
    assert _standard_for('CoD', STANDARDS) == (0.05, 0.15)
    assert _standard_for('RMSE', STANDARDS) is None


@pytest.mark.parametrize("standard", ['', None, (0.9,), 0.9, ('a', 'b')])
def test_standard_band_rejects_anything_that_is_not_a_numeric_pair(standard):
    """'Mkt_desc': '' and friends must not be read as a band."""
    assert _standard_band(standard, 'Mkt_desc', False) is None


def test_standard_band_gives_strata_more_latitude():
    # The population band is used as given.
    assert _standard_band((0.05, 0.15), 'CoD', False) == (0.05, 0.15)
    # A stratum CoD gets 5 more points.
    assert _standard_band((0.05, 0.15), 'CoD', True) == (0.05, 0.2)
    # A stratum median is compared with the overall level of assessment.
    assert _standard_band((0.9, 1.1), 'Median', True, overall_median=1.02) == (0.97, 1.07)
    # ... unless that comparison is turned off, or there is no overall row.
    assert _standard_band((0.9, 1.1), 'Median', True, overall_median=1.02,
                          compare_groups_median_to_overall_median=False) == (0.9, 1.15)
    assert _standard_band((0.9, 1.1), 'Median', True) == (0.9, 1.15)


@pytest.mark.parametrize("count, expected", [
    (500, None),
    (15, None),
    (14, 'Small sample'),
    (5, 'Small sample'),
    (4, 'Sample too small'),
    (None, None),
])
def test_small_sample_comment(count, expected):
    got = _small_sample_comment(count, 15)
    if expected is None:
        assert got is None
    else:
        assert got.startswith(expected)


# --------------------------------------------------------------------------
# Comments and markers
# --------------------------------------------------------------------------

def test_a_acceptable_row_gets_no_comment_and_no_markers():
    out = run([row()])
    assert comment(out) == ''
    assert out.loc[0, 'Median'] == '1.0000'
    assert NOT_ACCEPTABLE not in ''.join(out.loc[0].astype(str))


def test_a_value_outside_its_standard_is_commented_and_marked():
    out = run([row(PRD=1.09)])
    assert 'PRD suggests a review regarding assessment standards (0.98, 1.03)' in comment(out)
    assert out.loc[0, 'PRD'] == f'1.0900{NOT_ACCEPTABLE}'


def test_an_overlapping_confidence_interval_leaves_the_breach_unconfirmed():
    """The point estimate is out, but its CI still reaches the standard.

    This used to produce no comment at all, which reads as a pass: the
    value in the table is outside the standard either way, so the reason
    it is not being flagged has to be stated.
    """
    cols = COLS + ['PRD_ci']
    out = run([row(PRD=1.04, PRD_ci=(1.01, 1.06))], cols=cols)
    assert 'not confirmed' in comment(out)
    assert 'outside assessment standards' in comment(out)
    # Unconfirmed, so no non-acceptance marker on the value.
    assert out.loc[0, 'PRD'] == '1.0400'
    # A CI clear of the standard confirms it, and says so.
    out = run([row(PRD=1.09, PRD_ci=(1.06, 1.12))], cols=cols)
    assert 'PRD suggests a review' in comment(out)
    assert 'entirely outside' in comment(out)
    assert out.loc[0, 'PRD'].endswith(NOT_ACCEPTABLE)


def test_ci_based_comments_name_the_level_they_used():
    """`req_significance` and `ci_conf_level` are different numbers.

    A comment decided by an interval used to carry either the significance
    level or no level at all, so a reader could not tell which interval in
    the table it referred to.
    """
    cols = COLS + ['PRD_ci']
    for level, expected in ((0.90, '90% confidence interval'),
                            (0.99, '99% confidence interval')):
        out = run([row(PRD=1.09, PRD_ci=(1.06, 1.12))], cols=cols,
                  ci_conf_level=level)
        assert expected in comment(out)


def test_without_a_ci_level_the_comment_stays_unqualified():
    cols = COLS + ['PRD_ci']
    out = run([row(PRD=1.09, PRD_ci=(1.06, 1.12))], cols=cols)
    text = comment(out)
    assert 'confidence interval' in text
    assert '%' not in text.split('confidence interval')[0][-6:]


def test_a_non_significant_prb_is_reported_and_reformatted():
    """PRB arrives marked; the marker becomes a comment and a symbol."""
    out = run([row(PRB='-0.0812*')])
    assert comment(out) == 'PRB not significant at 95%'
    assert out.loc[0, 'PRB'] == f'-0.0812{NOT_ACCEPTABLE}'


def test_a_significant_prb_is_checked_against_its_standard():
    out = run([row(PRB=-0.0812)])
    assert 'PRB suggests a review regarding assessment standards (-0.05, 0.05)' in comment(out)


def test_a_p_value_above_the_requirement_is_reported():
    cols = COLS + ['Gini_Sale_Prices', 'Gini_Sale_Prices_Pvalue']
    out = run([row(Gini_Sale_Prices=0.4, Gini_Sale_Prices_Pvalue=0.42)], cols=cols)
    assert 'not significant at 95%: P-value 0.42' in comment(out)
    assert out.loc[0, 'Gini_Sale_Prices'].endswith(NOT_ACCEPTABLE)

    out = run([row(Gini_Sale_Prices=0.4, Gini_Sale_Prices_Pvalue=0.01)], cols=cols)
    assert comment(out) == ''


def test_without_a_standard_only_a_zero_spanning_ci_is_reported():
    cols = COLS + ['Gini_Sale_Prices', 'Gini_Sale_Prices_ci']
    out = run([row(Gini_Sale_Prices=0.02, Gini_Sale_Prices_ci=(-0.01, 0.05))], cols=cols)
    assert 'Gini Sale Prices could be zero' in comment(out)
    assert 'includes 0' in comment(out)

    out = run([row(Gini_Sale_Prices=0.3, Gini_Sale_Prices_ci=(0.2, 0.4))], cols=cols)
    assert comment(out) == ''


def test_an_unacceptable_vei_conclusion_marks_the_vei_value():
    out = run([row(VEI=-22.0, VEI_conclusion='Regressivity')])
    assert 'Vertical Equity Indicator does not pass: Regressivity.' in comment(out)
    assert out.loc[0, 'VEI'] == f'-22.00{NOT_ACCEPTABLE}'


@pytest.mark.parametrize("conclusion", ['Regressivity', 'Progressivity'])
def test_both_inequity_verdicts_are_reported(conclusion):
    out = run([row(VEI=-22.0, VEI_conclusion=conclusion)])
    assert f'Vertical Equity Indicator does not pass: {conclusion}.' in comment(out)


def test_an_acceptable_vei_is_not_reported():
    """`vei_analysis` says 'Acceptable', not 'Compliant'.

    The check used to be `value != 'Compliant'` -- a word the VEI never
    produces -- so every compliant VEI carried the self-contradicting
    comment "Vertical Equity Indicator is not acceptable: Acceptable." on
    every row of every study.
    """
    out = run([row(VEI=-2.0, VEI_conclusion='Acceptable')])
    assert 'Vertical Equity' not in comment(out)
    assert NOT_ACCEPTABLE not in str(out.loc[0, 'VEI'])


def test_a_vei_that_bailed_out_on_sample_size_is_not_an_inequity_finding():
    # 'Sample size too small' is the absence of a verdict, not a failed one.
    # Count is above the small-sample threshold here so that the row is
    # checked at all and only the VEI branch can produce the comment.
    out = run([row(VEI=0.0, VEI_conclusion='Sample size too small')])
    assert 'Vertical Equity' not in comment(out)


def test_the_reported_verdicts_are_the_ones_vei_analysis_emits():
    """Pins the two ends of the contract together.

    The original bug was a consumer testing for a string the producer never
    emitted, which no test caught because each side was checked alone.
    """
    from ratio_study.indicators import (VEI_ACCEPTABLE, VEI_CONCLUSIONS,
                                        VEI_INEQUITY_CONCLUSIONS)

    assert set(VEI_INEQUITY_CONCLUSIONS) <= set(VEI_CONCLUSIONS)
    assert VEI_ACCEPTABLE not in VEI_INEQUITY_CONCLUSIONS
    for conclusion in VEI_CONCLUSIONS:
        out = run([row(VEI=-22.0, VEI_conclusion=conclusion)])
        reported = 'Vertical Equity' in comment(out)
        assert reported == (conclusion in VEI_INEQUITY_CONCLUSIONS), conclusion


def test_underscores_in_column_names_read_as_spaces():
    cols = COLS + ['Gini_Sale_Prices', 'Gini_Sale_Prices_ci']
    out = run([row(Gini_Sale_Prices=0.0, Gini_Sale_Prices_ci=(-0.1, 0.1))], cols=cols)
    assert 'Gini Sale Prices' in comment(out)
    assert 'Gini_Sale_Prices' not in comment(out)


def test_several_breaches_are_joined_in_one_comment():
    out = run([row(Median=1.20, PRD=1.09)])
    assert comment(out).count(' | ') == 1
    assert comment(out).startswith('Median suggests')
    assert not comment(out).startswith(' | ')


# --------------------------------------------------------------------------
# Small samples
# --------------------------------------------------------------------------

def test_a_small_sample_replaces_the_checks_with_one_comment():
    """Nothing at this size supports a conclusion, so nothing is claimed."""
    out = run([row(Count=8, Median=1.9, PRD=1.4, PRB=0.3)])
    assert comment(out).startswith('Small sample (8)')
    assert ' | ' not in comment(out)
    assert out.loc[0, 'Count'] == f'8{SMALL_SAMPLE}'
    assert NOT_ACCEPTABLE not in ''.join(out.loc[0].astype(str))


def test_an_unusable_sample_says_so():
    out = run([row(Count=3)])
    assert comment(out).startswith('Sample too small (3)')
    assert out.loc[0, 'Count'] == f'3{SMALL_SAMPLE}'


# --------------------------------------------------------------------------
# Strata
# --------------------------------------------------------------------------

def test_a_stratum_median_is_checked_against_the_overall_median():
    rows = [row(), row(Level='nbhd', Group='A', Median=1.20)]
    out = run(rows)
    assert comment(out, 0) == ''
    assert 'Median suggests a review regarding assessment standards (0.95, 1.05)' in comment(out, 1)

    # Inside +/- 5% of the overall median but outside the fixed band.
    rows = [row(Median=1.15), row(Level='nbhd', Group='A', Median=1.13)]
    out = run(rows)
    assert 'Median suggests' in comment(out, 0)          # 1.15 breaches (0.9, 1.1)
    assert comment(out, 1) == ''                         # 1.13 is within 1.15 +/- 0.05


def test_a_stratum_median_ci_uses_the_iaao_percent_change_test():
    """The worked example from IAAO 11.2.1: 4.08% is acceptable."""
    cols = COLS + ['Median_ci']
    rows = [row(Median=1.01, Median_ci=(0.98, 1.05)),
            row(Level='nbhd', Group='A', Median=0.85, Median_ci=(0.81, 0.94))]
    out = run(rows, cols=cols)
    assert comment(out, 0) == ''
    assert comment(out, 1) == ''


def test_a_stratum_median_ci_confirms_a_breach_past_the_five_percent_limit():
    cols = COLS + ['Median_ci']
    rows = [row(Median=1.01, Median_ci=(0.98, 1.05)),
            row(Level='nbhd', Group='A', Median=0.80, Median_ci=(0.75, 0.90))]
    out = run(rows, cols=cols)
    assert comment(out, 0) == ''
    assert 'Median suggests a review under IAAO 11.2.1' in comment(out, 1)
    assert '8.16%' in comment(out, 1)


def test_a_stratum_median_ci_check_works_on_the_high_side_too():
    cols = COLS + ['Median_ci']
    rows = [row(Median=1.01, Median_ci=(0.98, 1.05)),
            row(Level='nbhd', Group='A', Median=1.20, Median_ci=(1.15, 1.25))]
    out = run(rows, cols=cols)
    assert 'Median suggests a review under IAAO 11.2.1' in comment(out, 1)
    assert '9.52%' in comment(out, 1)


def test_a_stratum_median_ci_falls_back_without_a_usable_interval():
    """Only the stratum has a CI: the fixed +/- 5 point band still applies."""
    cols = COLS + ['Median_ci']
    rows = [row(Median=1.15), row(Level='nbhd', Group='A', Median=1.13,
                                  Median_ci=(1.10, 1.16))]
    out = run(rows, cols=cols)
    assert comment(out, 1) == ''                         # 1.13 is within 1.15 +/- 0.05


def test_a_stratum_cod_gets_the_wider_band():
    rows = [row(), row(Level='nbhd', Group='A', CoD=0.18)]
    out = run(rows)
    assert comment(out, 1) == ''                         # within (0.05, 0.2)
    rows = [row(), row(Level='nbhd', Group='A', CoD=0.22)]
    out = run(rows)
    assert 'CoD suggests a review regarding assessment standards (0.05, 0.2)' in comment(out, 1)


# --------------------------------------------------------------------------
# Strata vs. strata (IAAO 11.2.2 -- uniformity between classes)
# --------------------------------------------------------------------------

def test_two_strata_within_ten_percent_of_each_other_get_no_comment():
    cols = COLS + ['Median_ci']
    rows = [row(Level='nbhd', Group='A', Median=1.05, Median_ci=(1.02, 1.08)),
            row(Level='nbhd', Group='B', Median=1.00, Median_ci=(0.97, 1.03))]
    out = run(rows, cols=cols)
    assert 'IAAO 11.2.2' not in comment(out, 0)
    assert 'IAAO 11.2.2' not in comment(out, 1)


def test_two_strata_past_the_ten_percent_limit_are_flagged_on_both():
    cols = COLS + ['Median_ci']
    rows = [row(Level='nbhd', Group='A', Median=1.30, Median_ci=(1.25, 1.35)),
            row(Level='nbhd', Group='B', Median=1.00, Median_ci=(0.95, 1.05))]
    out = run(rows, cols=cols)
    assert 'IAAO 11.2.2' in comment(out, 0)
    assert "Group 'B'" in comment(out, 0)
    assert 'IAAO 11.2.2' in comment(out, 1)
    assert "Group 'A'" in comment(out, 1)
    assert out.loc[0, 'Median'].endswith(NOT_ACCEPTABLE)
    assert out.loc[1, 'Median'].endswith(NOT_ACCEPTABLE)


def test_the_interclass_check_only_compares_strata_sharing_a_level():
    # Different grouping dimensions ('nbhd' vs 'age_group') are not peers,
    # however far apart their medians sit.
    cols = COLS + ['Median_ci']
    rows = [row(Level='nbhd', Group='A', Median=1.30, Median_ci=(1.25, 1.35)),
            row(Level='age_group', Group='old', Median=1.00, Median_ci=(0.95, 1.05))]
    out = run(rows, cols=cols)
    assert 'IAAO 11.2.2' not in comment(out, 0)
    assert 'IAAO 11.2.2' not in comment(out, 1)


def test_the_interclass_check_is_off_when_the_flag_is_off():
    cols = COLS + ['Median_ci']
    rows = [row(Level='nbhd', Group='A', Median=1.30, Median_ci=(1.25, 1.35)),
            row(Level='nbhd', Group='B', Median=1.00, Median_ci=(0.95, 1.05))]
    out = run(rows, cols=cols, compare_groups_median_to_overall_median=False)
    assert 'IAAO 11.2.2' not in comment(out, 0)
    assert 'IAAO 11.2.2' not in comment(out, 1)


def test_a_single_stratum_has_no_peer_to_compare_against():
    cols = COLS + ['Median_ci']
    rows = [row(), row(Level='nbhd', Group='A', Median=1.30, Median_ci=(1.25, 1.35))]
    out = run(rows, cols=cols)
    assert 'IAAO 11.2.2' not in comment(out, 1)


# --------------------------------------------------------------------------
# Vertical equity bundle
# --------------------------------------------------------------------------

def test_the_vertical_equity_bundle_summarises_direction():
    out = run([row(PRD=1.09, PRB=-0.08, VEI=-22.0, VEI_conclusion='Regressivity')],
              VE_bundle_comments=True)
    bundle = out.loc[0, 'Vertical_Equity_Analysis']
    assert 'VEI: Regressivity' in bundle
    assert 'PRD: regressive.' in bundle
    assert 'PRB: regressive.' in bundle

    out = run([row(PRD=0.90, PRB=0.08)], VE_bundle_comments=True)
    bundle = out.loc[0, 'Vertical_Equity_Analysis']
    assert 'PRD: progressive.' in bundle
    assert 'PRB: progressive.' in bundle


def test_no_bundle_column_unless_asked():
    assert 'Vertical_Equity_Analysis' not in run([row()]).columns


# --------------------------------------------------------------------------
# Acceptance binaries
# --------------------------------------------------------------------------

def test_one_boolean_column_per_standard_that_names_an_indicator():
    out = run([row()])
    assert [col for col in out.columns if col.endswith('_acceptable')] == [
        'Median_acceptable', 'PRB_acceptable', 'PRD_acceptable', 'CoD_acceptable']
    # 'Mkt_desc' has no column and no band, so it gets no flag.
    assert 'Mkt_desc_acceptable' not in out.columns


def test_binaries_follow_the_band_that_applies_to_the_row():
    rows = [row(CoD=0.18), row(Level='nbhd', Group='A', CoD=0.18)]
    out = run(rows)
    assert out.loc[0, 'CoD_acceptable'] == False          # population band (0.05, 0.15)
    assert out.loc[1, 'CoD_acceptable'] == True           # stratum band (0.05, 0.2)


def test_binaries_read_the_point_estimate_only():
    """Unlike the comments, they ignore the confidence interval ..."""
    cols = COLS + ['PRD_ci']
    out = run([row(PRD=1.04, PRD_ci=(1.01, 1.06))], cols=cols)
    # The comment records that the interval leaves the breach unconfirmed...
    assert 'not confirmed' in comment(out)
    # ...but the binary reads the point estimate alone, which is out.
    assert out.loc[0, 'PRD_acceptable'] == False


def test_binaries_are_filled_in_for_small_samples():
    """... and are still stated where no comment is made."""
    out = run([row(Count=4, PRD=1.30)])
    assert 'Sample too small' in comment(out)
    assert out.loc[0, 'PRD_acceptable'] == False


def test_a_missing_value_is_na_not_false():
    out = run([row(PRD=np.nan)])
    assert pd.isna(out.loc[0, 'PRD_acceptable'])
    assert out.loc[0, 'Median_acceptable'] == True


def test_a_marked_value_is_read_through_its_marker():
    out = run([row(PRB='-0.0812*')])
    assert out.loc[0, 'PRB_acceptable'] == False
    out = run([row(PRB='0.0100*')])
    assert out.loc[0, 'PRB_acceptable'] == True


def test_add_binaries_false_adds_nothing():
    out = run([row()], add_binaries=False)
    assert not [col for col in out.columns if col.endswith('_acceptable')]


def test_binaries_can_be_computed_on_their_own():
    df = pd.DataFrame([row(), row(Level='nbhd', Group='A', Median=1.20)])
    flags = _acceptance_binaries(df, STANDARDS, overall_median=1.00)
    assert flags['Median_acceptable'].tolist() == [True, False]
    assert flags['Median_acceptable'].dtype == 'boolean'


# --------------------------------------------------------------------------
# Frame handling
# --------------------------------------------------------------------------

def test_the_input_frame_is_left_alone():
    df = pd.DataFrame([row(PRB='-0.0812*')])
    before = df.copy()
    _add_comments(df, 0.05, STANDARDS, COLS, FORMATS, skip=['Level', 'Group'])
    pd.testing.assert_frame_equal(df, before)


def test_a_non_range_index_still_works():
    df = pd.DataFrame([row(), row(Level='nbhd', Group='A', PRD=1.09)],
                      index=['overall', 'a'])
    out = _add_comments(df, 0.05, STANDARDS, COLS, FORMATS, skip=['Level', 'Group'])
    assert out.loc['overall', COMMENTS] == ''
    assert 'PRD suggests' in out.loc['a', COMMENTS]
    assert out.loc['a', 'PRD'].endswith(NOT_ACCEPTABLE)


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
def test_other_backends_are_accepted(backend):
    frame = pytest.importorskip(backend)
    df = pd.DataFrame([row()])
    data = (frame.from_pandas(df) if backend == "polars"
            else frame.Table.from_pandas(df))
    out = _add_comments(data, 0.05, STANDARDS, COLS, FORMATS,
                        skip=['Level', 'Group'])
    assert COMMENTS in out.columns


# --------------------------------------------------------------------------
# compare_ratios
# --------------------------------------------------------------------------

def two_tables(prb1, prb2):
    keys = dict(Level='Overall', Group='')
    return (pd.DataFrame([dict(keys, PRB=prb1)]),
            pd.DataFrame([dict(keys, PRB=prb2)]))


def test_compare_ratios_keeps_the_sign_of_a_marked_prb():
    """A worsening negative PRB must not read as an improvement."""
    out = compare_ratios(*two_tables('-0.0500*', '-0.1000*'))
    assert out.loc[0, 'PRB_diff'] == '-0.05↓'


def test_compare_ratios_reads_a_marked_prb_at_its_real_value():
    out = compare_ratios(*two_tables('-0.0500*', '0.0500*'))
    assert out.loc[0, 'PRB_diff'] == '0.1↑'


def test_compare_ratios_handles_a_percent_formatted_prb():
    out = compare_ratios(*two_tables('-5.00%', '-10.00%'))
    assert out.loc[0, 'PRB_diff'] == '-0.05↓'


def test_compare_ratios_takes_unmarked_numeric_prb_too():
    out = compare_ratios(*two_tables(-0.05, -0.10))
    assert out.loc[0, 'PRB_diff'] == '-0.05↓'


def test_compare_ratios_leaves_an_unreadable_prb_out_of_the_diff():
    out = compare_ratios(*two_tables('n/a', '-0.1000*'))
    assert out.loc[0, 'PRB_diff'] == '-'
