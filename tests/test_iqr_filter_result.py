"""Tests for `iqr_filter_result`: reporting and exporting the sales an IQR trim removed.

Appendix C.6 of the Standard on Ratio Studies requires a report to state the
basis for excluding outliers and to identify what was removed. Trimming
silently is the failure mode these cover.
"""
import gzip

import pandas as pd
import pytest

import ratio_study as rs
from ratio_study.helpers import _abbreviate_value, report_iqr_trim

MV = 'time_adjusted_sale_price'
AV = 'assessment_value_new'


@pytest.fixture(scope='module')
def sales():
    return rs.load_sample_dataset()


def _lines(capsys):
    return capsys.readouterr().out


def _trim(sales, **kwargs):
    kwargs.setdefault('iqr_trim_multiple', 3)
    kwargs.setdefault('iqr_group_by', 'neighbourhood')
    return rs.ratios(sales, assessment_header=AV, market_value_header=MV,
                     return_dict=True, **kwargs)


# --- printing --------------------------------------------------------------

def test_printing_is_the_default_and_reports_the_three_totals(sales, capsys):
    _trim(sales)
    out = _lines(capsys)
    assert 'Sales trimmed' in out
    assert 'Total trimmed assessment value' in out
    assert 'Total trimmed market value' in out


def test_the_summary_reports_the_share_of_the_sample(sales, capsys):
    # Appendix C.5 caps trimming at 5-10%, so the share is the number that
    # tells a reader whether the trim was within the Standard.
    _trim(sales)
    out = _lines(capsys)
    line = next(l for l in out.splitlines() if l.strip().startswith('Sales trimmed'))
    assert '%)' in line


def test_nothing_is_printed_when_the_filter_is_off(sales, capsys):
    rs.ratios(sales, assessment_header=AV, market_value_header=MV)
    assert 'IQR filter' not in _lines(capsys)


def test_none_silences_the_report(sales, capsys):
    _trim(sales, iqr_filter_result=None)
    assert 'IQR filter' not in _lines(capsys)


def test_a_trim_that_removes_nothing_still_says_so(sales, capsys):
    # A zero can mean "nothing extreme" or "min_cod blocked it"; silence
    # would leave the reader unable to tell the filter had run at all.
    _trim(sales, iqr_group_by=None, iqr_min_cod=0.99)
    out = _lines(capsys)
    assert 'Sales trimmed          : 0' in out
    assert 'min_cod' in out


# --- saving ----------------------------------------------------------------

def test_save_writes_a_gzipped_csv_and_reports_the_path(sales, tmp_path, capsys):
    result = _trim(sales, iqr_filter_result='save', export_directory=tmp_path,
                   model_name='My Model')
    path = result['IQR_trimmed_sales_export']
    assert path.exists() and path.name.endswith('.csv.gz')
    assert path.parent == tmp_path
    assert 'MyModel' in path.name          # punctuation stripped, as elsewhere

    out = _lines(capsys)
    assert str(path) in out
    assert 'Total trimmed market value' in out     # the summary still follows the path


def test_a_directory_is_used_as_a_directory(sales, tmp_path, capsys):
    target = tmp_path / 'audit'            # does not exist yet
    result = _trim(sales, iqr_filter_result=str(target))
    path = result['IQR_trimmed_sales_export']
    assert path.parent == target and path.name.startswith('iqr_trimmed_sales_')


def test_a_file_path_is_used_verbatim_but_forced_to_gzip(sales, tmp_path):
    result = _trim(sales, iqr_filter_result=str(tmp_path / 'dropped.csv'))
    assert result['IQR_trimmed_sales_export'] == tmp_path / 'dropped.csv.gz'


def test_an_existing_csv_gz_name_is_not_doubled(sales, tmp_path):
    result = _trim(sales, iqr_filter_result=str(tmp_path / 'dropped.csv.gz'))
    assert result['IQR_trimmed_sales_export'] == tmp_path / 'dropped.csv.gz'


def test_printing_writes_no_file(sales, tmp_path):
    result = _trim(sales, iqr_filter_result='print', export_directory=tmp_path)
    assert result['IQR_trimmed_sales_export'] is None
    assert not list(tmp_path.glob('*.gz'))


# --- what lands in the file ------------------------------------------------

def test_the_export_carries_the_whole_record_not_just_the_study_columns(sales, tmp_path):
    # Appendix C.2 asks for outliers to be scrutinised. Two columns is not
    # enough to scrutinise anything.
    result = _trim(sales, iqr_filter_result='save', export_directory=tmp_path)
    written = pd.read_csv(result['IQR_trimmed_sales_export'], compression='gzip')
    for column in sales.columns:
        assert column in written.columns
    assert 'ASR' in written.columns


def test_the_export_matches_the_trimmed_frame_and_the_reported_count(sales, tmp_path, capsys):
    result = _trim(sales, iqr_filter_result='save', export_directory=tmp_path)
    trimmed = result['IQR_trimmed_sales']
    written = pd.read_csv(result['IQR_trimmed_sales_export'], compression='gzip')
    assert len(written) == len(trimmed) > 0
    assert f'{len(trimmed):,}' in _lines(capsys)


def test_the_file_really_is_gzipped(sales, tmp_path):
    result = _trim(sales, iqr_filter_result='save', export_directory=tmp_path)
    with gzip.open(result['IQR_trimmed_sales_export'], 'rt') as handle:
        assert handle.readline().startswith('sale_month')


def test_the_reported_totals_match_the_trimmed_records(sales, tmp_path):
    captured = []
    result = _trim(sales, iqr_filter_result=None)
    trimmed = result['IQR_trimmed_sales']
    report_iqr_trim(trimmed, AV, MV, sales[AV].sum(), sales[MV].sum(),
                    result='print', printer=captured.append)
    text = '\n'.join(captured)
    # Reported to a magnitude suffix: ten-digit totals side by side are
    # what the reader has to compare, and they do not compare by eye.
    assert _abbreviate_value(trimmed[AV].sum()) in text
    assert _abbreviate_value(trimmed[MV].sum()) in text
    assert _abbreviate_value(sales[AV].sum()) in text
    assert _abbreviate_value(sales[MV].sum()) in text


@pytest.mark.parametrize('value, expected', [
    (0, '0'),
    (742, '742'),
    (34_000, '34K'),
    (20_670_693, '20.7M'),
    (6_095_897_974, '6.1B'),
    (-1_500_000, '-1.5M'),
])
def test_totals_are_abbreviated_by_magnitude(value, expected):
    assert _abbreviate_value(value) == expected


# --- propagation -----------------------------------------------------------

def test_ratios_report_propagates_the_argument(sales, tmp_path, capsys):
    rs.ratios_report(sales, assessment_header=AV, market_value_header=MV,
                     iqr_trim_multiple=3, iqr_filter_result='save',
                     export_directory=tmp_path, model_name='Rep',
                     export_in_compacted_file='', return_dict=False)
    assert any(p.name.startswith('iqr_trimmed_sales_') for p in tmp_path.glob('*.gz'))
    assert 'Total trimmed market value' in _lines(capsys)


def test_ratios_report_can_be_silenced_too(sales, tmp_path, capsys):
    rs.ratios_report(sales, assessment_header=AV, market_value_header=MV,
                     iqr_trim_multiple=3, iqr_filter_result=None,
                     export_directory=tmp_path, model_name='Rep',
                     export_in_compacted_file='', return_dict=False)
    assert 'IQR filter' not in _lines(capsys)
