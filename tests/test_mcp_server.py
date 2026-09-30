"""Tests for the MCP server tools.

The server is how an AI assistant drives the package, so a broken keyword
argument here is invisible until someone's Copilot session fails. These
exercise each tool end to end against the bundled sample data.
"""
import asyncio
import json

import matplotlib
matplotlib.use('Agg')
import pytest

import ratio_study as rs

mcp_server = pytest.importorskip('ratio_study.mcp_server')
pytestmark = pytest.mark.skipif(mcp_server.FastMCP is None,
                                reason='the `mcp` package is not installed')

MV = 'time_adjusted_sale_price'
AV = 'assessment_value_new'


def call(tool, **arguments):
    """Invoke an MCP tool and return its JSON payload."""
    result = asyncio.run(mcp_server.mcp.call_tool(tool, arguments))
    payload = result[1] if isinstance(result, tuple) else result
    if isinstance(payload, dict) and 'result' in payload:
        payload = payload['result']
    return json.loads(json.dumps(payload, default=str))


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp('data') / 'sales.csv'
    rs.load_sample_dataset().to_csv(path, index=False)
    return str(path)


def test_every_advertised_tool_is_registered():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == {
        'get_module_capabilities', 'diagnose_setup', 'list_datasets',
        'inspect_dataset', 'run_ratio_study', 'generate_ratio_report',
        'generate_vertical_equity_panel',
    }


def test_capabilities_advertise_the_same_tools():
    caps = call('get_module_capabilities')
    registered = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert set(caps['tools']) == registered
    assert caps['version'] == rs.__version__


def test_inspect_dataset_reports_columns(dataset):
    info = call('inspect_dataset', file_path=dataset)
    assert 'error' not in info
    text = json.dumps(info)
    assert MV in text and AV in text


def test_run_ratio_study_returns_the_headline(dataset):
    out = call('run_ratio_study', file_path=dataset,
               market_value_col=MV, assessment_col=AV)
    assert 'error' not in out
    assert out['rows_used'] == 12583
    headline = out['ratios_headline']
    assert isinstance(headline, list) and headline[0]['Median']


def test_generate_ratio_report_writes_a_pdf(dataset, tmp_path):
    out = call('generate_ratio_report', file_path=dataset,
               market_value_col=MV, assessment_col=AV,
               output_directory=str(tmp_path), model_name='TestModel',
               group_by='property_type')
    assert 'error' not in out, out
    assert any(f.endswith('.pdf') for f in out['files_written'])
    assert out['ratios_headline']['Median']       # a single row here, not a list


def test_generate_ratio_report_can_export_the_numbers(dataset, tmp_path):
    out = call('generate_ratio_report', file_path=dataset,
               market_value_col=MV, assessment_col=AV,
               output_directory=str(tmp_path), export_ratios='xlsx')
    assert any(f.endswith('.xlsx') for f in out['files_written'])


def test_generate_ratio_report_survives_indicators_left_unset(dataset, tmp_path):
    # Regression: the report read `indicators` before defaulting it, so an
    # MCP call that omitted the argument died on `x in None`.
    out = call('generate_ratio_report', file_path=dataset,
               market_value_col=MV, assessment_col=AV,
               output_directory=str(tmp_path))
    assert 'error' not in out, out
    assert any(f.endswith('.pdf') for f in out['files_written'])


def test_ratios_report_defaults_indicators_like_ratios():
    # The same bug at the library level, where an AI is not involved.
    sales = rs.load_sample_dataset().head(400)
    frame = rs.ratios_report(sales, assessment_header=AV, market_value_header=MV,
                             indicators=None, pdf_args={'gen_pdf_report': False},
                             print_ci_settings=False, return_dict=False)
    assert 'Median' in frame.columns


def test_generate_vertical_equity_panel_writes_a_png(dataset, tmp_path):
    out = call('generate_vertical_equity_panel', file_path=dataset,
               market_value_col=MV, assessment_col=AV,
               output_directory=str(tmp_path), filename='panel')
    assert 'error' not in out, out
    assert out['file_written'].endswith('.png')
    assert (tmp_path / 'panel.png').exists()          # '.png' appended for us


def test_the_panel_tool_reports_the_gauge_values_as_text(dataset, tmp_path):
    # So the assistant can describe the verdict without opening the image.
    out = call('generate_vertical_equity_panel', file_path=dataset,
               market_value_col=MV, assessment_col=AV,
               output_directory=str(tmp_path))
    for key in ('Median', 'CoD', 'PRD', 'PRB', 'VEI', 'VEI_conclusion'):
        assert key in out['verdicts']


@pytest.mark.parametrize('tool', ['generate_ratio_report',
                                  'generate_vertical_equity_panel'])
def test_a_missing_output_directory_is_an_error_not_a_crash(dataset, tmp_path, tool):
    out = call(tool, file_path=dataset, market_value_col=MV, assessment_col=AV,
               output_directory=str(tmp_path / 'does_not_exist'))
    assert 'error' in out
    assert 'not an existing directory' in out['error']


@pytest.mark.parametrize('tool', ['run_ratio_study', 'generate_ratio_report',
                                  'generate_vertical_equity_panel'])
def test_a_bad_column_comes_back_as_a_readable_error(dataset, tmp_path, tool):
    kwargs = dict(file_path=dataset, market_value_col=MV, assessment_col='nope')
    if tool != 'run_ratio_study':
        kwargs['output_directory'] = str(tmp_path)
    out = call(tool, **kwargs)
    # A traceback is present when an exception was raised; the report tool
    # instead reads the package's own error file, which is more useful.
    assert 'error' in out
    assert 'nope' in out['error'] or 'traceback' in out


def test_no_tool_writes_to_stdout(dataset, tmp_path, capsys):
    # On a stdio transport stdout is the JSON-RPC channel; one stray print
    # from `print_ci_settings` or `iqr_filter_result` corrupts the stream.
    capsys.readouterr()
    call('run_ratio_study', file_path=dataset, market_value_col=MV,
         assessment_col=AV, iqr_trim_multiple=3)
    call('generate_ratio_report', file_path=dataset, market_value_col=MV,
         assessment_col=AV, output_directory=str(tmp_path), iqr_trim_multiple=3)
    call('generate_vertical_equity_panel', file_path=dataset, market_value_col=MV,
         assessment_col=AV, output_directory=str(tmp_path))
    assert capsys.readouterr().out == ''
