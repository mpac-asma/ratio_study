"""Generate docs/reference/*.md from the package's own docstrings.

Re-run after any API change:  python tools/generate_api_docs.py
"""
import importlib, inspect, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'docs' / 'reference'

import ratio_study as rs

# module -> (page filename, page title, one-line purpose)
PAGES = [
    ('ratio_study.ratio_study', 'ratio-study.md', 'Core study',
     'The two entry points most studies need: `ratios` and `ratios_report`.'),
    ('ratio_study.indicators', 'indicators.md', 'Indicators',
     'Individual IAAO indicators, callable on their own.'),
    ('ratio_study.confint', 'confidence-intervals.md', 'Confidence intervals',
     'Bootstrap and Appendix D closed-form interval estimators.'),
    ('ratio_study.graphs', 'charts.md', 'Charts',
     'Every chart the package draws, and the Vertical Equity Panel.'),
    ('ratio_study.spatial', 'spatial.md', 'Spatial',
     'Ratio maps, VEM maps and Moran’s I.'),
    ('ratio_study.gini', 'gini.md', 'Gini family',
     'Quintos’ Gini measures and decompositions (Appendix F).'),
    ('ratio_study.sales_chasing', 'sales-chasing.md', 'Sales chasing',
     'The Appendix B detection techniques.'),
    ('ratio_study.ml_diagnostics', 'ml-diagnostics.md', 'ML diagnostics',
     'ASR predictability and the SHAP explanation family.'),
    ('ratio_study.reporting', 'reporting.md', 'Reporting helpers',
     'Comparison tables and export helpers.'),
    ('ratio_study._data', 'sample-data.md', 'Sample data',
     'The bundled anonymised sales file.'),
]

# Features still under testing: callable, but kept out of the published docs.
UNDER_TEST_PREFIXES = ('bo_',)
PUBLIC_TOP = {n for n in rs.__all__ if not n.startswith(UNDER_TEST_PREFIXES)}


def signature_of(fn):
    try:
        sig = f'{fn.__name__}{inspect.signature(fn)}'
        # A Path.cwd() default is evaluated at import time; do not publish the
        # machine-specific path it resolved to.
        return re.sub(r"PosixPath\('[^']*'\)", 'Path.cwd()', sig)
    except (TypeError, ValueError):
        return fn.__name__ + '(...)'


def split_params(text):
    """
    Split a parameter list on its top-level commas only.

    A plain ``text.split(',')`` also splits inside a default value that is
    itself a tuple or a list -- ``standard_bands=(0.10, 0.25)`` came out as
    two parameters, and could be wrapped across a line in the middle of the
    tuple.
    """
    parts, depth, current = [], 0, ''
    for char in text:
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        if char == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def wrap_signature(text, width=88):
    """Break a long signature after commas so it stays readable in a code block."""
    if len(text) <= width:
        return text
    head, _, rest = text.partition('(')
    # Split the parameter list off at its own matching close paren, so a
    # trailing tuple default is not mistaken for the end of the signature
    # (`rstrip(')')` ate both) and a return annotation -- `) -> dict` -- is
    # carried through rather than having a second `)` appended after it.
    depth, close = 1, len(rest)
    for i, char in enumerate(rest):
        depth += {'(': 1, ')': -1}.get(char, 0)
        if depth == 0:
            close = i
            break
    rest, tail = rest[:close], rest[close + 1:]
    parts, line, out = split_params(rest), '', []
    indent = ' ' * (len(head) + 1)
    for part in parts:
        candidate = (line + ', ' + part) if line else part
        if len(candidate) + len(indent) > width and line:
            out.append(line + ',')
            line = part
        else:
            line = candidate
    out.append(line)
    return head + '(' + ('\n' + indent).join(out) + ')' + tail


def anchor(name):
    return name.lower().replace('_', '-')


def render_module(module_name, title, purpose):
    module = importlib.import_module(module_name)
    names = getattr(module, '__all__', None)
    if names is None:
        names = [n for n in dir(module)
                 if not n.startswith('_') and n in PUBLIC_TOP]
    functions = []
    for name in sorted(names):
        obj = getattr(module, name, None)
        if inspect.isfunction(obj) and obj.__module__ == module_name:
            functions.append((name, obj))
    if not functions:
        return None

    lines = [f'# {title}', '', purpose, '',
             f'*Module:* `{module_name}`', '',
             '## Contents', '']
    for name, fn in functions:
        summary = (fn.__doc__ or '').strip().split('\n')[0].strip() or '—'
        lines.append(f'- [`{name}`](#{anchor(name)}) — {summary}')
    lines.append('')

    for name, fn in functions:
        lines += ['---', '', f'## `{name}`', '', '```python',
                  wrap_signature(signature_of(fn)), '```', '']
        doc = inspect.cleandoc(fn.__doc__ or '_No docstring._')
        lines += [doc, '']
    return '\n'.join(lines).rstrip() + '\n'


def demo_coverage():
    """Which public callables the demo notebook exercises, and which it does not."""
    import json
    from ratio_study import spatial, ml_diagnostics as ml

    text = ''
    for notebook in sorted((ROOT / 'examples').glob('[0-9][0-9]_*.ipynb')):
        nb = json.loads(notebook.read_text())
        # Code cells only. A function named in prose is documented, not
        # exercised, and this page is about what the notebooks actually run.
        text += '\n'.join(''.join(c['source']) for c in nb['cells']
                          if c.get('cell_type') == 'code') + '\n'

    api = {}
    for name in rs.__all__:
        if name.startswith(UNDER_TEST_PREFIXES):
            continue
        obj = getattr(rs, name)
        if inspect.isfunction(obj):
            api[name] = ('ratio_study', obj)
    for module, label in ((spatial, 'spatial'), (ml, 'ml_diagnostics')):
        names = getattr(module, '__all__', [n for n in dir(module) if not n.startswith('_')])
        for name in names:
            obj = getattr(module, name, None)
            if (inspect.isfunction(obj) and obj.__module__.startswith('ratio_study')
                    and name not in api):
                api[name] = (label, obj)

    covered, missing = [], []
    for name in sorted(api):
        label, obj = api[name]
        used = re.search(rf'\b{re.escape(name)}\s*\(', text) is not None
        (covered if used else missing).append((name, label, obj))
    return covered, missing


def write_coverage(covered, missing):
    total = len(covered) + len(missing)
    lines = [
        '# Demo coverage',
        '',
        'Which public callables the five [`examples/`](../examples/README.md) notebooks '
        'exercise, and which they do not.',
        '',
        'Generated by `tools/generate_api_docs.py` — re-run it after changing the API '
        'or the notebook rather than editing this file.',
        '',
        f'**{len(covered)} of {total} public callables appear in the example notebooks; '
        f'{len(missing)} do not.**',
        '',
        'A function counts as covered only when the notebook calls it *by name*. '
        'Several of the uncovered ones do run in the notebooks, just indirectly — '
        '`gini_decomp` and `gini_factors` through `ratios(gini_args=...)`, '
        '`sc_concentration` and `sc_cdf_gap` through `detect_sales_chasing`, and every '
        'indicator through `ratios(indicators=[...])`. Those are marked below.',
        '',
        '## Not called directly in the example notebooks',
        '',
        '| Function | Module | Reached indirectly via | What it does |',
        '|---|---|---|---|',
    ]

    indirect = {
        'gini_decomp': '`ratios(gini_args={"gini_decomposition": True})`',
        'gini_factors': '`ratios(gini_args={"gini_factors_variables": [...]})`',
        'gini_lorenz_plot': '`lorenz_plot`',
        'sc_concentration': '`detect_sales_chasing`',
        'sc_cdf_gap': '`detect_sales_chasing`',
        'sc_mass_appraisal': '`detect_sales_chasing` (needs an independent value column)',
        'sc_independent_values': '`detect_sales_chasing` (needs an independent value column)',
        'rts_calculator': '`ratios`',
        'normalize_ci_args': '`ratios` (CI argument resolution)',
        'compute_median_ci': '`vem`, `vei_analysis`',
        'cmbi': '`ml_diagnostics.asr_predictability`',
        'cdr': '`ml_diagnostics.asr_predictability`',
        'dcorr': '`ml_diagnostics.asr_predictability`',
        'asr_target': '`ml_diagnostics.asr_predictability`',
        'log_residual': '`asr_predictability(target="log_residual")`',
        'r2_correlation': '`ml_diagnostics.asr_predictability`',
    }
    for name in ('cod', 'prd', 'prb', 'cod_residual', 'cov_adj', 'coc', 'fsd',
                 'rmse', 'mae', 'mape', 'mdmape', 'asr_median', 'asr_mean'):
        indirect[name] = '`ratios(indicators=[...])`'
    for name in ('mean_ci', 'median_ci_rank', 'median_ci_clapp',
                 'median_ci_distribution', 'weighted_mean_ci', 'cod_ci'):
        indirect[name] = '`ratios(ci_mode="distribution")`'

    for name, label, obj in missing:
        summary = (obj.__doc__ or '').strip().split('\n')[0].strip().rstrip('.') or '—'
        lines.append(f'| `{name}` | {label} | {indirect.get(name, "— (not exercised)")} '
                     f'| {summary} |')

    never = [n for n, _, _ in missing if n not in indirect]
    lines += [
        '',
        '## Genuinely unexercised',
        '',
        'Nothing in the notebooks reaches these, directly or indirectly:',
        '',
    ]
    lines += [f'- `{n}`' for n in never] or ['- (none)']
    lines += [
        '',
        '## Called directly in the example notebooks',
        '',
        ', '.join(f'`{n}`' for n, _, _ in covered),
        '',
    ]
    (ROOT / 'docs' / 'demo-coverage.md').write_text('\n'.join(lines))
    print(f'  demo-coverage.md           {len(missing)} uncovered of {total}')
    return never


def write_reference_index(written):
    lines = ['# API reference', '',
             'Generated from the package docstrings by '
             '`tools/generate_api_docs.py`. Do not edit by hand.', '',
             '| Page | Functions | Covers |', '|---|---|---|']
    for filename, title, count, purpose in written:
        lines.append(f'| [{title}]({filename}) | {count} | {purpose} |')
    total = sum(c for _, _, c, _ in written)
    lines += ['', f'**{total} documented functions.**', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(f'  reference/README.md        {total} functions indexed')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for module_name, filename, title, purpose in PAGES:
        page = render_module(module_name, title, purpose)
        if page is None:
            print(f'  skipped (no public functions): {module_name}')
            continue
        (OUT / filename).write_text(page)
        count = page.count('\n## `')
        written.append((filename, title, count, purpose))
        print(f'  {filename:26s} {count:3d} functions')
    write_reference_index(written)
    covered, missing = demo_coverage()
    write_coverage(covered, missing)
    return written


if __name__ == '__main__':
    main()
