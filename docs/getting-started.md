# Getting started

## Install

```bash
pip install ratio_study
```

The core install is deliberately light. Optional extras switch on the heavier
features:

| Extra | Enables |
|---|---|
| `ratio_study[spatial]` | Ratio maps, VEM maps, Moran's I |
| `ratio_study[ml]` | ASR predictability and the SHAP diagnostics |
| `ratio_study[interactive]` | Plotly vertical-equity charts |

```bash
pip install "ratio_study[spatial,ml,interactive]"
```

## A first study

The package ships an anonymised residential sales file, so nothing below needs
data of your own.

```python
import ratio_study as rs

sales = rs.load_sample_dataset()

rs.ratios(sales,
          assessment_header='assessment_value_new',
          market_value_header='time_adjusted_sale_price')
```

> **Name the two columns explicitly.** `ratios()` takes `assessment_header`
> *before* `market_value_header`. Passing them positionally in the other order
> silently inverts every ratio in the study.

That single call returns the level, uniformity and vertical-equity indicators
with bootstrap confidence intervals attached, and prints the interval settings
it used so nothing rests on a default you never saw.

## Add compliance commentary

`descriptive=True` grades the study against the IAAO standards and writes a
plain-language comment — only where an indicator falls short. A clean study
produces no commentary.

```python
rs.ratios(sales,
          assessment_header='assessment_value_new',
          market_value_header='time_adjusted_sale_price',
          descriptive=True,
          assessment_standards={'Median': (0.90, 1.10),   # s.11.1
                                'CoD':    (0.05, 0.15),   # Table 11.1
                                'VEI':    (-10.0, 10.0),  # s.11.2.6
                                'PRB':    (-0.05, 0.05)})
```

The comments are gated on the confidence intervals, not just the point
estimate — a breach the interval cannot confirm is not flagged, which is the
Standard's own decision rule.

## Stratify

```python
rs.ratios(sales,
          assessment_header='assessment_value_new',
          market_value_header='time_adjusted_sale_price',
          group_by=['neighbourhood',
                    'property_type',
                    ['neighbourhood', 'property_type']],   # nested
          descriptive=True)
```

A list gives several independent groupings; a nested list gives the
cross-tabulation.

## Produce a report

```python
rs.ratios_report(sales,
                 assessment_header='assessment_value_new',
                 market_value_header='time_adjusted_sale_price',
                 group_by='property_type',
                 descriptive=True,
                 model_name='MyModel',
                 base_date='20260101',
                 export_ratios='both',        # xlsx + csv beside the PDF
                 export_directory='.')
```

## Look at it

Every chart takes the same first three arguments and the same three-way output
convention — return a figure, save a PNG, or return a report flowable.

```python
rs.vertical_equity_panel(sales,
                         assessment_header='assessment_value_new',
                         market_value_header='time_adjusted_sale_price')

rs.asr_scatter(sales, 'assessment_value_new', 'time_adjusted_sale_price',
               x='time_adjusted_sale_price', log_x=True)

rs.asr_boxplot(sales, 'assessment_value_new', 'time_adjusted_sale_price',
               by='property_type', show_values=True)
```

| Call | Returns |
|---|---|
| `f(df, ...)` | a matplotlib `Figure`, displayed inline in a notebook |
| `f(df, ..., graph_filepath='chart.png')` | the saved path |
| `f(df, ..., return_object=True)` | a report-ready flowable for the PDF |

## Any dataframe backend

pandas, polars or PyArrow in — the same backend out, via
[narwhals](https://github.com/narwhals-dev/narwhals).

```python
sales = rs.load_sample_dataset('polars')
rs.ratios(sales, assessment_header='assessment_value_new',
          market_value_header='time_adjusted_sale_price')     # polars out
```

## Where to go next

- **[The example notebooks](../examples/README.md)** — the guided tour in five
  parts: [basics](../examples/01_basics.ipynb), [spatial](../examples/02_spatial.ipynb),
  [reporting](../examples/03_reporting.ipynb),
  [advanced diagnostics](../examples/04_advanced_diagnostics.ipynb) (ML and sales
  chasing) and [vertical equity](../examples/05_vertical_equity.ipynb).
- **[API reference](reference/README.md)** — every public function.
- **[MCP server](mcp-server.md)** — drive the package from an AI assistant.
- **[Concepts](concepts.md)** — what the indicators mean and which one to trust.
