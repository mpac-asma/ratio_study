# `ratio_study` documentation

Ratio studies for Computer Assisted Mass Appraisal (CAMA), in Python.

Aligned to the IAAO **Standard on Ratio Studies, September 2026**.

Developed and maintained by **ASMA-DS**, the Data Science team within the
Assessment Standards and Mass Appraisal department at the Municipal Property
Assessment Corporation (MPAC), Ontario, Canada. GPL-3.0.

```bash
pip install "ratio_study[spatial,ml,interactive]"
```

```python
import ratio_study as rs

sales = rs.load_sample_dataset()
rs.ratios(sales,
          assessment_header='assessment_value_new',
          market_value_header='time_adjusted_sale_price',
          descriptive=True)
```

---

## Start here

| | |
|---|---|
| **[Getting started](getting-started.md)** | Install, first study, first report, first chart |
| **[The example notebooks](../examples/README.md)** | The guided tour in five parts: [basics](../examples/01_basics.ipynb), [spatial](../examples/02_spatial.ipynb), [reporting](../examples/03_reporting.ipynb), [advanced diagnostics](../examples/04_advanced_diagnostics.ipynb) (ML and sales chasing), [vertical equity](../examples/05_vertical_equity.ipynb) |
| **[Concepts](concepts.md)** | What the indicators mean, which the Standard prefers, and where this package goes beyond it |

## Reference

| | |
|---|---|
| **[API reference](reference/README.md)** | Every public function, generated from the docstrings |
| **[MCP server](mcp-server.md)** | Drive the package from an AI assistant |
| **[Copilot prompt template](copilot-prompt-template.md)** | Run a study without writing Python |
| **[Copilot 365 setup guide](MCP_COPILOT_365_GUIDE.md)** | The venv-and-coaching route, for locked-down environments |
| **[Demo coverage](demo-coverage.md)** | Which public functions the example notebooks exercise, and which they do not |

---

## What it does

**Standards-complete.** The September 2026 indicators, their compliance ranges,
the Appendix D confidence intervals, the Appendix C outlier rules, the Appendix
E VEI and the Appendix B sales-chasing techniques — implemented once and tested.

**Beyond the Standard.** Appendix F's supporting tests (Quintos' Gini family,
Kruskal-Wallis, Spearman), a moving-median vertical equity measure, spatial
autocorrelation, and ML/SHAP inequity diagnostics — in the same API.

**Reproducible.** One call, seeded, versioned, scriptable in CI. The confidence
interval settings print themselves, so a published number never rests on a
default nobody reviewed.

**Report-ready.** PDF, Excel, CSV and PNG, with compliance commentary written
for a non-technical reader.

**Backend-agnostic.** pandas, polars or PyArrow in, same backend out, via
[narwhals](https://github.com/narwhals-dev/narwhals).

---

## At a glance

| Task | Call |
|---|---|
| Run a study | [`ratios`](reference/ratio-study.md#ratios) |
| Publish one | [`ratios_report`](reference/ratio-study.md#ratios-report) |
| Compare two models | [`compare_ratios`](reference/reporting.md#compare-ratios) |
| Vertical equity, one page | [`vertical_equity_panel`](reference/charts.md#vertical-equity-panel) |
| The preferred VE measure | [`vei_analysis`](reference/indicators.md#vei-analysis) |
| VE across the whole range | [`vem`](reference/indicators.md#vem) |
| Gini / Kakwani (Appendix F) | [`gini_stats`](reference/gini.md#gini-stats) |
| Screen for sales chasing | [`detect_sales_chasing`](reference/sales-chasing.md#detect-sales-chasing) |
| Map the ratios | [`spatial.ratio_map`](reference/spatial.md#ratio-map) |
| Spatial autocorrelation | [`moran_i`](reference/indicators.md#moran-i) |
| Explain the inequity | [`ml_diagnostics.asr_predictability`](reference/ml-diagnostics.md#asr-predictability) |
| Charts | [`asr_scatter`](reference/charts.md#asr-scatter), [`asr_histogram`](reference/charts.md#asr-histogram), [`asr_boxplot`](reference/charts.md#asr-boxplot), [`asr_facet`](reference/charts.md#asr-facet) |

---

## Regenerating these docs

`docs/reference/` and `docs/demo-coverage.md` are generated from the package's
own docstrings and the example notebooks. After changing the API, run:

```bash
python tools/generate_api_docs.py
```

Everything else in `docs/` is written by hand.

---

## References

- IAAO. 2026. *Standard on Ratio Studies.* Kansas City, MO: IAAO
- Gloudemans, R. & Almy, R. 2011. *Fundamentals of Mass Appraisal.* IAAO
- Quintos, C. 2020, 2021. *Journal of Property Tax Assessment & Administration* 17(2), 18(2)
- Bonett, D.G. & Seier, E. 2006. *Biometrical Journal* 48(1)
- Efron, B. & Tibshirani, R.J. 1993. *An Introduction to the Bootstrap*
- Bidanset, P. & Wehrli, J. 2025, GIS/Valuation Technologies Conference; Wehrli, J. 2026, IAAO/IPTI MAVS
