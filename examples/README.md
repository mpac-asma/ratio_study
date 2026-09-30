# Examples

Five notebooks walk through the package on the anonymised sample dataset that ships inside it (`ratio_study.load_sample_dataset()`), so nothing needs to be downloaded or prepared. Each is self-contained: run it top to bottom.

| Notebook | Covers | Extra needed |
|---|---|---|
| [01_basics](01_basics.ipynb) | What a ratio study is; indicators; confidence intervals; compliance commentary; stratification; the IQR outlier filter; charts; `compare_ratios` | none |
| [02_spatial](02_spatial.ipynb) | `ratio_map`, global and local Moran's I, `vem_map` | `spatial` |
| [03_reporting](03_reporting.ipynb) | `ratios_report`: PDF, XLSX, CSV, ZIP and PNG output; personalising the report | none |
| [04_advanced_diagnostics](04_advanced_diagnostics.ipynb) | ML / SHAP inequity diagnostics; sales-chasing detection (`detect_sales_chasing`, the `sc_*` family) | `ml` |
| [05_vertical_equity](05_vertical_equity.ipynb) | VEI, the Vertical Equity Panel, VEM, the Gini family | `interactive` (one chart) |

```bash
pip install "ratio_study[spatial,ml,interactive]" jupyter
```

The notebooks write their outputs (PDFs, CSVs, PNGs) to the directory they are run from. The sample dataset's latitude/longitude are randomly assigned placeholders (many points fall on the Ottawa River), so the spatial notebook demonstrates the tools, not a real geographic pattern. Interactive maps only render in a live Jupyter session; the saved notebooks show a placeholder.

[docs/demo-coverage.md](../docs/demo-coverage.md) lists which public functions the notebooks call and which they do not.
