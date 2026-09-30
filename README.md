# Ratio Study for Computer Assisted Mass Appraisal (CAMA) models

Ratio-Study is software package for Python developed by the Municipal Property Assessment Corporation (MPAC) of Ontario, Canada.
It has been developed and is maintained by ASMA-DS: the Data Science team within the Assessment Standards and Mass Appraisal Department at MPAC.

This package provides a comprehensive suite of functions for evaluating Computer Assisted Mass Appraisal (CAMA) real estate valuation models.
These functions comprise but are not limited to the International Association of Assessing Officers (IAAO)' Standard on Ratio Studies, as well as recent advances in this area.

## Backend support (pandas / polars / pyarrow)

The package is **backend-agnostic** thanks to [narwhals](https://github.com/narwhals-dev/narwhals).
Public functions (`ratios`, `rts_calculator`, `asr_median`, `cod`, `prd`, `prb`,
`rmse`, `mae`, `CoC`, `CoV_adj`, `vei_analysis`, `vem`,
`gini_stats`, `gini_decomp`, `gini_factors`,
`compare_ratios`, etc.) accept any of the following:

- `pandas.DataFrame` (default)
- `polars.DataFrame`
- `pyarrow.Table`
- A narwhals-wrapped frame (`narwhals.DataFrame` / `narwhals.LazyFrame`)
- 1D values can be any of: `numpy.ndarray`, `pandas.Series`, `polars.Series`,
  `narwhals.Series`, or a plain list.

Where the input is a polars / pyarrow / narwhals frame, the **result is
returned in the same backend** so you can keep your pipeline native.

A few third-party libraries remain pandas-only, so the package converts
at the function boundary where needed:

- `seaborn` (used in `graphs.py`)
- `xlsxwriter` via `pd.ExcelWriter` (used in `reporting.py`)
- `statsmodels.formula.api.ols` (used in `gini_factors`)

```python
import polars as pl
from ratio_study import ratios, load_sample_dataset

# Load a polars DataFrame from the bundled parquet sample
df = load_sample_dataset("polars")

# Compute IAAO ratio indicators — result is a polars.DataFrame
result = ratios(
    df,
    market_value_header="time_adjusted_sale_price",
    assessment_header="assessment_value_new",
    list_of_indicators=["Median", "Mean", "CoD", "PRD", "PRB"],
    group_by="property_type",
)
print(result.to_dicts())
```

## Installation

```bash
pip install ratio_study                          # core
pip install "ratio_study[spatial,ml,interactive]"  # maps, ML/SHAP diagnostics, Plotly charts
```

## Examples

Five notebooks on the bundled anonymised sample dataset. No data preparation is needed.

| Notebook | Covers |
|---|---|
| [01 Basics](examples/01_basics.ipynb) | Indicators, confidence intervals, compliance checks, stratification, IQR outlier filter, charts, model comparison |
| [02 Spatial](examples/02_spatial.ipynb) | Ratio maps, Moran's I, hot/cold spots, VEM map (`spatial` extra) |
| [03 Reporting](examples/03_reporting.ipynb) | Automated PDF / XLSX / CSV / PNG reports |
| [04 Advanced diagnostics](examples/04_advanced_diagnostics.ipynb) | ML / SHAP inequity diagnostics (`ml` extra) and sales-chasing detection |
| [05 Vertical equity](examples/05_vertical_equity.ipynb) | VEI, Vertical Equity Panel, VEM, Gini family |

## Documentation

See the [documentation index](docs/README.md): [getting started](docs/getting-started.md), [concepts](docs/concepts.md), the [API reference](docs/reference/README.md), and the [MCP server](docs/mcp-server.md) for driving the package from an AI assistant.

## References
IAAO's Standard on Ratio Studies available at https://www.iaao.org/media/standards/Standard_on_Ratio_Studies.pdf

Thomas J. DiCiccio. Bradley Efron. "Bootstrap confidence intervals." Statist. Sci. 11 (3) 189 - 228, August 1996.
https://doi.org/10.1214/ss/1032280214

Quintos, C. (2020). A Gini measure for vertical equity in property assessments. Journal of Property Tax
Assessment & Administration, 17(2). Retrieved from https://researchexchange.iaao.org/jptaa/vol17/iss2/2

Quintos, C. (2021). A Gini decomposition of the sources of inequality in property assessments. Journal of
Property Tax Assessment & Administration, 18(2). Retrieved from https://researchexchange.iaao.org/jptaa/vol18/iss2/6

Formulas on article by David Zaslavsky, retrieved from https://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
