""" Ratio Studies Package
This package makes easier running a Ratio Study, an instrument on the analysis of real property valuation models

Based on IAAO's Standard on Ratio Studies available at https://www.iaao.org/media/standards/Standard_on_Ratio_Studies.pdf
and on the following work:
Quintos, C. (2021). A Gini decomposition of the sources of inequality in property assessments. Journal of
Property Tax Assessment & Administration, 18(2). Retrieved from https://researchexchange.iaao.org/ jptaa/vol18/iss2/6
Quintos, C. (2020). A Gini measure for vertical equity in property assessments. Journal of Property Tax
Assessment & Administration, 17(2). Retrieved from https://researchexchange.iaao.org/jptaa/vol17/iss2/ 2
Thomas J. DiCiccio. Bradley Efron. "Bootstrap confidence intervals." Statist. Sci. 11 (3) 189 - 228, August 1996.
https://doi.org/10.1214/ss/1032280214
Formulas on article by David Zaslavsky: https://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
"""
__author__ = 'Eric Kutchukian @ MPAC'
__copyright__ = 'Copyright 2026, Municipal Property Assessment Corporation'
__credits__ = ['Eric Kutchukian, James Ellens, Micheal Boateng, James Bryan, James Gilbert, Mohammad Jaber']
__license__ = 'GPL v3'
__maintainer__ = '{Eric Kutchukian}'
__email__ = '{eric.kutchukian@mpac.ca}'
__status__ = '{Dev}'


"""ratio_study — ratio-study analysis for real-property assessment.

Quick start
-----------
>>> from ratio_study import ratios, IAAO_RESIDENTIAL
>>> results = ratios(df, "assessed_value", "sale_price", config=IAAO_RESIDENTIAL)

Backend support
---------------
The package is now backend-agnostic across pandas, polars, and
PyArrow thanks to :mod:`narwhals`.  Public functions accept any of
the supported frame types (or a ``narwhals.DataFrame`` / ``Series``)
and return a frame in the same backend as the input where feasible.

Frames that are converted internally:

* ``seaborn`` and ``xlsxwriter`` still require real pandas
  DataFrames.  These conversions happen at the function boundary,
  so callers can pass any supported frame.

* ``statsmodels.formula.api.ols`` is pandas-only and is invoked
  in :func:`ratio_study.gini.gini_factors`.  The frame is converted
  to pandas just for the regression call.

* Output ``DataFrame`` objects returned from
  :func:`ratio_study.ratios`, 
  and :func:`ratio_study.indicators.vem` are re-converted to the
  original backend on the way out.
"""

from ._version import __version__
from ratio_study._data import load_sample_dataset, sample_dataset_path
from ratio_study import nw_utils

# ``__all__`` advertises ``rts_calculator``; import it under that name too so
# ``from ratio_study import *`` does not raise. ``ratios_calculator`` is a thin
# alias kept for backwards compatibility.
from .ratio_study import rts_calculator, ratios_calculator, ratios, ratios_report
from .gini import gini_stats, gini_decomp, gini_factors, gini_lorenz_plot
from .reporting import compare_ratios
from .graphs import (
    vei_boxplot, vertical_equity_panel,
    asr_scatter, asr_histogram, asr_boxplot, asr_facet, lorenz_plot, vem_plot,
)
from .indicators import (
    vei_analysis, vem, asr_median, asr_mean, cod, cod_residual,
    prd, prb, rmse, mae, mape, mdmape, fsd, cov_adj, coc, moran_i,
    market_value_proxy, DEFAULT_MVP_SALE_PRICE_SHARE,
)
from .confint import (
    CI_DEFAULT_INDICATORS, CI_SUPPORTED_INDICATORS, compute_median_ci,
    normalize_ci_args,
    DISTRIBUTION_CI_DEFAULT_INDICATORS, DISTRIBUTION_CI_SUPPORTED_INDICATORS,
    MEDIAN_CI_SMALL_SAMPLE_MAX, cod_ci, distribution_analysis, mean_ci,
    median_ci_clapp, median_ci_distribution, median_ci_rank, prd_ci,
    weighted_mean_ci,
)
from .blinder_oaxaca import (
    bo_decompose, bo_multi_decomposition, bo_format_table,
    bo_coefficient_plot, bo_decomposition_plot, bo_detail_plot
)
from .sales_chasing import (
    SC_MIN_N, critical_concentration, detect_sales_chasing,
    sc_cdf_gap, sc_concentration, sc_independent_values, sc_mass_appraisal,
    sc_split_sample, sc_unit_values, sc_value_change,
)

__all__ = [
    "load_sample_dataset", 
    "sample_dataset_path",
    'rts_calculator', 
    'ratios',
    'ratios_report',
    'gini_stats',
    'gini_decomp',
    'gini_factors',
    'gini_lorenz_plot',
    'compare_ratios',
    'vei_boxplot',
    'vertical_equity_panel',
    'asr_scatter',
    'asr_histogram',
    'asr_boxplot',
    'asr_facet',
    'lorenz_plot',
    'vem_plot',
    'vei_analysis',
    'vem',
    'asr_median',
    'asr_mean',
    'cod',
    'cod_residual',
    'prd',
    'prb',
    'market_value_proxy',
    'DEFAULT_MVP_SALE_PRICE_SHARE',
    'rmse',
    'mae',
    'mape',
    'mdmape',
    'fsd',
    'cov_adj',
    'coc',
    'moran_i',
    'compute_median_ci',
    'normalize_ci_args',
    'CI_DEFAULT_INDICATORS',
    'CI_SUPPORTED_INDICATORS',
    'DISTRIBUTION_CI_DEFAULT_INDICATORS',
    'DISTRIBUTION_CI_SUPPORTED_INDICATORS',
    'MEDIAN_CI_SMALL_SAMPLE_MAX',
    'mean_ci',
    'median_ci_rank',
    'median_ci_clapp',
    'median_ci_distribution',
    'weighted_mean_ci',
    'prd_ci',
    'cod_ci',
    'distribution_analysis',
    'bo_decompose',
    'bo_multi_decomposition',
    'bo_format_table',
    'bo_coefficient_plot',
    'bo_decomposition_plot',
    'bo_detail_plot',
    'SC_MIN_N',
    'critical_concentration',
    'detect_sales_chasing',
    'sc_cdf_gap',
    'sc_concentration',
    'sc_independent_values',
    'sc_mass_appraisal',
    'sc_split_sample',
    'sc_unit_values',
    'sc_value_change',
    'nw_utils',
]


# from __future__ import annotations

# from ratio_study._version import __version__
# from ratio_study.presets import (
#     IAAO_INCOME_PRODUCING,
#     IAAO_RESIDENTIAL,
#     IAAO_VACANT_LAND,
#     KNOWN_INDICATORS,
#     MPAC_DEFAULT,
#     STRICT,
#     BootstrapConfig,
#     ExcelConfig,
#     GiniConfig,
#     GraphsConfig,
#     PdfConfig,
#     RatioStudyConfig,
# )

# # Phase 2+ imports will be added here as modules are created.
# # e.g.:
# # from ratio_study.ratio_study import ratios, ratios_report, analyze
# # from ratio_study.ratio_study import gini_stats, gini_decomp, vei_analysis

# __all__ = [
#     "__version__",
#     # config models
#     "RatioStudyConfig",
#     "BootstrapConfig",
#     "GiniConfig",
#     "PdfConfig",
#     "GraphsConfig",
#     "ExcelConfig",
#     "KNOWN_INDICATORS",
#     # presets
#     "IAAO_RESIDENTIAL",
#     "IAAO_INCOME_PRODUCING",
#     "IAAO_VACANT_LAND",
#     "STRICT",
#     "MPAC_DEFAULT",
# ]