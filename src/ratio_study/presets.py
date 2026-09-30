"""Configuration models and named presets for ratio-study analysis.

Every run of ``ratios()`` or ``ratios_report()`` accepts a single
:class:`RatioStudyConfig` instance that centralises **all** settings —
indicator selection, IAAO assessment standards, bootstrap parameters,
Gini / vertical-equity options, graph layout, PDF / Excel export tweaks,
and error-handling policy.

Five ready-made presets are provided (``IAAO_RESIDENTIAL``, etc.) so
that common workflows require zero manual configuration.

Typical usage
-------------
>>> from ratio_study import RatioStudyConfig, IAAO_RESIDENTIAL
>>> cfg = IAAO_RESIDENTIAL.model_copy(
...     update={"model_name": "2025 Residential"}
... )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ratio_study.confint import CI_DEFAULT_INDICATORS, confidence_level_as_fraction


# ── public constant ─────────────────────────────────────────────────
KNOWN_INDICATORS: frozenset[str] = frozenset(
    {
        "Median",
        "Mean",
        "CoD",
        "CoD_residual",
        "CoD_weighted",
        "PRD",
        "PRB",
        "VEI",
        "MKI",
        "RRI",
        "RMSE",
        "MAE",
        "MAPE",
        "MdAPE",
        "CoV",
        "CoC_10",
        "CoC_15",
        "CoC_20",
        "FSD",
        # Spelled as `ratios` spells it: the study reads `'Weighted_Mean'`
        # out of `indicators`, so the lower-case form this set used to carry
        # matched nothing and rejected the real name.
        "Weighted_Mean",
        # `DEFAULT_INDICATORS` carries `'VEM'`, so leaving it out here made
        # the package's own default list fail this validator.
        "VEM",
        "gini_complete",
        "Gini_Sale_Prices",
        "Gini_Assessments",
        "Concentration_Index",
        "Modified_Kakwani_Index",
        "Kakwani_Index",
        "Redistributive_Effect",
        "Reranking_Effect",
        "Reranking_Index",
        "Count",
        "Sale_Count",
    }
)


# ── sub-models ──────────────────────────────────────────────────────
class BootstrapConfig(BaseModel):
    """Parameters that control bootstrap confidence-interval estimation.

    Attributes
    ----------
    enabled_for_population : bool
        Run bootstrap CIs on the overall (ungrouped) population.  On by
        default, matching ``ratios()``.
    enabled_for_groups : bool
        Run bootstrap CIs on each aggregation group.  Off by default: the
        cost is per group, so it is opted into.
    ci_mode : {"bootstrap", "distribution"}
        Which family of interval to build.  ``"distribution"`` (the
        default) uses the closed-form formulas of the IAAO Exposure Draft
        Appendix D for ``Median``, ``Mean``, ``Weighted_Mean`` and
        ``CoD``, plus ``PRB``'s own slope-regression interval and a
        delta-method interval for ``PRD`` (the ratio of ``Mean`` to
        ``Weighted_Mean``); ``"bootstrap"`` resamples.  ``method``,
        ``resamplings``, ``concurrency`` and ``seed`` are ignored in
        distribution mode, and so is ``min_count_threshold`` -- each
        formula has its own sample-size floor.
    method : {"BCa", "percentile"}
        Bootstrap CI method.  BCa is bias-corrected and accelerated.
    resamplings : int
        Number of bootstrap resamples (≥ 10).
    confint_level : float
        Confidence level, e.g. 0.9 for a 90 % interval -- the default, and
        the level the Standard on Ratio Studies uses to decide compliance
        (s.11.1.2).  Accepts the percentage spelling too (``90``), and
        rejects a value below 0.5 -- almost always a significance level
        passed by mistake.
    ci_only : bool
        If *True*, only the CI bounds are returned (no p-values).
    concurrency : {"threads", "processes"} or None
        Parallelism back-end.  *None* means single-threaded.
    indicators : list[str] or None
        Subset of indicators to bootstrap, intersected with the study's own
        indicators.  *None* → every requested indicator that supports a CI.
    min_count_threshold : int
        Minimum sales needed for bootstrap CIs. The whole sample and each
        group are gated on it independently; anything smaller is skipped
        and its CI columns come out blank. Lowerable to 4, the floor
        ``normalize_ci_args`` enforces, but 20 is about the point at which
        a resampled interval means anything. Ignored by
        ``ci_mode='distribution'``, whose formulas each have their own
        sample-size floor.
    seed : int or None
        Random seed for reproducibility.  *None* draws fresh entropy, so
        two identical runs publish slightly different intervals.
    """

    model_config = ConfigDict(extra="forbid")

    enabled_for_population: bool = True
    enabled_for_groups: bool = False
    ci_mode: Literal["bootstrap", "distribution"] = "distribution"
    method: Literal["BCa", "percentile"] = "BCa"
    resamplings: int = Field(default=1000, ge=10)
    confint_level: float = 0.9
    ci_only: bool = True
    concurrency: Literal["threads", "processes"] | None = None
    indicators: list[str] | None = Field(
        default_factory=lambda: list(CI_DEFAULT_INDICATORS),
    )
    # 4 is the floor `normalize_ci_args` enforces, so anything lower was
    # silently raised to it; rejecting it here says so instead.
    min_count_threshold: int = Field(default=20, ge=4)
    seed: int | None = None

    @field_validator("confint_level")
    @classmethod
    def _normalise_confint_level(cls, v: float) -> float:
        # Same gate as `ci_conf_level`, so 95 and 0.95 both mean 95% and a
        # significance level passed by mistake is rejected.
        return confidence_level_as_fraction(v, "confint_level")


class GiniConfig(BaseModel):
    """Parameters for Gini coefficient analysis and decomposition.

    Attributes
    ----------
    decomposition : list
        Grouping column name(s) for Gini decomposition.  Each element
        is either a single column name (str) or a list of column names
        for a multi-level decomposition.
    factors_variables : list[str]
        Independent-variable column names passed to ``gini_factors()``.
    lorenz_plots : bool
        Whether to generate Lorenz-curve plots.
    unbiased : bool
        Use the small-sample bias correction (*n* / (*n* − 1)).
    decomp_chart : {"bar", "pie_with_bar", ""}
        Chart type for decomposition results.  Empty string → no chart.
    run_regression : bool
        Run the auxiliary regression inside ``gini_factors()``.
    population_ratios : dict
        Pre-computed population ratios forwarded to ``gini_decomp()``.
    """

    model_config = ConfigDict(extra="forbid")

    decomposition: list = Field(default_factory=list)
    factors_variables: list[str] = Field(default_factory=list)
    lorenz_plots: bool = False
    unbiased: bool = False
    decomp_chart: Literal["bar", "pie_with_bar", ""] = "bar"
    run_regression: bool = True
    population_ratios: dict = Field(default_factory=dict)


class PdfConfig(BaseModel):
    """Options that affect PDF report generation.

    Attributes
    ----------
    grouping_aliases : dict[str, str]
        Display-name overrides for grouping columns, e.g.
        ``{"nbhd_code": "Neighbourhood"}``.
    starting_comments : list[str]
        Free-text paragraphs inserted at the top of the PDF.
    ending_comments : list[str]
        Free-text paragraphs appended at the bottom of the PDF.
    classification_header : str or None
        Optional sensitivity / classification banner printed on every
        page header (e.g. ``"CONFIDENTIAL"``).
    export_in_compacted_file : {"zip", "gzip", ""}
        Archive format when multiple PDFs are produced.  Empty string →
        no compression (individual files).
    """

    model_config = ConfigDict(extra="forbid")

    grouping_aliases: dict[str, str] = Field(default_factory=dict)
    starting_comments: list[str] = Field(default_factory=list)
    ending_comments: list[str] = Field(default_factory=list)
    classification_header: str | None = None
    export_in_compacted_file: Literal["zip", "gzip", ""] = "zip"


class GraphsConfig(BaseModel):
    """Layout and styling options for ratio-study charts.

    Attributes
    ----------
    overall_graphs : dict
        Graph definitions applied to the overall population.  Keys are
        graph names; values are dicts with ``col_y``, ``col_x``, etc.
    group_graphs : dict
        Per-group graph definitions.
    group_mosaic_graphs : dict
        Mosaic (small-multiple) graph definitions by group.
    dist_group_mosaic_graphs : dict
        Distribution-oriented mosaic graphs by group.
    facet_col_limit_per_page : int or None
        Maximum facet columns per page.  *None* → auto.
    discrete_variables : list[str]
        Columns that should be treated as categorical on axes.
    fontsize : int
        Base font size (pt) for graph text.
    lowess : float
        LOWESS smoother fraction (0–1).
    zoom_in : bool
        If *True*, axis limits are tightened to the data range.
    """

    model_config = ConfigDict(extra="forbid")

    overall_graphs: dict = Field(default_factory=dict)
    group_graphs: dict = Field(default_factory=dict)
    group_mosaic_graphs: dict = Field(default_factory=dict)
    dist_group_mosaic_graphs: dict = Field(default_factory=dict)
    facet_col_limit_per_page: int | None = None
    discrete_variables: list[str] = Field(default_factory=list)
    fontsize: int = 14
    lowess: float = Field(default=0.5, ge=0, le=1)
    zoom_in: bool = False


class ExcelConfig(BaseModel):
    """Options specific to Excel / CSV export.

    Attributes
    ----------
    sensitivity_tag : str or None
        Text stamped in a header row of every exported sheet.
    site_id : str or None
        Identifier appended to file names for multi-site runs.
    """

    model_config = ConfigDict(extra="forbid")

    sensitivity_tag: str | None = None
    site_id: str | None = None


# ── top-level config ────────────────────────────────────────────────
class RatioStudyConfig(BaseModel):
    """Master configuration for a single ratio-study execution.

    All parameters required by ``ratios()`` and ``ratios_report()`` are
    collected here so that a run is fully reproducible from one
    serialisable object.

    Attributes
    ----------
    indicators : list[str]
        Which ratio-study statistics to compute.
    assessment_standards : dict
        IAAO-style upper/lower bounds keyed by indicator name.
        Each value is a ``(lower, upper)`` tuple **or** a free-form
        string (for ``Mkt_desc``).
    req_significance : float
        Required significance level for statistical tests.
    vei_conf_level : float
        Confidence level used by the VEI (Vertical Equity Index) test.
        Accepts a fraction or a percentage.
    vem_conf_level : float
        Confidence level for the VEM's local-median band, which drives the
        VEM verdict.  Accepts a fraction or a percentage.
    vem_ci_method : {"binom", "R6", "R7"}
        Median-CI family behind the VEM band.
    vei_percentile_method : {"R6", "R7"}
        Percentile interpolation used to form the VEI quantile groups; the
        VEI's median-CI family follows it.
    iqr_trim_multiple : float or None
        IQR multiplier for outlier trimming.  *None* disables trimming.
    iqr_log_transform : bool
        Compute the IQR trimming bounds on log(ASR) rather than the raw
        ratio, since ASR is right-skewed.
    iqr_group_by : str or list, optional
        Column(s) to compute the IQR trim independently within, instead
        of once over the whole sample. Same shape as `group_by`.
    iqr_min_cod : float
        Minimum CoD a sample (or group) must have for the IQR trim to
        run on it, so low-dispersion samples aren't stripped of
        legitimate sales.
    descriptive : bool
        Include descriptive statistics in output.
    round_values : bool
        Round numeric results for display.
    log_transform : bool
        Apply a log transform to ratios before computing statistics.
    compare_groups_median_to_overall_median : bool
        Compare each group's median to the population median.
    bundle_vertical_equity_comments : bool
        Combine VE narrative into a single comment block.
    model_name : str
        Human-readable label for the model / valuation run.
    on_error : {"errors_zip", "raise", "warn"}
        Error-handling policy when a group computation fails.
    bootstrap : BootstrapConfig
        Bootstrap CI settings.
    gini : GiniConfig
        Gini / vertical-equity decomposition settings.
    pdf : PdfConfig
        PDF export settings.
    graphs : GraphsConfig
        Chart / graph layout settings.
    excel : ExcelConfig
        Excel / CSV export settings.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    indicators: list[str] = Field(
        default=["Median", "Mean", "Weighted_Mean", "CoD", "PRD", "PRB", "VEI"],
    )
    assessment_standards: dict = Field(
        default_factory=lambda: {
            "Median": (0.90, 1.10),
            "CoD": (0.05, 0.10),
            "PRD": (0.98, 1.03),
            "PRB": (-0.10, 0.10),
            "Modified_Kakwani_Index": (0.95, 1.05),
            "Mkt_desc": "Strict",
        },
    )
    req_significance: float = Field(default=0.05, gt=0, lt=1)
    vei_conf_level: float = 0.90
    vem_conf_level: float = 0.90
    vem_ci_method: Literal["binom", "R6", "R7"] = "binom"
    vei_percentile_method: Literal["R6", "R7"] = "R7"
    iqr_trim_multiple: float | None = Field(
        default=None,
        description="IQR multiplier for outlier trimming. None = no trimming.",
    )
    iqr_log_transform: bool = Field(
        default=True,
        description="Compute IQR trimming bounds on log(ASR) rather than the raw ratio.",
    )
    iqr_group_by: str | list | None = Field(
        default=None,
        description="Column(s) to compute the IQR trim independently within. Same shape as group_by.",
    )
    iqr_min_cod: float = Field(
        default=0.06,
        ge=0,
        description="Minimum CoD a sample (or group) must have for the IQR trim to run on it.",
    )
    descriptive: bool = True
    round_values: bool = True
    log_transform: bool = False
    compare_groups_median_to_overall_median: bool = True
    bundle_vertical_equity_comments: bool = False
    model_name: str = "model"
    on_error: Literal["errors_zip", "raise", "warn"] = "errors_zip"

    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    gini: GiniConfig = Field(default_factory=GiniConfig)
    pdf: PdfConfig = Field(default_factory=PdfConfig)
    graphs: GraphsConfig = Field(default_factory=GraphsConfig)
    excel: ExcelConfig = Field(default_factory=ExcelConfig)

    # -- validators ---------------------------------------------------
    @field_validator("vei_conf_level", "vem_conf_level")
    @classmethod
    def _normalise_conf_levels(cls, v: float, info) -> float:
        return confidence_level_as_fraction(v, info.field_name)

    @field_validator("iqr_trim_multiple")
    @classmethod
    def _iqr_must_be_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            msg = f"iqr_trim_multiple must be > 0, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _check_indicators(self) -> "RatioStudyConfig":
        bad = sorted(set(self.indicators) - KNOWN_INDICATORS)
        if bad:
            msg = (
                f"Unknown indicator(s): {bad}. "
                f"Valid indicators: {sorted(KNOWN_INDICATORS)}"
            )
            raise ValueError(msg)
        return self


# ── named presets ───────────────────────────────────────────────────
_SHARED_STANDARDS: dict = {
    "Median": (0.90, 1.10),
    "PRD": (0.98, 1.03),
    "PRB": (-0.10, 0.10),
    "Modified_Kakwani_Index": (0.95, 1.05),
}

IAAO_RESIDENTIAL: RatioStudyConfig = RatioStudyConfig(
    assessment_standards={
        **_SHARED_STANDARDS,
        "CoD": (0.05, 0.15),
        "Mkt_desc": "IAAO Residential Improved",
    },
)

IAAO_INCOME_PRODUCING: RatioStudyConfig = RatioStudyConfig(
    assessment_standards={
        **_SHARED_STANDARDS,
        "CoD": (0.05, 0.20),
        "Mkt_desc": "IAAO Income-Producing",
    },
)

IAAO_VACANT_LAND: RatioStudyConfig = RatioStudyConfig(
    assessment_standards={
        **_SHARED_STANDARDS,
        "CoD": (0.05, 0.25),
        "Mkt_desc": "IAAO Vacant Land",
    },
)

STRICT: RatioStudyConfig = RatioStudyConfig(
    assessment_standards={
        **_SHARED_STANDARDS,
        "CoD": (0.05, 0.10),
        "Mkt_desc": "Strict",
    },
)


# ── public API ──────────────────────────────────────────────────────
__all__ = [
    "KNOWN_INDICATORS",
    "BootstrapConfig",
    "ExcelConfig",
    "GiniConfig",
    "GraphsConfig",
    "PdfConfig",
    "RatioStudyConfig",
    "IAAO_RESIDENTIAL",
    "IAAO_INCOME_PRODUCING",
    "IAAO_VACANT_LAND",
    "STRICT",
]