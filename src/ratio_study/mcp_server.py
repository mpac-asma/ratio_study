"""
MCP (Model Context Protocol) server for the `ratio_study` package.

Exposes the `ratios()` function and related utilities as tools that LLMs
(Claude Desktop, VS Code Copilot Chat, Cursor, Continue.dev, etc.) can call
via the MCP standard.

Run locally with stdio transport:
    python -m ratio_study.mcp_server

Add to Claude Desktop config (~/.config/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "ratio_study": {
          "command": "python",
          "args": ["-m", "ratio_study.mcp_server"]
        }
      }
    }
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# MCP SDK is optional at import time; the server is only useful when installed.
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - allow import without mcp installed
    FastMCP = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

# Written by `ratios_report` when it catches a failure internally.
_ERRORS_FILENAME = 'errors_in_ratio_study.txt'

from ratio_study import ratios
from ratio_study._version import __version__
# Read rather than restated: this list drifted from `ratios`' own default
# once already, and the response reports it back as what was computed.
from ratio_study.ratio_study import DEFAULT_INDICATORS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cap on file size (in bytes) the server will load. 500 MB default.
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

# Cap on row count returned by inspect_dataset.
MAX_INSPECT_ROWS = 5

# Plain (uncompressed) dataset extensions.
_PLAIN_EXTS = (".csv", ".parquet", ".xlsx", ".xls", ".json", ".tsv", ".sav")

# Compound (compacted) suffixes. Each entry is the full suffix string paired
# with a callable that reads the file in one pandas call — pandas handles
# the container/codec transparently, so no temp files or manual extraction
# is required.
_COMPOUND_SUFFIX_READERS: Dict[str, Any] = {
    # Outer gzip around a CSV (CSV is gzip-compressed at file level).
    ".csv.gz":      lambda p: pd.read_csv(p, compression="infer"),
    # Parquet file with internal gzip codec — pandas reads the .parquet.gz
    # path directly, no special args.
    ".parquet.gz":  lambda p: pd.read_parquet(p),
    # Outer zip around a CSV (CSV is the zip entry).
    ".csv.zip":     lambda p: pd.read_csv(p, compression="zip"),
    # Outer zip around a parquet file — pyarrow/pandas reads it directly.
    ".parquet.zip": lambda p: pd.read_parquet(p),
}

# All extensions we accept, in a stable order used for error messages and
# the ``get_module_capabilities`` advertising endpoint.
_ALL_EXTS: List[str] = list(_PLAIN_EXTS) + list(_COMPOUND_SUFFIX_READERS)


def _read_sav(path: Path) -> pd.DataFrame:
    """
    Read an SPSS .sav file using pyreadstat.

    Parameters
    ----------
    path : Path
        Path to the ``.sav`` file to read.

    Returns
    -------
    pandas.DataFrame
        The data read from the file, with value labels preserved by
        pyreadstat.
    """
    try:
        import pyreadstat  # type: ignore
    except ImportError as ex:
        raise ImportError(
            "Reading SPSS .sav files requires the 'pyreadstat' package. "
            "Install it with: pip install pyreadstat"
        ) from ex
    df, _meta = pyreadstat.read_sav(str(path))
    return df


def _plain_reader_for(ext: str):
    """
    Return a callable that reads an uncompressed file with the given extension.

    Parameters
    ----------
    ext : str
        File extension (e.g. ``".csv"``, ``".xlsx"``) to get a reader for.

    Returns
    -------
    callable
        A function that takes a path and returns a ``pandas.DataFrame``.
    """
    if ext == ".csv":
        return lambda p: pd.read_csv(p)
    if ext == ".tsv":
        return lambda p: pd.read_csv(p, sep="\t")
    if ext == ".parquet":
        return lambda p: pd.read_parquet(p)
    if ext == ".xlsx":
        return lambda p: pd.read_excel(p)
    if ext == ".xls":
        return lambda p: pd.read_excel(p)
    if ext == ".json":
        return lambda p: pd.read_json(p)
    if ext == ".sav":
        return _read_sav
    raise ValueError(f"No plain reader for extension '{ext}'")


def _resolve_reader(path: Path) -> Tuple[str, Any]:
    """
    Pick the right pandas reader for a dataset path.

    Parameters
    ----------
    path : Path
        Path to the dataset file. Lookup order:

        * the full filename (e.g. ``sales.csv.gz``) -- covers compound forms.
        * the lowercased final extension (e.g. ``.csv``) -- covers plain forms.

    Returns
    -------
    tuple of (str, callable or None)
        The matched suffix/extension and its reader callable, or
        ``(ext, None)`` if no reader is registered for the extension.
    """
    name_lower = path.name.lower()
    for suffix, reader in _COMPOUND_SUFFIX_READERS.items():
        if name_lower.endswith(suffix):
            return suffix, reader
    ext = path.suffix.lower()
    if ext in _PLAIN_EXTS:
        return ext, _plain_reader_for(ext)
    return ext, None


# ---------------------------------------------------------------------------
# JSON sanitisation helpers
# ---------------------------------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    """
    Recursively convert numpy / pandas / polars / pyarrow / narwhals objects into JSON-serialisable types.

    Parameters
    ----------
    value : Any
        Object to convert. May be a dict, list, tuple, DataFrame/Series
        (pandas, polars, or narwhals-wrapped), pyarrow Table/ChunkedArray,
        numpy scalar/array, or a plain JSON-compatible value.

    Returns
    -------
    Any
        A value composed only of ``dict``, ``list``, ``str``, ``int``,
        ``float``, ``bool``, and ``None`` -- safe to pass to
        ``json.dumps``. NaN/Inf floats are converted to ``None``.
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    # Handle narwhals-wrapped frames first (they're not in the native
    # frame checks below because ``isinstance`` against the narwhals
    # type succeeds regardless of the underlying backend).
    try:
        import narwhals as nw
        if isinstance(value, (nw.DataFrame, nw.LazyFrame)):
            try:
                native = value.to_native() if isinstance(value, nw.DataFrame) else value.collect().to_native()
            except Exception:
                native = value
            return _to_jsonable(native)
        if isinstance(value, nw.Series):
            return _to_jsonable(value.to_native())
    except ImportError:
        pass
    if isinstance(value, pd.DataFrame):
        # Replace NaN with None for clean JSON; cap very wide frames.
        return _to_jsonable(value.replace({np.nan: None}).to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _to_jsonable(value.replace({np.nan: None}).tolist())
    # polars frames: also serialise as records.  Imported lazily because
    # polars is an optional dependency of the package.
    try:
        import polars as pl
        if isinstance(value, pl.DataFrame):
            return _to_jsonable(value.to_dicts())
        if isinstance(value, pl.Series):
            return _to_jsonable(value.to_list())
    except ImportError:
        pass
    # pyarrow tables: convert to pandas for the dict-of-records path.
    try:
        import pyarrow as pa
        if isinstance(value, pa.Table):
            return _to_jsonable(value.to_pandas().replace({np.nan: None}).to_dict(orient="records"))
        if isinstance(value, pa.ChunkedArray):
            return _to_jsonable(value.to_pylist())
    except ImportError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value
    # Fallback: stringify unknown objects
    return str(value)


def _error_response(message: str, exc: Optional[BaseException] = None) -> Dict[str, Any]:
    """
    Return a structured error dict for the LLM to self-correct on.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    exc : BaseException, optional
        The exception that triggered the error, if any. When given, its
        type name and a formatted traceback are included in the payload.

    Returns
    -------
    dict
        JSON-serialisable payload with keys ``"error"`` and, if ``exc``
        was given, ``"exception_type"`` and ``"traceback"``.
    """
    payload: Dict[str, Any] = {"error": message}
    if exc is not None:
        payload["exception_type"] = type(exc).__name__
        payload["traceback"] = traceback.format_exc(limit=5)
    return _to_jsonable(payload)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _read_dataset(file_path: str) -> pd.DataFrame:
    """
    Read a dataset from disk, validating path, size, and extension.

    Parameters
    ----------
    file_path : str
        Path to the dataset file. Supported suffixes are ``.csv``,
        ``.tsv``, ``.parquet``, ``.xlsx``, ``.xls``, ``.json``, ``.sav``
        and the compound forms ``.csv.gz``, ``.parquet.gz``, ``.csv.zip``,
        ``.parquet.zip``. Compressed variants are decoded with pandas'
        native ``compression=`` support, so no temp files or manual
        extraction is performed.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a regular file: {p}")

    suffix, reader = _resolve_reader(p)
    if reader is None:
        supported = ", ".join(_ALL_EXTS)
        raise ValueError(
            f"Unsupported file extension '{p.suffix}'. "
            f"Supported: {supported}"
        )

    size = p.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large ({size / 1024 / 1024:.1f} MB). "
            f"Limit is {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB."
        )

    return reader(p)


def _is_supported_dataset(path: Path) -> bool:
    """
    Return True if ``path`` is a regular file with a supported extension.

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    bool
        True if ``path`` is a regular file whose extension has a
        registered reader.
    """
    if not path.is_file():
        return False
    suffix, reader = _resolve_reader(path)
    return reader is not None


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

if FastMCP is None:
    # Defer the hard error to runtime so the rest of the package still imports.
    _MCP_IMPORT_ERROR = (
        "The `mcp` package is required for the MCP server. "
        "Install it with: pip install mcp"
    )
else:
    _MCP_IMPORT_ERROR = None

    mcp = FastMCP(
        "ratio_study",
        instructions=(
            "Tools for running IAAO Standards' based ratio studies on property "
            "assessment data. Typical workflow: list_datasets → "
            "inspect_dataset → run_ratio_study. Always inspect the data "
            "first so you can suggest correct column names to the user."
        ),
    )

    # -----------------------------------------------------------------------
    # Tool: get_module_capabilities
    # -----------------------------------------------------------------------
    @mcp.tool()
    def get_module_capabilities() -> Dict[str, Any]:
        """
        Return information about this `ratio_study` MCP server.

        Use this tool to discover what indicators and options are available
        before asking the user questions.

        Returns
        -------
        dict
            Server metadata with keys:

            * name : str -- server name.
            * version : str -- package version.
            * max_file_size_mb : float -- maximum accepted file size.
            * supported_file_extensions : list of str -- accepted dataset extensions.
            * available_indicators : list of str -- indicator names usable in ``run_ratio_study``.
            * typical_workflow : list of str -- suggested tool call order.
        """
        return _to_jsonable({
            "name": "ratio_study",
            "version": __version__,
            "max_file_size_mb": MAX_FILE_SIZE_BYTES / 1024 / 1024,
            "supported_file_extensions": sorted(_ALL_EXTS),
            "available_indicators": [
                "Median", "Mean", "CoD", "PRD", "PRB", "VEI",
                "RMSE", "MAE", "MAPE", "MdAPE", "CoV", "CoC_10",
                "CoC_15", "CoC_20", "MKI", "RRI", "gini_complete",
                "VEM",
            ],
            "typical_workflow": [
                "1. list_datasets(directory) – discover available files.",
                "2. inspect_dataset(file_path) – see columns & dtypes.",
                "3. run_ratio_study(file_path, market_value_col, "
                "assessment_col, ...) – run the analysis.",
                "4. generate_ratio_report(...) – write a PDF report.",
                "5. generate_vertical_equity_panel(...) – write the VE panel PNG.",
            ],
            "tools": [
                "get_module_capabilities", "diagnose_setup", "list_datasets",
                "inspect_dataset", "run_ratio_study", "generate_ratio_report",
                "generate_vertical_equity_panel",
            ],
        })

    # -----------------------------------------------------------------------
    # Tool: diagnose_setup
    # -----------------------------------------------------------------------
    @mcp.tool()
    def diagnose_setup() -> Dict[str, Any]:
        """
        Run a self-check and report the install / environment status.

        Use this when a user says things like "is it set up right?" or
        "something isn't working." The tool checks Python version, package
        presence, optional dependencies, and reports the recommended
        connector command for the user's OS.

        Returns
        -------
        dict
            Diagnostic report with keys:

            * python_version : str -- e.g. ``"3.11.4"``.
            * python_executable : str -- path to the running interpreter.
            * platform : str -- ``"Windows"``, ``"Darwin"``, or ``"Linux"``.
            * package_installed : bool -- whether ``ratio_study`` imports.
            * package_version : str or None -- installed package version.
            * mcp_installed : bool -- whether the ``mcp`` package is available.
            * pyreadstat_installed : bool -- whether SPSS ``.sav`` support is available.
            * ready_to_run : bool -- whether the server can run.
            * connector_command : str -- recommended command for an MCP connector.
            * connector_args : str -- recommended arguments for the connector command.
            * issues : list of str -- problems found.
            * next_steps : list of str -- suggested remediation steps.
        """
        import platform as _platform
        import shutil
        import subprocess
        import sys as _sys

        issues: List[str] = []
        next_steps: List[str] = []

        # 1. Python version
        py_version = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"
        if _sys.version_info < (3, 9):
            issues.append(
                f"Python {py_version} is too old. The package needs Python 3.9 or newer."
            )

        # 2. Package installed?
        pkg_version: Optional[str] = None
        pkg_installed = False
        try:
            from ratio_study._version import __version__ as _v
            pkg_version = _v
            pkg_installed = True
        except Exception as ex:
            issues.append(f"The `ratio_study` package is not importable: {ex}")
            next_steps.append(
                "Run: python -m pip install \"ratio_study[mcp]\""
            )

        # 3. MCP SDK
        mcp_installed = FastMCP is not None
        if not mcp_installed:
            issues.append("The `mcp` package is not installed.")
            next_steps.append("Run: python -m pip install mcp")

        # 4. pyreadstat (for SPSS)
        pyreadstat_installed = False
        try:
            import pyreadstat  # noqa: F401
            pyreadstat_installed = True
        except ImportError:
            next_steps.append(
                "(Optional) For SPSS .sav support, run: "
                "python -m pip install pyreadstat"
            )

        # 5. Ready to run?
        ready = pkg_installed and mcp_installed and _sys.version_info >= (3, 9)

        # 6. Recommended connector command
        os_name = _platform.system()
        py_exe = _sys.executable
        if os_name == "Windows":
            connector_command = py_exe
            connector_args = "-m ratio_study.mcp_server"
        else:
            # Use the exact same Python that has the package installed
            connector_command = py_exe
            connector_args = "-m ratio_study.mcp_server"

        if not ready:
            next_steps.insert(0, "Fix the issues above, then run diagnose_setup again.")
        else:
            next_steps.insert(
                0,
                "Everything looks good. In Copilot 365, add a new MCP "
                "connector with the command and args shown below.",
            )

        return _to_jsonable({
            "python_version": py_version,
            "python_executable": py_exe,
            "platform": os_name,
            "package_installed": pkg_installed,
            "package_version": pkg_version,
            "mcp_installed": mcp_installed,
            "pyreadstat_installed": pyreadstat_installed,
            "ready_to_run": ready,
            "connector_command": connector_command,
            "connector_args": connector_args,
            "issues": issues,
            "next_steps": next_steps,
        })

    # -----------------------------------------------------------------------
    # Tool: list_datasets
    # -----------------------------------------------------------------------
    @mcp.tool()
    def list_datasets(directory: str) -> Dict[str, Any]:
        """
        List loadable dataset files in a directory.

        Discovers plain datasets (CSV, TSV, Parquet, Excel, JSON, SAV) and
        compressed / archived variants of them (``*.csv.gz``, ``*.parquet.zip``,
        …).

        Parameters
        ----------
        directory : str
            Absolute or relative path to a directory.

        Returns
        -------
        dict
            Directory listing with keys:

            * directory : str -- resolved absolute path.
            * count : int -- number of dataset files found.
            * files : list of str -- absolute paths to the discovered files.

            On failure, an error dict is returned instead (see ``_error_response``).
        """
        try:
            d = Path(directory).expanduser().resolve()
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")
            if not d.is_dir():
                raise ValueError(f"Not a directory: {d}")

            files = sorted(
                str(p) for p in d.iterdir() if _is_supported_dataset(p)
            )
            return _to_jsonable({
                "directory": str(d),
                "count": len(files),
                "files": files,
            })
        except Exception as ex:
            return _error_response(
                f"Failed to list datasets in '{directory}': {ex}", ex
            )

    # -----------------------------------------------------------------------
    # Tool: inspect_dataset
    # -----------------------------------------------------------------------
    @mcp.tool()
    def inspect_dataset(
        file_path: str,
        n_rows: int = MAX_INSPECT_ROWS,
    ) -> Dict[str, Any]:
        """
        Inspect a dataset's columns, dtypes, and a small sample.

        Use this BEFORE calling run_ratio_study so you can confirm column
        names and data types with the user.

        Parameters
        ----------
        file_path : str
            Path to a CSV / TSV / Parquet / Excel / JSON / SAV file. The
            following compressed variants are also supported: ``.csv.gz``,
            ``.parquet.gz``, ``.csv.zip``, ``.parquet.zip``. They are read
            via pandas' native ``compression=`` / zip-archive support.
        n_rows : int, default 5
            How many sample rows to return. Capped at 20.

        Returns
        -------
        dict
            Inspection report with keys:

            * file_path : str -- resolved absolute path.
            * n_rows_total : int -- total number of rows in the dataset.
            * n_cols : int -- number of columns.
            * columns : list of dict -- ``{"name": str, "dtype": str}`` per column.
            * sample : list of dict -- up to ``n_rows`` sample records.
            * numeric_summary : dict -- mean/std/min/max etc. for numeric columns.

            On failure, an error dict is returned instead (see ``_error_response``).
        """
        try:
            n_rows = max(1, min(int(n_rows), 20))
            df = _read_dataset(file_path)

            columns = [
                {"name": str(c), "dtype": str(df[c].dtype)}
                for c in df.columns
            ]
            sample = df.head(n_rows).replace({np.nan: None}).to_dict(
                orient="records"
            )

            numeric_summary: Dict[str, Any] = {}
            try:
                desc = df.select_dtypes(include="number").describe()
                numeric_summary = desc.replace({np.nan: None}).to_dict()
            except Exception:
                pass

            return _to_jsonable({
                "file_path": str(Path(file_path).expanduser().resolve()),
                "n_rows_total": int(len(df)),
                "n_cols": int(df.shape[1]),
                "columns": columns,
                "sample": sample,
                "numeric_summary": numeric_summary,
            })
        except Exception as ex:
            return _error_response(
                f"Failed to inspect '{file_path}': {ex}", ex
            )

    # -----------------------------------------------------------------------
    # Tool: run_ratio_study
    # -----------------------------------------------------------------------
    @mcp.tool()
    def run_ratio_study(
        file_path: str,
        market_value_col: str,
        assessment_col: str,
        mkt_value_proxy_col: Optional[str] = None,
        prb_mvp_col: Optional[str] = None,
        mvp_sale_price_share: float = 0.5,
        group_by: Optional[Union[str, List[str]]] = None,
        indicators: Optional[List[str]] = None,
        log_transform: bool = False,
        iqr_trim_multiple: float = 0.0,
        iqr_log_transform: bool = True,
        iqr_group_by: Optional[Union[str, List[str]]] = None,
        iqr_min_cod: float = 0.06,
        weights_col: Optional[str] = None,
        return_expanded_dict: bool = False,
        req_significance: float = 0.05,
        confidence_intervals_aggs: Optional[Union[bool, str, List[str]]] = None,
        ci_mode: str = "distribution",
        ci_conf_level: Optional[float] = None,
        ci_indicators: Optional[List[str]] = None,
        ci_seed: Optional[int] = None,
        vei_conf_level: float = 0.9,
        vem_conf_level: float = 0.9,
        vem_acceptable_inequity_level: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run an IAAO Standards' based ratio study on a dataset.

        Parameters
        ----------
        file_path : str
            Path to the dataset. Supported suffixes are ``.csv``, ``.tsv``,
            ``.parquet``, ``.xlsx``, ``.xls``, ``.json``, ``.sav`` and the
            four compound forms ``.csv.gz``, ``.parquet.gz``, ``.csv.zip``,
            ``.parquet.zip``. Compressed variants are decoded with pandas'
            native ``compression=`` / zip-archive support — no temp files
            or manual extraction is performed.
        market_value_col : str
            Column with the time-adjusted sale price (the "true" market value).
        assessment_col : str
            Column with the assessed / predicted value.
        mkt_value_proxy_col : str, optional
            Column used as the market-value proxy for vertical-equity plots.
            Defaults to `market_value_col` if not provided.
        prb_mvp_col : str, optional
            Column used as the market-value proxy in the PRB regression,
            overriding the blend PRB builds itself from `market_value_col`
            and the ASR-adjusted assessment. Independent of
            `mkt_value_proxy_col`, which is for vertical-equity plots.
        mvp_sale_price_share : float, default 0.5
            Share of the sale price in the market-value proxy built for the
            PRB regression and the VEI grouping; the assessment restated at
            the median ASR takes the rest. 0.5 is the Standard's 50/50
            proxy; 0.3 gives a 30/70 one. Ignored for whichever proxy is
            supplied as a column.
        group_by : str or list of str, optional
            Column(s) to group by. Examples: "neighborhood" or
            ["neighborhood", "property_type"].
        indicators : list of str, optional
            Indicators to compute. Defaults to a core set (Median, Mean,
            CoD, PRD, PRB, VEI, VEM). Available: Median, Mean, CoD, PRD,
            PRB, VEI, VEM, RMSE, MAE, MAPE, MdAPE, CoV, CoC_10, CoC_15,
            CoC_20, MKI, RRI, gini_complete.
        log_transform : bool, default False
            True if prices are log-transformed in the source data.
        iqr_trim_multiple : float, default 0.0
            IQR multiplier for outlier trimming (0 disables).
        iqr_log_transform : bool, default True
            Compute the IQR trimming bounds on log(ASR) rather than the
            raw ratio, since ASR is right-skewed.
        iqr_group_by : str or list of str, optional
            Column(s) to compute the IQR trim independently within,
            instead of once over the whole sample.
        iqr_min_cod : float, default 0.06
            Minimum CoD a sample (or group) must have for the IQR trim
            to run on it, so low-dispersion samples aren't stripped of
            legitimate sales.
        weights_col : str, optional
            Column with sample weights.
        return_expanded_dict : bool, default False
            If True, also returns a nested dict with extra detail.
        req_significance : float, default 0.05
            p-value threshold for the PRB significance flag. This is a
            *significance* level; the confidence intervals are controlled
            by `ci_conf_level` below.
        confidence_intervals_aggs : bool, str, or list of str, optional
            Which intervals to compute. None keeps the default (overall
            only); False switches them off; 'both' / 'all' adds per-group
            intervals; a list names the indicators to cover.
        ci_mode : {'bootstrap', 'distribution'}, default 'distribution'
            Which family of interval to build. 'distribution' uses the
            closed-form formulas of IAAO Exposure Draft Appendix D for
            Median, Mean, Weighted_Mean and CoD, PRB's own slope-regression
            interval, and a delta-method interval for PRD (the ratio of
            Mean to Weighted_Mean); it has no minimum-count gate, so it
            reports intervals for small samples the bootstrap skips.
            'bootstrap' resamples.
        ci_conf_level : float, optional
            Confidence level for the intervals, as a fraction (0.9) or a
            percentage (90). Defaults to 0.90. Values below 0.5 are
            rejected as a significance level passed by mistake.
        ci_indicators : list of str, optional
            Indicators to build intervals for, intersected with
            `indicators`.
        ci_seed : int, optional
            Seed for the bootstrap resampler, for reproducible intervals.
            Unused when `ci_mode='distribution'`.
        vei_conf_level : float, default 0.9
            Confidence level for the VEI's median intervals (0.9 = 90%).
        vem_conf_level : float, default 0.9
            Confidence level for the VEM's local-median band, which drives
            the VEM verdict.
        vem_acceptable_inequity_level : float, default 0.1
            Acceptable inequity band for VEM (e.g. 0.1 = ±10%).

        Returns
        -------
        dict
            Result payload with keys:

            * file_path : str -- resolved absolute path.
            * rows_used : int -- number of rows read from the dataset.
            * columns_used : dict -- the market_value/assessment/group_by/weights/mkt_value_proxy columns used.
            * indicators_requested : list of str -- indicators computed.
            * ratios : dict or None -- formatted results DataFrame as list-of-dicts.
            * ratios_headline : list of dict or None -- first 50 rows of ``ratios``, for quoting verbatim.
            * expanded : dict, optional -- present only if ``return_expanded_dict=True``; a slimmed copy of the full result dict with plot objects and raw data removed.

            On failure, an error dict is returned instead (see ``_error_response``).
        """
        try:
            df = _read_dataset(file_path)

            # Validate that the requested columns exist.
            missing = []
            for col in (market_value_col, assessment_col):
                if col not in df.columns:
                    missing.append(col)
            if weights_col and weights_col not in df.columns:
                missing.append(weights_col)
            if mkt_value_proxy_col and mkt_value_proxy_col not in df.columns:
                missing.append(mkt_value_proxy_col)
            if prb_mvp_col and prb_mvp_col not in df.columns:
                missing.append(prb_mvp_col)
            if group_by:
                groups = [group_by] if isinstance(group_by, str) else list(group_by)
                for g in groups:
                    if g not in df.columns:
                        missing.append(g)
            if missing:
                raise ValueError(
                    f"Column(s) not found in dataset: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

            kwargs: Dict[str, Any] = dict(
                input_df=df,
                assessment_header=assessment_col,
                market_value_header=market_value_col,
                # Nothing may print: on a stdio transport, stdout *is* the
                # JSON-RPC channel, and one stray line corrupts the stream.
                # The settings the summary would report are returned in the
                # response instead.
                print_ci_settings=False,
                iqr_filter_result=None,
                log_transform=log_transform,
                iqr_trim_multiple=iqr_trim_multiple,
                iqr_log_transform=iqr_log_transform,
                iqr_group_by=iqr_group_by,
                iqr_min_cod=iqr_min_cod,
                req_significance=req_significance,
                ci_mode=ci_mode,
                vei_conf_level=vei_conf_level,
                vem_conf_level=vem_conf_level,
                vem_acceptable_inequity_level=vem_acceptable_inequity_level,
                return_dict=return_expanded_dict,
            )
            # Left out of the dict above so that "not specified" stays
            # distinguishable from an explicit value -- `normalize_ci_args`
            # gives None its own meaning for several of these.
            if confidence_intervals_aggs is not None:
                kwargs["confidence_intervals_aggs"] = confidence_intervals_aggs
            if ci_conf_level is not None:
                kwargs["ci_conf_level"] = ci_conf_level
            if ci_indicators:
                kwargs["ci_indicators"] = ci_indicators
            if ci_seed is not None:
                kwargs["ci_seed"] = ci_seed
            if mkt_value_proxy_col:
                kwargs["mkt_value_proxy_header"] = mkt_value_proxy_col
            if prb_mvp_col:
                kwargs["prb_mvp_header"] = prb_mvp_col
            kwargs["mvp_sale_price_share"] = mvp_sale_price_share
            if group_by:
                kwargs["group_by"] = group_by
            if indicators:
                kwargs["indicators"] = indicators
            if weights_col:
                kwargs["weights_header"] = weights_col

            result = ratios(**kwargs)

            # Pull out the formatted DataFrame; the rest is meta / dict.
            if isinstance(result, pd.DataFrame):
                formatted = result
                expanded = None
            elif isinstance(result, dict):
                formatted = result.get("df_all_ratios_formatted")
                expanded = result
            else:
                formatted = pd.DataFrame({"result": [result]})
                expanded = None

            # Extract a small "headline" subset the LLM can quote verbatim.
            headline: Optional[List[Dict[str, Any]]] = None
            try:
                if isinstance(formatted, pd.DataFrame) and not formatted.empty:
                    headline = (
                        formatted.replace({np.nan: None})
                        .head(50)
                        .to_dict(orient="records")
                    )
            except Exception:
                headline = None

            response: Dict[str, Any] = {
                "file_path": str(Path(file_path).expanduser().resolve()),
                "rows_used": int(len(df)),
                "columns_used": {
                    "market_value": market_value_col,
                    "assessment": assessment_col,
                    "group_by": group_by,
                    "weights": weights_col,
                    "mkt_value_proxy": mkt_value_proxy_col,
                    "prb_mvp": prb_mvp_col,
                },
                "indicators_requested": (
                    indicators
                    or list(DEFAULT_INDICATORS)
                ),
                "ratios": _to_jsonable(formatted) if formatted is not None else None,
                "ratios_headline": headline,
            }
            if return_expanded_dict and expanded is not None:
                # Strip out the heavy plot objects & raw DataFrames to keep
                # the response small; the LLM only needs the metrics.
                slim_expanded = {
                    k: v for k, v in expanded.items()
                    if k not in ("vertical_equity_panel", "ve_median_plot",
                                 "vei_plot", "data")
                }
                response["expanded"] = _to_jsonable(slim_expanded)
            return response
        except Exception as ex:
            return _error_response(
                f"run_ratio_study failed: {ex}", ex
            )


    # -----------------------------------------------------------------------
    # Tool: generate_ratio_report
    # -----------------------------------------------------------------------
    @mcp.tool()
    def generate_ratio_report(
        file_path: str,
        market_value_col: str,
        assessment_col: str,
        output_directory: str,
        mkt_value_proxy_col: Optional[str] = None,
        prb_mvp_col: Optional[str] = None,
        mvp_sale_price_share: float = 0.5,
        model_name: str = "model",
        group_by: Optional[Union[str, List[str]]] = None,
        indicators: Optional[List[str]] = None,
        base_date: str = "",
        descriptive: bool = True,
        export_ratios: str = "",
        opening_comments: Optional[List[str]] = None,
        closing_comments: Optional[List[str]] = None,
        include_vertical_equity_panel: bool = True,
        iqr_trim_multiple: float = 0.0,
        ci_conf_level: Optional[float] = None,
        ci_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Produce a PDF ratio study report, optionally with the data exported alongside.

        Wraps :func:`ratio_study.ratios_report`. The PDF carries the ratio
        tables, the acceptance commentary, the charts and (by default) the
        Vertical Equity Panel -- the exhibits section 10.1.2 of the Standard
        on Ratio Studies asks a published study to include.

        Parameters
        ----------
        file_path : str
            Dataset to read. Same formats as ``run_ratio_study``.
        market_value_col, assessment_col : str
            Time-adjusted sale price column, and the assessed value column.
        output_directory : str
            Existing directory to write the report into.
        mkt_value_proxy_col : str, optional
            Column used as the market-value proxy for the VEI and the
            Vertical Equity Panel. Independent of `prb_mvp_col`.
        prb_mvp_col : str, optional
            Column used as the market-value proxy in the PRB regression,
            overriding the blend PRB builds itself from `market_value_col`
            and the ASR-adjusted assessment.
        mvp_sale_price_share : float, default 0.5
            Share of the sale price in the market-value proxy built for the
            PRB regression and the VEI grouping. 0.5 is the Standard's
            50/50 proxy; 0.3 gives a 30/70 one.
        model_name : str, default 'model'
            Name shown in the report title and used in the filename.
        group_by : str or list of str, optional
            Stratification, e.g. "neighbourhood" or ["neighbourhood", "property_type"].
        indicators : list of str, optional
            Indicators to compute; see ``get_module_capabilities``.
        base_date : str, default ''
            Valuation date shown on the report, e.g. '20260101'.
        descriptive : bool, default True
            Add plain-language acceptance comments where a standard is not met.
        export_ratios : {'', 'csv', 'xlsx', 'both'}, default ''
            Also write the numbers beside the PDF.
        opening_comments, closing_comments : list of str, optional
            Narrative paragraphs placed at the start and end of the PDF.
        include_vertical_equity_panel : bool, default True
            Include the PRB / PRD / VEI / VEM panel.
        iqr_trim_multiple : float, default 0.0
            IQR outlier trim; 0 disables it.
        ci_conf_level : float, optional
            Confidence level, as a fraction (0.9) or a percentage (90).
        ci_seed : int, optional
            Seed, so the published intervals are reproducible.

        Returns
        -------
        dict
            ``files_written`` (list of paths), ``output_directory``,
            ``rows_used``, and ``ratios_headline`` -- or an error dict.
        """
        try:
            from ratio_study import ratios_report

            out_dir = Path(output_directory).expanduser()
            if not out_dir.is_dir():
                return _error_response(
                    f"output_directory is not an existing directory: {out_dir}")

            df = _read_dataset(file_path)
            before = {p.name for p in out_dir.iterdir()}

            pdf_args: Dict[str, Any] = {
                "pdf_vertical_equity_panel": bool(include_vertical_equity_panel),
            }
            if opening_comments:
                pdf_args["pdf_starting_comments"] = list(opening_comments)
            if closing_comments:
                pdf_args["pdf_ending_comments"] = list(closing_comments)

            result = ratios_report(
                df,
                assessment_header=assessment_col,
                market_value_header=market_value_col,
                mkt_value_proxy_header=mkt_value_proxy_col,
                prb_mvp_header=prb_mvp_col,
                mvp_sale_price_share=mvp_sale_price_share,
                group_by=group_by,
                indicators=indicators,
                descriptive=descriptive,
                model_name=model_name,
                base_date=base_date,
                export_directory=str(out_dir),
                export_ratios=export_ratios,
                export_in_compacted_file="",
                iqr_trim_multiple=iqr_trim_multiple,
                ci_conf_level=ci_conf_level,
                ci_seed=ci_seed,
                print_ci_settings=False,
                iqr_filter_result=None,
                return_dict=False,
                pdf_args=pdf_args,
            )

            written = sorted(p.name for p in out_dir.iterdir() if p.name not in before)

            # `ratios_report` catches its own failures into an errors file and
            # returns normally, so "wrote only that file" is a failed report.
            # An assistant needs to be told, not handed an empty success.
            reports = [f for f in written if f != _ERRORS_FILENAME]
            if not reports:
                detail = ''
                errors_path = out_dir / _ERRORS_FILENAME
                if errors_path.exists():
                    detail = errors_path.read_text()[:1000].strip()
                return _error_response(
                    'generate_ratio_report produced no report. '
                    + (f'The package reported: {detail}' if detail
                       else 'No error detail was written.'))

            headline = None
            if isinstance(result, pd.DataFrame) and not result.empty:
                headline = _to_jsonable(result.iloc[0].to_dict())

            return _to_jsonable({
                "output_directory": str(out_dir),
                "files_written": reports,
                "rows_used": int(len(df)),
                "ratios_headline": headline,
            })
        except Exception as ex:
            return _error_response(f"generate_ratio_report failed: {ex}", ex)

    # -----------------------------------------------------------------------
    # Tool: generate_vertical_equity_panel
    # -----------------------------------------------------------------------
    @mcp.tool()
    def generate_vertical_equity_panel(
        file_path: str,
        market_value_col: str,
        assessment_col: str,
        output_directory: str,
        mkt_value_proxy_col: Optional[str] = None,
        prb_mvp_col: Optional[str] = None,
        mvp_sale_price_share: float = 0.5,
        filename: str = "vertical_equity_panel.png",
        title: str = "Vertical Equity Panel",
        subtitle: str = "",
        vei_conf_level: float = 0.9,
        vem_conf_level: float = 0.9,
        vem_acceptable_inequity_level: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Render the Vertical Equity Panel to a PNG file.

        Four acceptance gauges (PRB, PRD, VEI, VEM) over the VEM moving-median
        chart and the VEI quantile boxplot, with the Appendix F supporting
        tests in a strip beneath the gauges.

        Parameters
        ----------
        file_path : str
            Dataset to read. Same formats as ``run_ratio_study``.
        market_value_col, assessment_col : str
            Time-adjusted sale price column, and the assessed value column.
        output_directory : str
            Existing directory to write the PNG into.
        mkt_value_proxy_col : str, optional
            Column used as the market-value proxy for the VEI grouping.
            When omitted, the IAAO proxy (`mvp_sale_price_share` * market
            value + the remainder * assessment / median ASR) is built
            internally. Independent of `prb_mvp_col`.
        prb_mvp_col : str, optional
            Column used as the market-value proxy in the PRB regression,
            overriding the blend PRB builds itself from `market_value_col`
            and the ASR-adjusted assessment. This panel recomputes PRB itself
            (there is no precomputed value to reuse here), so this is the
            only way to steer that regression's proxy through this tool.
        filename : str, default 'vertical_equity_panel.png'
            Output filename; '.png' is appended if missing.
        title, subtitle : str
            Headings drawn on the panel.
        mvp_sale_price_share : float, default 0.5
            Share of the sale price in the market-value proxy built for the
            VEI grouping and the PRB regression. 0.5 is the Standard's
            50/50 proxy; 0.3 gives a 30/70 one.
        vei_conf_level, vem_conf_level : float, default 0.9
            Confidence levels for the VEI interval and the VEM band.
        vem_acceptable_inequity_level : float, default 0.1
            Acceptable inequity band for the VEM, as a fraction.

        Returns
        -------
        dict
            ``file_written`` (full path), ``rows_used``, and the four gauge
            values under ``verdicts`` -- or an error dict.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            from ratio_study import ratios, vertical_equity_panel

            out_dir = Path(output_directory).expanduser()
            if not out_dir.is_dir():
                return _error_response(
                    f"output_directory is not an existing directory: {out_dir}")

            name = filename if filename.lower().endswith(".png") else filename + ".png"
            target = out_dir / name

            df = _read_dataset(file_path)
            written = vertical_equity_panel(
                df,
                assessment_header=assessment_col,
                market_value_header=market_value_col,
                mkt_value_proxy_header=mkt_value_proxy_col,
                prb_mvp_header=prb_mvp_col,
                mvp_sale_price_share=mvp_sale_price_share,
                title=title,
                subtitle=subtitle,
                vei_conf_level=vei_conf_level,
                vem_conf_level=vem_conf_level,
                vem_acceptable_inequity_level=vem_acceptable_inequity_level,
                graph_filepath=str(target),
            )

            # The numbers behind the dials, so the answer is readable without
            # opening the image. Same proxies as the panel above, or the
            # verdicts here silently reread the panel's own gauges and
            # boxplot off a different market-value proxy than the one
            # they are supposed to describe.
            row = ratios(df, assessment_header=assessment_col,
                         market_value_header=market_value_col,
                         mkt_value_proxy_header=mkt_value_proxy_col,
                         prb_mvp_header=prb_mvp_col,
                         mvp_sale_price_share=mvp_sale_price_share,
                         indicators=["Median", "CoD", "PRD", "PRB", "VEI", "VEM"],
                         print_ci_settings=False).iloc[0].to_dict()
            verdicts = {k: v for k, v in row.items()
                        if k in ("Median", "CoD", "PRD", "PRB", "VEI",
                                 "VEI_conclusion", "Inequity_conclusion",
                                 "Inequity_severity", "Inequity_prevalence")}

            return _to_jsonable({
                "file_written": str(written),
                "rows_used": int(len(df)),
                "verdicts": verdicts,
            })
        except Exception as ex:
            return _error_response(f"generate_vertical_equity_panel failed: {ex}", ex)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the MCP server over stdio.

    Returns
    -------
    None
        This function does not return; it starts the server and blocks,
        or exits the process if the ``mcp`` package is not installed.
    """
    if FastMCP is None:
        sys.stderr.write(_MCP_IMPORT_ERROR + "\n")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
