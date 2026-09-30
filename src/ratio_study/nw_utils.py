"""Narwhals compatibility layer for the ratio_study package.

This module provides utility functions to make the ratio_study package
backend-agnostic across pandas, polars, and (where supported) PyArrow
tables. The package uses a "narwhals-wrapped boundary" approach:

* Public-facing functions accept any supported frame (pandas, polars,
  pyarrow, or already-wrapped narwhals).
* Inside the package, data is unwrapped to pandas or NumPy for heavy
  operations involving libraries that are not narwhals-aware (e.g.,
  ``seaborn``, ``xlsxwriter``, ``statsmodels.formula.api``,
  ``statsmodels.stats.weightstats.DescrStatsW``).
* Where feasible, the heavy operations are rewritten to use narwhals
  expressions so the work runs in the user's chosen backend.
* Results are returned in the same backend as the input where possible.

Why not use narwhals everywhere?
--------------------------------
Some third-party libraries require real pandas DataFrames:

* ``seaborn`` uses string column names and ``data=df, x='col'`` kwargs.
* ``xlsxwriter`` is invoked via ``pd.ExcelWriter``.
* ``statsmodels.formula.api.ols`` requires pandas for R-style formulas.
* ``statsmodels.stats.weightstats.DescrStatsW`` requires a pandas
  Series/array.

For these, we convert at the boundary, run the legacy code, and
optionally return the result in the original backend.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Union

import narwhals as nw
import numpy as np

# Re-export narwhals so callers can do ``from ratio_study.nw_utils import nw``
# without importing narwhals directly. This also keeps a single version of
# narwhals in use across the package.
nw = nw


# ---------------------------------------------------------------------------
# Frame detection
# ---------------------------------------------------------------------------

# Native (non-narwhals) frame types we accept.
_PANDAS_DF = None
_PANDAS_SERIES = None
_POLARS_DF = None
_POLARS_SERIES = None
_PYARROW_TBL = None
_PYARROW_CHUNKED = None

try:
    import pandas as _pd
    _PANDAS_DF = _pd.DataFrame
    _PANDAS_SERIES = _pd.Series
except Exception:  # pragma: no cover - pandas is required dep
    pass

try:
    import polars as _pl
    _POLARS_DF = _pl.DataFrame
    _POLARS_SERIES = _pl.Series
except Exception:  # pragma: no cover - polars is optional
    pass

try:
    import pyarrow as _pa
    _PYARROW_TBL = _pa.Table
    _PYARROW_CHUNKED = _pa.ChunkedArray
except Exception:  # pragma: no cover - pyarrow is required for parquet
    pass


def is_native_frame(obj: Any) -> bool:
    """
    Check whether an object is a native pandas, polars, or pyarrow frame or series.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if ``obj`` is a pandas/polars DataFrame or Series, or a
        pyarrow Table; False otherwise (including when ``obj`` is
        ``None``).
    """
    if obj is None:
        return False
    if _PANDAS_DF is not None and isinstance(obj, (_PANDAS_DF, _PANDAS_SERIES)):
        return True
    if _POLARS_DF is not None and isinstance(obj, (_POLARS_DF, _POLARS_SERIES)):
        return True
    if _PYARROW_TBL is not None and isinstance(obj, _PYARROW_TBL):
        return True
    return False


def is_narwhals_frame(obj: Any) -> bool:
    """
    Check whether an object is a narwhals-wrapped frame or series.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if ``obj`` is an ``nw.DataFrame``, ``nw.LazyFrame``, or
        ``nw.Series``; False otherwise.
    """
    return isinstance(obj, nw.DataFrame) or isinstance(obj, nw.LazyFrame) \
        or isinstance(obj, nw.Series)


def is_supported_frame(obj: Any) -> bool:
    """
    Check whether an object is any frame-like type the package can accept.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if ``obj`` is a native frame (see :func:`is_native_frame`)
        or a narwhals-wrapped frame (see :func:`is_narwhals_frame`);
        False otherwise.
    """
    return is_native_frame(obj) or is_narwhals_frame(obj)


# ---------------------------------------------------------------------------
# Wrapping / unwrapping
# ---------------------------------------------------------------------------

def to_narwhals(obj: Any, *, allow_series: bool = True) -> nw.DataFrame:
    """
    Coerce a pandas/polars/pyarrow/narwhals frame to ``nw.DataFrame``.

    Series are wrapped as a one-column DataFrame; set
    ``allow_series=False`` to raise instead.

    Parameters
    ----------
    obj : Any
        Frame-like object to coerce: a pandas/polars DataFrame, a
        pyarrow Table, or an already-wrapped narwhals
        DataFrame/LazyFrame/Series.
    allow_series : bool, default True
        If True, a narwhals ``Series`` input is wrapped as a
        one-column ``nw.DataFrame`` named after the series (or
        ``"value"`` if unnamed). If False, a ``Series`` input raises
        ``TypeError``.

    Returns
    -------
    nw.DataFrame
        The input coerced to a narwhals DataFrame.
    """
    if obj is None:
        raise ValueError("Cannot wrap None as a narwhals DataFrame")

    if isinstance(obj, nw.DataFrame):
        return obj
    if isinstance(obj, nw.LazyFrame):
        return obj.collect()
    if isinstance(obj, nw.Series):
        if not allow_series:
            raise TypeError("Series input not allowed here; expected a frame")
        return nw.DataFrame({obj.name or "value": obj})

    if is_native_frame(obj):
        return nw.from_native(obj, eager_only=True)

    raise TypeError(
        f"Unsupported frame type: {type(obj).__name__}. "
        "Expected pandas.DataFrame, polars.DataFrame, pyarrow.Table, "
        "or a narwhals DataFrame/Series."
    )


def to_native(
    obj: Any,
    *,
    backend: Optional[str] = None,
) -> Any:
    """
    Convert a narwhals frame back to a native frame.

    Parameters
    ----------
    obj : Any
        Either a narwhals DataFrame/LazyFrame/Series, or a native
        frame (which is returned as-is).
    backend : str, optional
        Target backend, one of ``"pandas"``, ``"polars"``,
        ``"pyarrow"``. If ``None`` (default), returns the input
        unchanged if it is already native, otherwise falls back to
        the backend of the underlying native frame (for narwhals
        inputs that were already wrapped from a known backend).

    Returns
    -------
    Any
        The native frame. Returns ``None`` if ``obj`` is ``None``.
    """
    if obj is None:
        return obj
    if is_narwhals_frame(obj):
        if backend is None:
            return obj.to_native()
        return obj.to_native().__class__.__module__  # placeholder, see below
    if is_native_frame(obj):
        if backend is None:
            return obj
        return _convert_native(obj, backend)
    raise TypeError(f"Cannot convert {type(obj).__name__} to a native frame")


def _convert_native(obj: Any, backend: str) -> Any:
    """
    Convert a native frame to the requested backend (best-effort).

    Parameters
    ----------
    obj : Any
        Native pandas/polars/pyarrow frame to convert.
    backend : str
        Target backend, one of ``"pandas"``, ``"polars"``,
        ``"pyarrow"``.

    Returns
    -------
    Any
        ``obj`` converted to the requested backend. If ``obj`` is
        already in a form not covered by a specific conversion path,
        it is returned unchanged.
    """
    if backend == "pandas":
        if _PANDAS_DF is None:
            raise ImportError("pandas is not installed")
        if _POLARS_DF is not None and isinstance(obj, _POLARS_DF):
            return obj.to_pandas()
        if _PYARROW_TBL is not None and isinstance(obj, _PYARROW_TBL):
            return obj.to_pandas()
        return obj
    if backend == "polars":
        if _POLARS_DF is None:
            raise ImportError("polars is not installed")
        if _PANDAS_DF is not None and isinstance(obj, _PANDAS_DF):
            return nw.from_native(obj, eager_only=True).to_native() if False else _pd_to_pl(obj)
        if _PYARROW_TBL is not None and isinstance(obj, _PYARROW_TBL):
            return nw.from_native(obj, eager_only=True).to_native()
        return obj
    if backend == "pyarrow":
        if _PYARROW_TBL is None:
            raise ImportError("pyarrow is not installed")
        if _PANDAS_DF is not None and isinstance(obj, _PANDAS_DF):
            import pyarrow as pa
            return pa.Table.from_pandas(obj)
        if _POLARS_DF is not None and isinstance(obj, _POLARS_DF):
            # polars supports arrow conversion directly
            return obj.to_arrow()
        return obj
    raise ValueError(f"Unknown backend '{backend}'")


def _pd_to_pl(df):
    """
    Convert a pandas DataFrame to a polars DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to convert.

    Returns
    -------
    polars.DataFrame
        ``df`` converted to a polars DataFrame.
    """
    if _POLARS_DF is None:
        raise ImportError("polars is not installed")
    import polars as pl
    return pl.from_pandas(df)


def native_backend(obj: Any) -> Optional[str]:
    """
    Best-effort detection of the original backend of a frame.

    Parameters
    ----------
    obj : Any
        Frame-like object to inspect: a native pandas/polars/pyarrow
        frame or series, or a narwhals DataFrame/LazyFrame/Series.

    Returns
    -------
    str or None
        One of ``"pandas"``, ``"polars"``, ``"pyarrow"``, or ``None``
        if the backend cannot be determined (including when ``obj``
        is ``None``). For narwhals inputs, returns the backend of the
        underlying native frame.
    """
    if obj is None:
        return None
    if _PANDAS_DF is not None and isinstance(obj, (_PANDAS_DF, _PANDAS_SERIES)):
        return "pandas"
    if _POLARS_DF is not None and isinstance(obj, (_POLARS_DF, _POLARS_SERIES)):
        return "polars"
    if _PYARROW_TBL is not None and isinstance(obj, _PYARROW_TBL):
        return "pyarrow"
    if isinstance(obj, (nw.DataFrame, nw.LazyFrame, nw.Series)):
        try:
            native = obj.to_native() if hasattr(obj, "to_native") else obj.native
            return native_backend(native)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Series / array conversion
# ---------------------------------------------------------------------------

def to_numpy_1d(
    value: Any,
    *,
    name_hint: Optional[str] = None,
) -> np.ndarray:
    """
    Convert any 1D value (narwhals Series, pandas Series, list, ...) to a 1D numpy array.

    Narwhals Series/DataFrames and native pandas/polars Series expose
    ``.to_numpy()``. Single-column DataFrames are reduced to their one
    column first. Lists and tuples are wrapped with ``np.asarray``,
    and scalars are wrapped as a length-1 array. All outputs are
    reshaped to a flat 1D array.

    Parameters
    ----------
    value : Any
        Value to convert: a numpy array, a narwhals Series or
        single-column DataFrame, a native pandas/polars Series or
        single-column DataFrame, a list, tuple, int, or float.
    name_hint : str, optional
        Currently unused by the function body; reserved for future
        use (e.g. naming the resulting array).

    Returns
    -------
    numpy.ndarray
        A 1D numpy array built from ``value``.
    """
    if value is None:
        raise ValueError("Cannot convert None to numpy array")
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (nw.Series, nw.DataFrame)):
        # DataFrame shouldn't really be passed but be defensive.
        if isinstance(value, nw.DataFrame):
            if value.shape[1] != 1:
                raise ValueError(
                    f"Expected a single column, got DataFrame with "
                    f"{value.shape[1]} columns"
                )
            value = value.get_column(value.columns[0])
        arr = value.to_numpy()
        return np.asarray(arr).reshape(-1)
    if is_native_frame(value):
        # Series (pandas/polars) or single-column DataFrame.
        if _PANDAS_SERIES is not None and isinstance(value, _PANDAS_SERIES):
            return np.asarray(value.to_numpy()).reshape(-1)
        if _POLARS_SERIES is not None and isinstance(value, _POLARS_SERIES):
            return np.asarray(value.to_numpy()).reshape(-1)
        # Frame-like: try narwhals
        wrapped = nw.from_native(value, eager_only=True)
        if wrapped.shape[1] == 1:
            return np.asarray(wrapped.get_column(wrapped.columns[0]).to_numpy()).reshape(-1)
        raise ValueError("Expected a single column or a Series")
    if isinstance(value, (list, tuple)):
        return np.asarray(value).reshape(-1)
    if isinstance(value, (int, float)):
        return np.asarray([value])
    raise TypeError(f"Cannot convert {type(value).__name__} to a 1D numpy array")


# ---------------------------------------------------------------------------
# Frame preparation utilities used at the public boundary
# ---------------------------------------------------------------------------

def coerce_frame(obj: Any) -> "nw.DataFrame":
    """
    Coerce a frame-like object to ``nw.DataFrame`` with a clear error.

    Parameters
    ----------
    obj : Any
        Frame-like object to coerce; see :func:`to_narwhals`. Series
        inputs are not allowed and raise ``TypeError``.

    Returns
    -------
    nw.DataFrame
        The input coerced to a narwhals DataFrame.
    """
    return to_narwhals(obj, allow_series=False)


def require_pandas(obj: Any, *, reason: str) -> Any:
    """
    Convert ``obj`` to a pandas DataFrame, raising if not possible.

    Use at the boundary of code that needs pandas (seaborn, xlsxwriter,
    statsmodels formula API, etc.).

    Parameters
    ----------
    obj : Any
        Frame-like object to convert: a pandas DataFrame (returned
        as-is), a narwhals DataFrame/LazyFrame, or another native
        frame (polars/pyarrow).
    reason : str
        Short description of why pandas is required, used in the
        ``ImportError``/``TypeError`` messages raised on failure.

    Returns
    -------
    pandas.DataFrame
        ``obj`` converted to a pandas DataFrame.
    """
    if _PANDAS_DF is None:
        raise ImportError("pandas is required for " + reason)
    if isinstance(obj, _PANDAS_DF):
        return obj
    if isinstance(obj, nw.DataFrame) or isinstance(obj, nw.LazyFrame):
        return obj.to_native() if isinstance(obj, nw.DataFrame) else obj.collect().to_native()
    if is_native_frame(obj):
        return _convert_native(obj, "pandas")
    raise TypeError(f"Cannot coerce {type(obj).__name__} to pandas for {reason}")


def is_pandas_frame(obj: Any) -> bool:
    return _PANDAS_DF is not None and isinstance(obj, _PANDAS_DF)


def is_polars_frame(obj: Any) -> bool:
    return _POLARS_DF is not None and isinstance(obj, _POLARS_DF)


# ---------------------------------------------------------------------------
# Public-API decorator helper
# ---------------------------------------------------------------------------

class BackendTracker:
    """Track the backend of an input frame so outputs can match it.

    Usage::

        tracker = BackendTracker(input_df)
        ... # do internal work, possibly converting to pandas
        result_df = tracker.match(result_df)
    """

    def __init__(self, input_frame: Any) -> None:
        self._backend: Optional[str] = native_backend(input_frame)
        self._original = input_frame

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def is_pandas(self) -> bool:
        return self._backend == "pandas"

    @property
    def is_polars(self) -> bool:
        return self._backend == "polars"

    def match(self, result: Any) -> Any:
        """
        Convert ``result`` to the same backend as the tracked input.

        If the original input is pandas (or no backend was detected)
        the result is returned untouched to preserve legacy
        behaviour.

        Parameters
        ----------
        result : Any
            Native frame to convert to the tracked backend.

        Returns
        -------
        Any
            ``result`` converted to the tracked backend (via
            :func:`_convert_native`), or returned unchanged if the
            tracked backend is pandas or unknown.
        """
        if self._backend is None or self._backend == "pandas":
            return result
        return _convert_native(result, self._backend)


# ---------------------------------------------------------------------------
# Column-membership helpers (work across backends)
# ---------------------------------------------------------------------------

def frame_columns(obj: Any) -> List[str]:
    """
    Return the list of column names for any supported frame.

    Parameters
    ----------
    obj : Any
        Frame to read columns from: a narwhals DataFrame/LazyFrame, or
        a native pandas/polars DataFrame or pyarrow Table.

    Returns
    -------
    list of str
        Column names of ``obj``, as strings.
    """
    if isinstance(obj, nw.DataFrame) or isinstance(obj, nw.LazyFrame):
        return list(obj.columns)
    if is_native_frame(obj):
        if _PANDAS_DF is not None and isinstance(obj, _PANDAS_DF):
            return [str(c) for c in obj.columns]
        if _POLARS_DF is not None and isinstance(obj, _POLARS_DF):
            return [str(c) for c in obj.columns]
        if _PYARROW_TBL is not None and isinstance(obj, _PYARROW_TBL):
            return [str(c) for c in obj.column_names]
    raise TypeError(f"Cannot read columns from {type(obj).__name__}")


def has_columns(obj: Any, cols: Iterable[str]) -> bool:
    """
    Return True if all of ``cols`` exist on ``obj``.

    Parameters
    ----------
    obj : Any
        Frame to check columns against; see :func:`frame_columns`.
    cols : Iterable[str] or str or None
        Column name(s) to check for. A single string is treated as
        one column name. ``None`` or an empty iterable is treated as
        trivially satisfied.

    Returns
    -------
    bool
        True if every column in ``cols`` exists on ``obj`` (or
        ``cols`` is empty/None); False otherwise, including when
        ``obj``'s columns cannot be determined.
    """
    if cols is None:
        return True
    cols_list = list(cols) if not isinstance(cols, str) else [cols]
    if not cols_list:
        return True
    try:
        existing = set(frame_columns(obj))
    except TypeError:
        return False
    return all(c in existing for c in cols_list)


def get_series(obj: Any, column: str) -> Any:
    """
    Get a single column as a 1D numpy array.

    Accepts any supported frame type or a Series directly.

    Parameters
    ----------
    obj : Any
        Frame or Series to read from: a narwhals DataFrame/LazyFrame,
        a native pandas/polars Series or DataFrame, a pyarrow Table,
        or a list/tuple.
    column : str
        Name of the column to extract. Ignored when ``obj`` is
        already a list/tuple.

    Returns
    -------
    numpy.ndarray
        A 1D numpy array containing the values of ``column``.
    """
    if isinstance(obj, nw.DataFrame) or isinstance(obj, nw.LazyFrame):
        if isinstance(obj, nw.LazyFrame):
            obj = obj.collect()
        return to_numpy_1d(obj.get_column(column))
    if is_native_frame(obj):
        if _PANDAS_SERIES is not None and isinstance(obj, _PANDAS_SERIES):
            return np.asarray(obj.to_numpy()).reshape(-1)
        if _POLARS_SERIES is not None and isinstance(obj, _POLARS_SERIES):
            return np.asarray(obj.to_numpy()).reshape(-1)
        if _PANDAS_DF is not None and isinstance(obj, _PANDAS_DF):
            return np.asarray(obj[column].to_numpy()).reshape(-1)
        if _POLARS_DF is not None and isinstance(obj, _POLARS_DF):
            return np.asarray(obj.get_column(column).to_numpy()).reshape(-1)
        if _PYARROW_TBL is not None and isinstance(obj, _PYARROW_TBL):
            return np.asarray(obj.column(column).to_pylist()).reshape(-1)
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj).reshape(-1)
    raise TypeError(f"Cannot get column {column!r} from {type(obj).__name__}")


def select_columns(obj: Any, columns: Sequence[str]) -> Any:
    """
    Select a subset of columns from a frame, preserving its backend.

    Parameters
    ----------
    obj : Any
        Frame to select from: a narwhals DataFrame/LazyFrame, or a
        native pandas/polars DataFrame or pyarrow Table. ``None`` is
        returned unchanged.
    columns : Sequence[str]
        Column names to select, in order.

    Returns
    -------
    Any
        A frame of the same kind as ``obj`` containing only
        ``columns``. Returns ``None`` if ``obj`` is ``None``.
    """
    if obj is None:
        return obj
    if isinstance(obj, nw.DataFrame):
        return obj.select(list(columns))
    if isinstance(obj, nw.LazyFrame):
        return obj.select(list(columns))
    if is_native_frame(obj):
        if _PANDAS_DF is not None and isinstance(obj, _PANDAS_DF):
            return obj[list(columns)]
        if _POLARS_DF is not None and isinstance(obj, _POLARS_DF):
            return obj.select(list(columns))
        if _PYARROW_TBL is not None and isinstance(obj, _PYARROW_TBL):
            return obj.select(list(columns))
    raise TypeError(f"Cannot select columns from {type(obj).__name__}")


__all__ = [
    "nw",
    "BackendTracker",
    "is_native_frame",
    "is_narwhals_frame",
    "is_supported_frame",
    "is_pandas_frame",
    "is_polars_frame",
    "to_narwhals",
    "to_native",
    "native_backend",
    "to_numpy_1d",
    "coerce_frame",
    "require_pandas",
    "frame_columns",
    "has_columns",
    "get_series",
    "select_columns",
]
