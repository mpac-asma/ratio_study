"""Internal helpers for accessing packaged data files."""
from importlib import resources
from pathlib import Path
from typing import Literal, overload

import pandas as pd


def sample_dataset_path() -> Path:
    """
    Return a filesystem path to the bundled sample dataset.

    Works whether the package is installed normally, in editable mode,
    or run from a wheel/zip.

    Returns
    -------
    pathlib.Path
        Filesystem path to the bundled sample dataset parquet file,
        materialized to disk if the package is loaded from a zip.
    """
    # files() returns a Traversable; for parquet we need a real path
    # because pyarrow/fastparquet read from disk
    data_files = resources.files("ratio_study").joinpath(
        "data", "ratio_study_sample_dataset.parquet"
    )
    # as_file() materializes to disk if needed (e.g., inside a zip)
    with resources.as_file(data_files) as p:
        return Path(p)


@overload
def load_sample_dataset(backend: Literal["pandas"] = "pandas") -> "pd.DataFrame": ...


@overload
def load_sample_dataset(backend: Literal["polars"]) -> "object": ...


@overload
def load_sample_dataset(backend: Literal["pyarrow"]) -> "object": ...


def load_sample_dataset(backend: str = "pandas"):
    """
    Load the bundled sample dataset as a frame in the requested backend.

    The ``latitude`` and ``longitude`` columns are randomly assigned, only so
    the spatial functions have coordinates to run on. They are not realistic
    (many homes fall on the Ottawa River) and are unrelated to where a
    property really is, so ratios show no geographic pattern by construction.

    Parameters
    ----------
    backend : str, default 'pandas'
        Which dataframe library to load the parquet file with. One of
        ``"pandas"``, ``"polars"``, or ``"pyarrow"``.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame, or pyarrow.Table
        The sample dataset, in the type corresponding to ``backend``:

        * ``"pandas"`` -- a ``pandas.DataFrame``.
        * ``"polars"`` -- a ``polars.DataFrame``.
        * ``"pyarrow"`` -- a ``pyarrow.Table``.
    """
    if backend == "pandas":
        return pd.read_parquet(sample_dataset_path())
    if backend == "polars":
        import polars as pl
        return pl.read_parquet(sample_dataset_path())
    if backend == "pyarrow":
        import pyarrow.parquet as pq
        return pq.read_table(sample_dataset_path())
    raise ValueError(
        f"Unknown backend '{backend}'. Expected 'pandas', 'polars', or 'pyarrow'."
    )