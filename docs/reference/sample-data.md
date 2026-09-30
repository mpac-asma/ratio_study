# Sample data

The bundled anonymised sales file.

*Module:* `ratio_study._data`

## Contents

- [`load_sample_dataset`](#load-sample-dataset) — Load the bundled sample dataset as a frame in the requested backend.
- [`sample_dataset_path`](#sample-dataset-path) — Return a filesystem path to the bundled sample dataset.

---

## `load_sample_dataset`

```python
load_sample_dataset(backend: str = 'pandas')
```

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

---

## `sample_dataset_path`

```python
sample_dataset_path() -> pathlib.Path
```

Return a filesystem path to the bundled sample dataset.

Works whether the package is installed normally, in editable mode,
or run from a wheel/zip.

Returns
-------
pathlib.Path
    Filesystem path to the bundled sample dataset parquet file,
    materialized to disk if the package is loaded from a zip.
