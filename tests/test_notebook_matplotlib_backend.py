"""Regression test for the Jupyter inline-backend clobbering bug.

``gini.py``, ``blinder_oaxaca.py`` and ``reporting.py`` used to call
``matplotlib.use("Agg")`` at *module import time*. Since ``ratio_study``'s
``__init__`` imports all three, a plain ``import ratio_study`` after
``%matplotlib inline`` silently reset the backend to Agg, turning every
subsequent ``plt.show()`` in the notebook into a no-op -- figures were built
but never displayed. See ``ratio_study.helpers._use_agg_if_headless``, which
now only switches backends when nothing display-capable is already active.

This has to run inside a real IPython kernel (not just ``get_ipython() is
None`` in-process) because the bug is specifically about clobbering the
kernel's inline backend.
"""
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _execute(cell_sources):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(src) for src in cell_sources]
    nbclient.NotebookClient(nb, timeout=60, kernel_name="python3").execute()
    return nb


def test_import_after_inline_magic_keeps_plt_show_working():
    nb = _execute([
        f"import sys; sys.path.insert(0, {SRC_DIR!r})",
        "%matplotlib inline",
        "import ratio_study",
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([0, 1], [0, 1])\n"
        "plt.show()",
    ])

    outputs = nb.cells[-1].outputs
    image_outputs = [
        o for o in outputs
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})
    ]
    assert image_outputs, (
        "plt.show() produced no image/png display_data output -- "
        "`import ratio_study` reset the inline backend"
    )
