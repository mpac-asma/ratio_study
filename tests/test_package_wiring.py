"""Guards on how the test suite resolves the package under test.

The repo is developed with a non-editable ``ratio_study`` sometimes present
in site-packages. If that copy shadows ``src/``, the whole suite silently
tests a released version instead of the working tree -- passing while the
code under development is broken. ``pythonpath = ["src"]`` in
``pyproject.toml`` prevents that; these tests prove it is still in effect.
"""
from pathlib import Path

import ratio_study

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_under_test_is_the_working_tree_not_site_packages():
    resolved = Path(ratio_study.__file__).resolve()
    expected = REPO_ROOT / "src" / "ratio_study" / "__init__.py"
    assert resolved == expected, (
        f"tests are importing {resolved}, not the working tree at {expected}. "
        'Check `pythonpath = ["src"]` under [tool.pytest.ini_options].'
    )


def test_version_matches_the_working_tree_source():
    src_version = {}
    exec((REPO_ROOT / "src" / "ratio_study" / "_version.py").read_text(), src_version)
    assert ratio_study.__version__ == src_version["__version__"]
