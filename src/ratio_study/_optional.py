"""Helpers for gating optional third-party dependencies.

The base install carries only what the core metrics and matplotlib charts
need. Heavier libraries live behind extras, so a missing one should produce
a message naming the extra to install rather than a bare
``ModuleNotFoundError`` from somewhere deep in the call stack.
"""
import importlib

__all__ = ['require', 'have']


def require(module, extra, feature):
    """Import ``module``, or raise an ImportError naming the extra.

    Parameters
    ----------
    module : str
        Importable module path, e.g. ``'plotly.graph_objects'``.
    extra : str
        The extra that provides it, e.g. ``'interactive'``.
    feature : str
        Human-readable name of what the caller was trying to do, used to
        open the error message.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If the module is unavailable. The message names the pip command
        that fixes it.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires '{module}', which is not installed.\n"
            f'    pip install "ratio_study[{extra}]"'
        ) from exc


def have(module):
    """Report whether ``module`` can be imported, without raising.

    Useful for degrading gracefully -- picking a matplotlib chart when the
    Plotly one is unavailable, for instance -- rather than erroring.
    """
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
