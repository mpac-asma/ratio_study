import numpy as np
import pandas as pd
import statsmodels.api as sm
import unicodedata
import re
from datetime import datetime
from pathlib import Path
from ratio_study.confint import _format_ci_string, _format_ci
from . import nw_utils


def _use_agg_if_headless():
    """Force matplotlib's Agg backend, unless something has already set up
    a display-capable one.

    Chart-producing functions need a backend that can render without a
    display (CI runners, the MCP server, plain scripts). Forcing Agg
    unconditionally at import time, though, clobbers whatever backend the
    caller already selected -- including Jupyter's ``%matplotlib inline``,
    which turns ``plt.show()`` into a silent no-op for the rest of the
    notebook. So only switch when nothing has claimed a backend yet.
    """
    import matplotlib

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return
    except ImportError:
        pass

    current_backend = matplotlib.get_backend().lower()
    headless_safe = ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template')
    if current_backend in headless_safe or 'inline' in current_backend:
        return

    matplotlib.use('Agg')


def _add_needed_variables(vars, required_cols, input_df, error_messages):
    try:
        if vars:
            if isinstance(vars, dict):
                vars_to_check = vars.keys()
            elif isinstance(vars, list) or isinstance(vars, set):
                vars_to_check = _flatten_list(vars)
            else:
                vars_to_check = list(vars)
            for h in vars_to_check: # add all necessary columns for graphs
                if isinstance(h, list):
                    for item in h:
                        if item not in required_cols:
                            if not nw_utils.has_columns(input_df, item):
                                error_messages.append('input error: required column missing in sales file:' + item)
                                # if gen_log:
                                #     logging.warning('input error: required column missing in sales file:' + item)
                            else:
                                required_cols.add(item)
                elif isinstance(h, str):
                    if not nw_utils.has_columns(input_df, h):
                        error_messages.append('input error: required column missing in sales file:' + h)
                        # if gen_log:
                        #     logging.warning('input error: required column missing in sales file:' + h)
                    elif h not in required_cols:
                        required_cols.add(h)
                else:
                    error_messages.append('error on add_needed_variables: required variables must be list or str, istead got ' + str(type(h)) + h)
                    # if gen_log:
                    #     logging.warning('error on add_needed_variables: variables must be list or str, istead got ' + str(type(h)) + h)
        return required_cols, error_messages
    except:
        error_messages.append('error in add_needed_variables')
        # if gen_log:
        #     logging.warning('error in add_needed_variables')
        return required_cols, error_messages

# ---------------------------------------------------------------------------------------------------------------------

def leaf_values(d):
    return [x for v in d.values()
              for x in (leaf_values(v) if isinstance(v, dict) else v)]

# ---------------------------------------------------------------------------------------------------------------------

def slugify(value, allow_unicode=False):
    """
    Convert a string into a URL-friendly slug.

    Sourced from
    https://github.com/django/django/blob/master/django/utils/text.py.
    Converts spaces or repeated dashes to single dashes, removes
    characters that aren't alphanumerics, underscores, or hyphens, and
    converts to lowercase. Also strips leading and trailing whitespace,
    dashes, and underscores.

    Parameters
    ----------
    value : Any
        Value to slugify. Coerced to ``str`` before processing.
    allow_unicode : bool, default False
        If True, normalize using Unicode NFKC and keep non-ASCII
        alphanumeric characters. If False, transliterate to ASCII,
        dropping characters that have no ASCII equivalent.

    Returns
    -------
    str
        The slugified string.
    """
    try:
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')
    except Exception as ex:
        raise RuntimeError('error in ratio study: slugify:', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _ols_slope(y, x, weights=None):
    """
    Slope of a single-regressor least-squares fit, in closed form.

    ``cov(x, y) / var(x)``, which is the whole of what an intercept-plus-one-
    regressor OLS reports as its slope. `_simple_regression` gets the same
    number out of statsmodels, but also builds residuals, fitted values,
    t-values, p-values and an F-test -- about an order of magnitude more work
    than the slope alone. This is for the callers that want only the slope
    and run in a tight loop, notably PRB inside the bootstrap resampler.

    NaNs in either array are dropped pairwise, matching
    `_simple_regression`.

    Parameters
    ----------
    y : array-like
        Dependent variable (numpy/pandas/polars/narwhals).
    x : array-like
        Independent variable (numpy/pandas/polars/narwhals).
    weights : array-like, optional
        Weights. Only their relative size matters, so no rescaling is
        needed.

    Returns
    -------
    float
        The slope, or ``nan`` when fewer than two points survive or ``x``
        has no spread (an undetermined slope, which is what statsmodels
        would also report).
    """
    y = nw_utils.to_numpy_1d(y).astype(np.float64, copy=False)
    x = nw_utils.to_numpy_1d(x).astype(np.float64, copy=False)

    mask = ~np.isnan(y) & ~np.isnan(x)
    if not mask.all():
        y, x = y[mask], x[mask]
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(np.float64, copy=False)[mask]
    if y.size < 2:
        return float('nan')

    if weights is None:
        x_centred = x - x.mean()
        denominator = x_centred @ x_centred
        numerator = x_centred @ (y - y.mean())
    else:
        weights = nw_utils.to_numpy_1d(weights).astype(np.float64, copy=False)
        total = weights.sum()
        if total == 0:
            return float('nan')
        x_centred = x - (weights @ x) / total
        weighted_x = weights * x_centred
        denominator = weighted_x @ x_centred
        numerator = weighted_x @ (y - (weights @ y) / total)

    if denominator == 0:
        return float('nan')
    return float(numerator / denominator)


# ---------------------------------------------------------------------------------------------------------------------

def _simple_regression(y
                    , x
                    , group=None
                    , constant=True
                    , weights=None
                    , ci_alpha=None):
    """
    Run an OLS (or WLS) regression and return summary statistics.

    Used by function ``gini_stats``.

    Parameters
    ----------
    y : array-like
        Numpy array, pandas Series, or narwhals Series. Dependent variable.
    x : array-like
        Numpy array, pandas Series, or narwhals Series. Independent variable.
    group : array-like, optional
        Numpy array, pandas Series, or narwhals Series. Grouping values
        converted to one-hot dummy columns appended to ``x``, used to
        isolate group effects.
    constant : bool, default True
        If True, a constant term is added to the regression.
    weights : array-like, optional
        Array of weights, assumed to have mean 1. If provided, a
        weighted least squares (WLS) regression is run instead of OLS.
    ci_alpha : float, optional
        Confidence interval alpha level used to compute the slope's
        confidence interval via ``model.conf_int``.

    Returns
    -------
    dict
        Dictionary of regression results with keys:

        * slope : dict -- ``beta``, ``pvalue``, ``tvalue`` for the
          independent variable, plus ``beta_ci`` (the confidence
          interval, or None on error) when ``ci_alpha`` is given.
        * rsquared_adj : float -- adjusted R-squared.
        * rsquared : float -- R-squared.
        * f_pvalue : float -- p-value of the model's F-statistic.
        * resid : numpy.ndarray -- model residuals.
        * fitted_values : numpy.ndarray -- fitted values.
        * intercept : dict, optional -- ``coefficient``, ``pvalue``,
          ``tvalue`` for the intercept term, present only when
          ``constant`` is True.
    """
    errors = []
    try:
        # Coerce inputs to 1D float64 numpy arrays (narwhals / pandas /
        # polars Series are all accepted via ``nw_utils.to_numpy_1d``).
        # Casting to float64 is critical because ``np.isnan`` is only
        # defined for numeric dtypes.
        y = nw_utils.to_numpy_1d(y).astype(np.float64, copy=False)
        x = nw_utils.to_numpy_1d(x).astype(np.float64, copy=False)
        if group is not None:
            group = nw_utils.to_numpy_1d(group)
        if weights is not None:
            weights = nw_utils.to_numpy_1d(weights).astype(np.float64, copy=False)

        combined_mask = np.logical_and(~np.isnan(y), ~np.isnan(x))
        y_clean = y[combined_mask]
        x_clean = x[combined_mask]

        if group is not None:
            group = group[combined_mask]
            # Build one-hot dummies from a string/object array without using
            # ``pd.get_dummies`` so this works equally with polars/numpy inputs.
            group_str = np.asarray(group, dtype=str)
            unique_groups, inverse = np.unique(group_str, return_inverse=True)
            n_groups = len(unique_groups)
            group_dummies = np.zeros((len(group_str), n_groups), dtype=float)
            group_dummies[np.arange(len(group_str)), inverse] = 1.0
            x_clean = np.column_stack((x_clean, group_dummies))

        if constant:
            x_ready = sm.add_constant(x_clean, has_constant='add')
        else:
            x_ready = x_clean

        if weights is not None:
            weights = weights[combined_mask]
            weights = weights / np.mean(weights)
            model = sm.WLS(y_clean, x_ready, weights=weights).fit()
        else:
            model = sm.OLS(y_clean, x_ready).fit()

        result = {
            'slope': {
                'beta': model.params[1] if constant else model.params[0],
                'pvalue': model.pvalues[1] if constant else model.pvalues[0],
                'tvalue': model.tvalues[1] if constant else model.tvalues[0]
            },
            'rsquared_adj': model.rsquared_adj,
            'rsquared': model.rsquared,
            'f_pvalue': model.f_pvalue,
            'resid': model.resid,
            'fitted_values': model.fittedvalues
        }

        if constant:
            try:
                result['intercept'] = {
                    'coefficient': model.params[0],
                    'pvalue': model.pvalues[0],
                    'tvalue': model.tvalues[0]
                }
            except IndexError:
                result['intercept'] = {
                    'coefficient': None,
                    'pvalue': None,
                    'tvalue': None
                }
                errors.append('Warning: No intercept in regression model. Check input data.')
        if ci_alpha:
            try:
                result['slope']['beta_ci'] = model.conf_int(alpha=ci_alpha)
            except Exception as e:
                errors.append(f'Error calculating confidence intervals: {e}')
                result['slope']['beta_ci'] = None

        return result
    except Exception as ex:
        raise RuntimeError('Error in simple_regression:', ex, errors)

# ---------------------------------------------------------------------------------------------------------------------

def iqr_filter(asr, iqr_mult=3, log_transform=True, groups=None, min_cod=0.06):
    """
    Flag observations outside ``iqr_mult`` IQRs from the [25%, 75%] range.

    Parameters
    ----------
    asr : array-like
        1D value to check (pandas/numpy/narwhals Series, list, etc.).
        Coerced to a 1D numpy float array before filtering.
    iqr_mult : float, default 3
        Multiplier applied to the interquartile range (IQR) to compute
        the lower and upper thresholds. The lower threshold is clipped
        at 0.
    log_transform : bool, default True
        Compute the IQR bounds on ``log(asr)`` rather than the raw
        ratio. ASR is right-skewed, so quartiles taken on the raw scale
        push the upper threshold out and make it easy to miss high
        outliers while still catching low ones; log space keeps the
        bounds symmetric. Bounds are exponentiated back before being
        applied to ``asr``. Falls back to the raw scale for any group
        that contains a non-positive value.
    groups : array-like, optional
        Group labels, same length as ``asr``. When given, the IQR bounds
        (and the ``min_cod`` gate) are computed independently within
        each group instead of once over the whole sample.
    min_cod : float, default 0.06
        Minimum coefficient of dispersion a sample (or a group, when
        ``groups`` is given) must have for the filter to run on it.
        Below this, the sample's ratios are already tight enough that
        trimming would only remove legitimate sales rather than genuine
        outliers, so nothing is flagged. Set to 0 or None to disable.

    Returns
    -------
    numpy.ndarray
        1D array of int flags the same length as ``asr``: 1 for
        observations outside the [lower, upper] threshold range (or
        that are NaN), 0 otherwise. All-zero array of the same length
        if every value in ``asr`` is NaN.
    """
    arr = nw_utils.to_numpy_1d(asr).astype(float)
    n = arr.shape[0]

    if groups is None:
        return _iqr_filter_one_group(arr, iqr_mult=iqr_mult, log_transform=log_transform, min_cod=min_cod)

    group_labels = nw_utils.to_numpy_1d(groups)
    if group_labels.shape[0] != n:
        raise ValueError("`groups` must be the same length as `asr`.")

    flags = np.zeros(n, dtype=int)
    for label in pd.unique(group_labels):
        mask = group_labels == label
        flags[mask] = _iqr_filter_one_group(arr[mask], iqr_mult=iqr_mult, log_transform=log_transform, min_cod=min_cod)
    return flags

# ---------------------------------------------------------------------------------------------------------------------

def _iqr_filter_one_group(arr, iqr_mult, log_transform, min_cod):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=int)

    if min_cod:
        median = np.median(valid)
        if median > 0 and np.mean(np.abs(valid - median) / median) < min_cod:
            return np.zeros_like(arr, dtype=int)

    use_log = log_transform and np.all(valid > 0)
    calc_valid = np.log(valid) if use_log else valid
    q1, q3 = np.percentile(calc_valid, [25, 75])
    iqr = q3 - q1
    min_threshold = q1 - iqr * iqr_mult
    max_threshold = q3 + iqr * iqr_mult
    if use_log:
        min_threshold = np.exp(min_threshold)
        max_threshold = np.exp(max_threshold)
    min_threshold = max(0, min_threshold)

    flags = np.where((arr >= min_threshold) & (arr <= max_threshold), 0, 1)
    return flags.astype(int)

# ---------------------------------------------------------------------------------------------------------------------

def _iqr_trim_filepath(result, export_directory, model_name):
    """
    Resolve where the trimmed-sales export should be written.

    Parameters
    ----------
    result : str
        ``'save'``, a directory, or a file path. A path with no suffix is
        read as a directory, so ``'audit/'`` and ``'audit'`` behave alike.
    export_directory : str or Path or None
        Directory used when `result` is ``'save'``. Defaults to the
        working directory.
    model_name : str
        Stem for the generated filename.

    Returns
    -------
    pathlib.Path
        A ``.csv.gz`` path whose parent directory exists.
    """
    stamp = datetime.now().strftime('%d%b%Y_%Hh%M').lower()
    slug = re.sub(r'[^0-9A-Za-z]+', '', str(model_name)) or 'model'
    generated = f'iqr_trimmed_sales_{slug}_{stamp}.csv.gz'

    if str(result).strip().lower() == 'save':
        directory = Path(export_directory) if export_directory else Path.cwd()
        path = directory / generated
    else:
        candidate = Path(str(result)).expanduser()
        if candidate.is_dir() or not candidate.suffix:
            path = candidate / generated
        else:
            # Always gzip, whatever the caller asked for: these files are
            # whole sales records and compress by roughly an order of magnitude.
            path = candidate if candidate.name.endswith('.csv.gz') else \
                candidate.with_suffix(candidate.suffix + '.gz') \
                if candidate.suffix == '.csv' else \
                candidate.with_suffix('.csv.gz')

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _abbreviate_value(value):
    """
    Abbreviate a dollar total to a magnitude suffix, e.g. ``'20.7M'``.

    Assessment and market value totals run to ten digits, which crowds the
    trim summary and makes the two figures hard to compare at a glance.

    Parameters
    ----------
    value : float
        The amount to abbreviate.

    Returns
    -------
    str
        The amount with a ``K``/``M``/``B`` suffix, or comma-grouped when
        below a thousand.
    """
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f'{value / 1e9:.1f}B'
    if magnitude >= 1e6:
        return f'{value / 1e6:.1f}M'
    if magnitude >= 1e3:
        return f'{value / 1e3:.0f}K'
    return f'{value:,.0f}'


def report_iqr_trim(trimmed, assessment_header, market_value_header, total_assessed_value, total_market_value,
                    result='print', export_directory=None, model_name='model',
                    total_count=None, printer=print):
    """
    Report -- and optionally export -- the sales an IQR trim removed.

    Appendix C.6 of the Standard on Ratio Studies is explicit that statistics
    from a trimmed distribution are not comparable with untrimmed ones, and
    that a report must state the basis for excluding outliers and identify
    what was removed. Trimming silently is the failure mode this guards
    against: the summary always says how much of the sample went, and
    ``result='save'`` writes the records themselves so the exclusions can be
    audited.

    Parameters
    ----------
    trimmed : pandas.DataFrame
        The removed sales.
    assessment_header, market_value_header : str
        Columns totalled in the summary.
    result : str or None, default 'print'
        * ``'print'`` -- summary only.
        * ``'save'`` -- write a ``.csv.gz`` to `export_directory` under a
          generated name, then print the path and the summary.
        * a directory -- as ``'save'``, but there.
        * a file path -- write exactly there, forced to ``.csv.gz``.
        * ``None`` / ``''`` / ``False`` -- silent, and nothing written.
    export_directory : str or Path, optional
        Directory used by ``'save'``. Defaults to the working directory.
    model_name : str, default 'model'
        Stem for the generated filename.
    total_count : int, optional
        Sample size before trimming, so the summary can report the share
        removed -- the figure Appendix C.5 caps at 5-10%.
    printer : callable, default print
        Where the summary goes. Injected for testing.

    Returns
    -------
    pathlib.Path or None
        The file written, or None when nothing was.
    """
    if not result:
        return None

    path = None
    if str(result).strip().lower() != 'print':
        path = _iqr_trim_filepath(result, export_directory, model_name)
        trimmed.to_csv(path, index=False, compression='gzip')
        printer(f'IQR filter: trimmed sales saved to {path}')

    else:
        printer("If you'd like to save the trimmed sales for audit, pass argument "
               "`iqr_filter_result='save'` or `iqr_filter_result='/path/to/file.csv.gz'`")

    count = len(trimmed)
    share = f' ({count / total_count:.2%})' if total_count else ''
    lines = [
        'IQR filter - trimmed sales',
        f'  Sales trimmed          : {count:,}'
        + (f' of {total_count:,}{share}' if total_count else ''),
    ]
    if count:
        trimmed_assessed_value = trimmed[assessment_header].sum()
        trimmed_market_value = trimmed[market_value_header].sum()
        lines += [
            f'  Total trimmed assessment value : {_abbreviate_value(trimmed_assessed_value)}'+ (f' of {_abbreviate_value(total_assessed_value)} ({trimmed_assessed_value/total_assessed_value:.2%})'),
            f'  Total trimmed market value     : {_abbreviate_value(trimmed_market_value)}'+ (f' of {_abbreviate_value(total_market_value)} ({trimmed_market_value/total_market_value:.2%})'),
        ]
    else:
        # A zero can mean "nothing was extreme" or "min_cod blocked the trim".
        # Saying nothing at all would leave the reader unable to tell.
        lines.append('  Nothing was outside the threshold, or min_cod blocked the trim.')
    printer('\n'.join(lines))

    return path


# ---------------------------------------------------------------------------------------------------------------------

def _split_list(input_list, items_per_sublist):
    sublists = []
    num_sublists = -(-len(input_list) // items_per_sublist)  # Ceiling division to calculate the number of sublists needed
    items_per_sublist_adjusted = -(-len(input_list) // num_sublists)  # Calculate the adjusted items per sublist

    for i in range(0, len(input_list), items_per_sublist_adjusted):
        sublists.append(input_list[i:i + items_per_sublist_adjusted])

    return sublists

# ---------------------------------------------------------------------------------------------------------------------

def save_errors_to_text(error_messages, text_file_path):
    try:
        with open(text_file_path, 'w') as file:
            for string in error_messages:
                file.write(f"{string}\n")
        return file.name  # Return the path of the created text file
    except Exception as e:
        return None

# ---------------------------------------------------------------------------------------------------------------------

def _add_suffixes_to_list(header_list, suffixes):
    """
    Add column headers with suffixes while keeping the order.

    Parameters
    ----------
    header_list : list
        List of column header names, in the order to preserve.
    suffixes : dict
        Dictionary mapping each suffix (str) to the list of header
        names it should be appended to. For every header in
        ``header_list`` that appears in ``suffixes[suffix]``, a new
        entry ``header + suffix`` is inserted immediately after it.

    Returns
    -------
    list
        The original headers with suffixed variants inserted in place.
    """
    try:
        result = []
        for item in header_list:
            result.append(item)
            for suffix in suffixes.keys():
                if item in suffixes[suffix]:
                    result.append(item + suffix)
        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: add_suffixes_to_list', ex)

# ---------------------------------------------------------------------------------------------------------------------

def _flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, set):
            item = list(item)
        if isinstance(item, list):
            result.extend(_flatten_list(item))
        else:
            result.append(item)
    return result

# ---------------------------------------------------------------------------------------------------------------------

def _check_inputs(x, rankvar):
    try:
        # Accept any 1D value by coercing to numpy first; this covers
        # pandas Series, polars Series, narwhals Series, lists, and ndarrays.
        x_arr = nw_utils.to_numpy_1d(x)
        rv_arr = nw_utils.to_numpy_1d(rankvar)
        if x_arr.shape[0] != rv_arr.shape[0]:
            raise ValueError("x and rankvar must have the same length")
    except Exception as ex:
        raise RuntimeError('error in ratio study: check_inputs', ex)


# ---------------------------------------------------------------------------------------------------------------------

def _apply_format(x, format_str):
    # Accept narwhals scalar, numpy scalar, python scalar.
    if x is None:
        return ''
    # Treat NaN as missing
    try:
        if isinstance(x, float) and np.isnan(x):
            return ''
    except (TypeError, ValueError):
        pass
    if isinstance(x, (int, float, np.integer, np.floating)):
        return format_str.format(x)
    elif isinstance(x, str):
        return str(x)
    else:
        return ''

# ---------------------------------------------------------------------------------------------------------------------

def custom_format(df, cols, formats, skip=[], all_str=False, ci_string=True):
    """
    Format columns of a dataframe using per-column format strings.

    Accepts any supported frame (pandas, polars, pyarrow, narwhals). For
    backend-agnostic operation, the frame is converted to pandas
    internally for the per-cell string formatting pass. The original
    frame's backend is restored on return when possible.

    Parameters
    ----------
    df : DataFrame
        Pandas, polars, pyarrow, or narwhals dataframe to format.
    cols : list of str or None
        Columns to format. If None, all columns in ``df`` are
        considered.
    formats : dict
        Mapping of column name (or column-name prefix) to a
        ``str.format``-style format string. Must include an
        ``'All others'`` key used as the fallback format for columns
        that don't match any other key. If a column isn't an exact
        key, the first key that is a prefix of the column name is
        used instead (e.g. ``'CoD'`` matches ``'CoD_ci'``).
    skip : list, default []
        Columns to leave unformatted (passed through unchanged).
    all_str : bool, default False
        If True, formatted columns are cast to ``str`` (with missing
        values filled as ``''``) after formatting.
    ci_string : bool, default True
        Controls how columns containing ``'_ci'`` in their name are
        formatted:

        * True -- format as a confidence-interval string via
          ``_format_ci_string``, inferring rounding from the format
          string (defaulting to 4 decimal places, +2 if the format
          contains ``'%'``).
        * False -- format via ``_format_ci`` using the ``formats``
          pattern directly.

    Returns
    -------
    DataFrame
        A new dataframe with the requested columns formatted (as
        strings) and all other columns unchanged, in the same column
        order and backend as the input ``df``.
    """
    try:
        backend = nw_utils.native_backend(df)
        df_native = nw_utils.require_pandas(df, reason="custom_format() per-cell formatting")
        df_result = pd.DataFrame()
        cols_to_iter = list(df_native.columns) if cols is None else list(cols)
        for col in cols_to_iter:
            if (col in df_native.columns) and (col not in skip):
                # determine which format to use
                format_str = formats['All others'] # basic format if no specific one is to be found
                if col in formats: # if the column is in the formats dictionary
                    format_str = formats[col]
                else:
                    for key in formats.keys():
                        if key == col[:len(key)]: # if the column starts with an item from the formats dictionary, like 'CoD' in 'CoD_ci.
                            format_str = formats[key]
                            break

                if '_ci' in col:
                    if ci_string:
                        try:
                            rounding = int(format_str[-3:-2])
                            if '%' in format_str: rounding += 2
                        except:
                            rounding = 4
                        df_result[col] = df_native[col].apply(lambda x: _format_ci_string(x, pattern=format_str, rounding=rounding))
                    else:
                        df_result[col] = df_native[col].apply(_format_ci, pattern=format_str)
                else:
                    df_result[col] = df_native[col].apply(lambda x: _apply_format(x, format_str))

                if all_str:
                    df_result[col] = df_result[col].fillna('').astype(str)

        other_cols = [x for x in df_native.columns if x not in df_result.columns]
        df_result[other_cols] = df_native[other_cols]
        # Restore the order of the input frame's columns.
        result = df_result[df_native.columns]
        # Return in the original backend if it was a supported native type.
        if backend == "polars":
            import polars as pl
            return pl.from_pandas(result)
        if backend == "pyarrow":
            import pyarrow as pa
            return pa.Table.from_pandas(result)
        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: custom_format', ex)
    
# ---------------------------------------------------------------------------------------------------------------------

# Compute roughly equal group sizes for the VEI percentile partition.
def _get_group_sizes(n, m, percentile_method='R7'):
    """
    Compute group sizes for an ``m``-way partition of ``n`` observations
    in ascending order, using a Hyndman & Fan (1996) percentile rank
    method as specified in the IAAO Exposure Draft (May 2026) Section E.3.

    Parameters
    ----------
    n : int
        Total number of observations.
    m : int
        Number of percentile groups (e.g. 2 halves, 4 quartiles, 10 deciles).
    percentile_method : {'R6', 'R7'}, default 'R7'
        Hyndman-Fan percentile rank method. Both produce roughly equal
        groups with slightly different handling of unequal-divisibility
        remainder observations. R7 is the default and matches the
        draft's worked example in Table 23 (N=54 quartiles).

    Returns
    -------
    list[int]
        Length-``m`` list of integer group sizes summing to ``n``.

    Notes
    -----
    The draft Section E.3 defines (1-indexed):

        R6:  h_i = floor(P_i / 100 * (N + 1))
        R7:  h_i = floor(P_i / 100 * (N - 1) + 1)

    and ``OC_i = h_i - h_{i-1}`` for ``h_0 = 0``. We compute the
    boundary positions for ``P_i = 100 * i / m`` for ``i = 1..m`` and
    differencing them.
    """
    if n < m:
        # Defensive; the VEI n>=20 minimum guarantees this can't happen.
        return [1] * n  # best-effort fallback
    method_up = str(percentile_method).upper()
    if method_up not in ('R6', 'R7'):
        raise ValueError(f"Unknown percentile_method: {percentile_method!r}; expected 'R6' or 'R7'.")
    # Per the draft Section E.3, an observation with 1-indexed position
    # ``k`` belongs to PG i iff ``h_{i-1} < k <= h_i``, where ``h_i`` is
    # computed from the boundary percentile P_i = 100*i/m:
    #   R6:  h_i = floor(P_i/100 * (N + 1))
    #   R7:  h_i = floor(P_i/100 * (N - 1) + 1)
    # Vectorize: pick the smallest i in [1..m-1] such that h_i >= k;
    # values of k larger than every h_i fall into the last group.
    if method_up == 'R6':
        h = np.floor(((np.arange(1, m) * (100.0 / m) / 100.0) * (n + 1))).astype(int)
    else:  # R7
        h = np.floor(((np.arange(1, m) * (100.0 / m) / 100.0) * (n - 1) + 1)).astype(int)
    h = np.clip(h, 1, n)
    # Ensure strictly increasing to avoid empty groups from rounding ties.
    for i in range(1, len(h)):
        if h[i] <= h[i - 1]:
            h[i] = h[i - 1] + 1
    # For each observation position k=1..n, find which boundary it first exceeds.
    k_arr = np.arange(1, n + 1)
    # ``idx[i] = count of k such that h_j < k <= h_{i}`` after the cumulative
    # step; we use a direct assignment: positions in (h[i-1], h[i]] go to group i.
    sizes = [0] * m
    prev = 0
    for i in range(m - 1):
        hi = int(h[i])
        cnt = max(0, min(n, hi) - max(prev, 0))
        sizes[i] = cnt
        prev = hi
    sizes[m - 1] = n - prev
    # Final safety: if rounding produced an empty middle group, redistribute.
    for i in range(m):
        if sizes[i] == 0 and n >= m:
            sizes[i] = 1
            # borrow one from a larger neighbour
            j = max(range(m), key=lambda x: sizes[x] if x != i else 0)
            if sizes[j] > 1:
                sizes[j] -= 1
    return sizes

# ---------------------------------------------------------------------------------------------------------------------

# Determine number of percentile groups
def _determine_num_groups(n):
    """
    Determine the number of percentile groups for a given sample size.

    Parameters
    ----------
    n : int
        The sample size. Must be at least 20.

    Returns
    -------
    int
        The number of groups to use based on sample size:

        * 2 groups (halves) if ``n <= 50``.
        * 4 groups (quartiles) if ``51 <= n <= 500``.
        * 10 groups (deciles) if ``n > 500``.

    Raises
    ------
    ValueError
        If ``n`` is less than 20.

    Examples
    --------
    >>> _determine_num_groups(25)
    2
    >>> _determine_num_groups(100)
    4
    >>> _determine_num_groups(1000)
    10
    """
    if n < 20:
        raise ValueError("Sample size too small")
    elif n <= 50:
        return 2  # halves
    elif n <= 500:
        return 4  # quartiles
    else:
        return 10  # deciles