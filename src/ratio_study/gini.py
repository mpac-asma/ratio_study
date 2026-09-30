import numpy as np
import pandas as pd
from scipy.stats import shapiro, skew, kurtosis
import matplotlib.pyplot as plt
import os
from matplotlib.patches import ConnectionPatch
import statsmodels.formula.api as smf
import math
from ratio_study.helpers import custom_format, _check_inputs, _simple_regression, _use_agg_if_headless
from . import nw_utils

MIN_SAMPLE_GINI = 3

# -------------------------------
# Gini indicators
# -------------------------------

# ---------------------------------------------------------------------------------------------------------------------

def gini_decomp(sale_prices, assessed_values, group_by, grouping_label, population_ratios, export_directory, dict_results=None, unbiased=False, decomp_chart='', weights=None):
    """
    Calculate the subgroup effects of Gini-based inequality indices via decomposition.

    Parameters
    ----------
    sale_prices : array-like
        Numeric sale prices (numpy array, pandas series, or list).
    assessed_values : array-like
        Numeric assessed values (numpy array, pandas series, or list).
    group_by : array-like
        Group classifications, one per observation. Can be numeric or
        string; values are coerced to string.
    grouping_label : str
        Description of the grouping variable, used in chart titles.
    population_ratios : dict
        Population-level Gini ratios, as returned by ``gini_stats``. Index
        0 holds the overall statistics, index 1 holds the statistics used
        for the residual calculation.
    export_directory : str
        Directory where generated chart images are saved.
    dict_results : dict, optional
        Accumulator dictionary tracking exported file paths under the
        ``'exported_files'`` key. Created internally as
        ``{'exported_files': []}`` when None.
    unbiased : bool, default False
        Use True if sample size < 30, to apply the small-sample correction.
    decomp_chart : str, default ''
        Type of chart to generate for the decomposition:

        * 'bar' -- single bar chart with within/between/residual effects.
        * 'pie_with_bar' -- pie chart with 'within' values decomposed in a bar chart on the right hand side.
        * '' -- no chart is generated.
    weights : array-like, optional
        Observation weights, assumed to have mean = 1.

    Returns
    -------
    tuple of (dict, dict)
        `group_results` -- the Gini decomposition results (within,
        between, residual, and optionally chart objects); and
        `dict_results` -- the accumulator dict with any exported chart
        file paths appended.
    """
    # cleanup
    try:
        if dict_results is None:
            dict_results = {'exported_files': []}
        # Coerce inputs to 1D numpy arrays so that we can build a
        # working pandas DataFrame internally. ``pd.Series`` /
        # ``polars.Series`` / ``nw.Series`` and lists are all accepted.
        if not isinstance(sale_prices, np.ndarray):
            sale_prices = nw_utils.to_numpy_1d(sale_prices)
        if not isinstance(assessed_values, np.ndarray):
            assessed_values = nw_utils.to_numpy_1d(assessed_values)
        if not isinstance(group_by, np.ndarray):
            group_by = nw_utils.to_numpy_1d(group_by)
        if weights is not None and not isinstance(weights, np.ndarray):
            weights = nw_utils.to_numpy_1d(weights)

        # consolidate input
        if weights is None:
            weights = np.ones(len(sale_prices), dtype=float)
        df_g = pd.DataFrame({
            'sale_prices': np.asarray(sale_prices, dtype=float),
            'assessed_values': np.asarray(assessed_values, dtype=float),
            'group': np.asarray(group_by, dtype=str),
            'weights': np.asarray(weights, dtype=float),
        })

        # create list of groups
        list_of_groups = np.unique(np.asarray(group_by, dtype=str)).tolist()

        # check if there are at least two groups
        # if len(list_of_groups) <= 1:
        #     if gen_log:
        #         logging.warning('input error: no group_by found in sales dataframe')
        #     return ''

        # Calculations
        group_results = {}
        group_results['n'] = len(df_g.index)
        group_results['total_price'] = df_g['sale_prices'].sum()
        group_results['total_assessment'] = df_g['assessed_values'].sum()
        group_results['Gini_within_groups'] = {'group_total': {'Weighted_Gini_Sale_Prices': 0, 'Weighted_Gini_Assessments': 0, 'Weighted_Concentration_Index': 0, 'Weighted_Kakwani_Index': 0, 'Weighted_Modified_Kakwani_Index': 0, 'Weighted_Redistributive_Effect': 0, 'Weighted_Reranking_Effect': 0, 'Weighted_Reranking_Index': 0}}

        df_between_groups = pd.DataFrame(columns=['sale_price', 'assessed', 'weights', 'fractional_rank'])
        # loop through each group
        for group in list_of_groups:
            g = df_g[df_g['group'] == group]
            # if len(g.index) <= 1:
            #     if gen_log:
            #         logging.warning('Gini Decomposition: Group ' + grouping_label + ' = ' + group + ': invalid due to limited sale count')
                # continue
            # gini_plot_ref = model_name + ' ' + grouping_label + ' ' + group

            group_results['Gini_within_groups'][group] = {'n': len(g.index), 'group_price': g['sale_prices'].sum(), 'group_assessment': g['assessed_values'].sum()}
            group_price_weight = (group_results['Gini_within_groups'][group]['n'] / group_results['n']) * (group_results['Gini_within_groups'][group]['group_price'] / group_results['total_price'])
            group_assessment_weight = (group_results['Gini_within_groups'][group]['n'] / group_results['n']) * (group_results['Gini_within_groups'][group]['group_assessment'] / group_results['total_assessment'])
            group_results['Gini_within_groups'][group]['group_price_weight'] = group_price_weight
            group_results['Gini_within_groups'][group]['group_assessment_weight'] = group_assessment_weight
            group_stats = gini_stats(g['sale_prices'], g['assessed_values'], unbiased=unbiased, weights=g['weights'])
            group_results['Gini_within_groups'][group]['group_gini_stats'] = group_stats
            group_results['Gini_within_groups'][group]['Weighted_Gini_Sale_Prices'] = group_price_weight * group_stats['Gini_Sale_Prices']
            group_results['Gini_within_groups'][group]['Weighted_Gini_Assessments'] = group_assessment_weight * group_stats['Gini_Assessments']
            group_results['Gini_within_groups'][group]['Weighted_Concentration_Index'] = group_assessment_weight * group_stats['Concentration_Index']
            group_results['Gini_within_groups'][group]['Weighted_Redistributive_Effect'] = group_price_weight * group_stats['Redistributive_Effect']
            group_results['Gini_within_groups'][group]['Weighted_Reranking_Effect'] = group_price_weight * group_stats['Reranking_Effect']
            group_results['Gini_within_groups'][group]['Weighted_Reranking_Index'] = group_price_weight * group_stats['Reranking_Index']
            group_results['Gini_within_groups'][group]['Weighted_Kakwani_Index'] = group_price_weight * group_stats['Regressions']['Kakwani_Index']['slope']['beta']

            _group_mki = (
                group_stats['Concentration_Index'] / group_stats['Gini_Sale_Prices']
                if group_stats['Gini_Sale_Prices'] > 0 else np.nan
            )
            group_results['Gini_within_groups'][group]['Weighted_Modified_Kakwani_Index'] = group_price_weight * _group_mki
            group_n = group_results['Gini_within_groups'][group]['n']
            group_df_mean = pd.DataFrame({
                'sale_price': np.full(group_n, g['sale_prices'].mean()),
                'assessed': np.full(group_n, g['assessed_values'].mean()),
                'weights': np.full(group_n, g['weights'].mean()),
            })
            # average rank calculated with sum of arithmetic progression / n
            running_total = len(df_between_groups)
            group_df_mean['fractional_rank'] = ((group_n / 2) * ((2 * (running_total + 1)) + (group_n - 1)) / group_n)
            df_between_groups = pd.concat([df_between_groups, group_df_mean], axis=0, ignore_index=True)

            group_results['Gini_within_groups']['group_total']['Weighted_Gini_Sale_Prices'] += group_results['Gini_within_groups'][group]['Weighted_Gini_Sale_Prices']
            group_results['Gini_within_groups']['group_total']['Weighted_Gini_Assessments'] += group_results['Gini_within_groups'][group]['Weighted_Gini_Assessments']
            group_results['Gini_within_groups']['group_total']['Weighted_Concentration_Index'] += group_results['Gini_within_groups'][group]['Weighted_Concentration_Index']
            group_results['Gini_within_groups']['group_total']['Weighted_Redistributive_Effect'] += group_results['Gini_within_groups'][group]['Weighted_Redistributive_Effect']
            group_results['Gini_within_groups']['group_total']['Weighted_Reranking_Effect'] += group_results['Gini_within_groups'][group]['Weighted_Reranking_Effect']
            group_results['Gini_within_groups']['group_total']['Weighted_Kakwani_Index'] += group_results['Gini_within_groups'][group]['Weighted_Kakwani_Index']

        group_results['Gini_within_groups']['group_total']['Weighted_Modified_Kakwani_Index'] = group_results['Gini_within_groups']['group_total']['Weighted_Concentration_Index'] / population_ratios[0]['Gini_Sale_Prices']

        group_results['Gini_within_groups']['group_total']['Weighted_Reranking_Index'] = group_results['Gini_within_groups']['group_total']['Weighted_Reranking_Effect'] / (2 * group_results['Gini_within_groups']['group_total']['Weighted_Gini_Assessments'])
        df_between_groups['fractional_rank'] = df_between_groups['fractional_rank'] / len(df_between_groups.index)
        df_between_groups.reset_index(drop=True, inplace=True)

        # between groups
        group_results['Gini_between_groups'] = gini_stats(df_between_groups['sale_price'], df_between_groups['assessed'], unbiased=unbiased, between_groups_rank=df_between_groups['fractional_rank'], weights=weights)
        group_results['Gini_between_groups']['corrected_MKI'] = group_results['Gini_between_groups']['Concentration_Index'] / population_ratios[0]['Gini_Sale_Prices']

        # residual
        Gini_rsd = _gini_residuals(population_ratios[1], group_results)
        group_results['Gini_residual'] = {'Effect': 'Residual', 'Group': 'Sales Population', 'Sales Count': population_ratios[0]['Count'], 'Gini Sale Prices': Gini_rsd['Gini Sale Prices'], 'Gini Assessments': Gini_rsd['Gini Assessments'], 'Concentration Index': Gini_rsd['Concentration Index'], 'Modified Kakwani Index': Gini_rsd['Modified Kakwani Index'], 'Kakwani Index': Gini_rsd['Kakwani Index'], 'Redistributive Effect': Gini_rsd['Redistributive Effect'], 'Reranking Effect': Gini_rsd['Reranking Effect'], 'Reranking Index': Gini_rsd['Reranking Index']}

        # prepare one pie chart for each grouping, index for Gini decomposition
        if decomp_chart in ('pie_with_bar', 'bar'):
            _use_agg_if_headless()
        if decomp_chart == 'pie_with_bar':
            group_results['Gini_Charts'] = {}
            for item in ['Gini_Sale_Prices', 'Gini_Assessments', 'Concentration_Index', 'Redistributive_Effect', 'Reranking_Effect', 'Kakwani_Index']:
                chart_filename = 'bar_chart' + '_' + grouping_label + '_' + item + '.png'
                chart_filepath = os.path.join(export_directory, chart_filename)
                # prep data
                overall_values = []
                overall_labels = []
                within_values = []
                within_labels = []
                overall_values.append(round(group_results['Gini_within_groups']['group_total']['Weighted_' + item], 5))
                overall_labels.append('Within groups')
                overall_values.append(round(group_results['Gini_between_groups'][item], 5))
                overall_labels.append('Between groups')
                overall_values.append(round(group_results['Gini_residual'][item], 5))
                overall_labels.append('Interaction effects')
                for group in list_of_groups:
                    if group in group_results['Gini_within_groups']:
                        within_values.append(round(group_results['Gini_within_groups'][group]['Weighted_' + item], 5))
                        within_labels.append('Within ' + group)
                # fig_title = item.replace('_', ' ') + ' (' + str(round(group_stats[item], 5)) + ') Decomposition by ' + grouping_label + ':'
                fig_title = item.replace('_', ' ') + ' Decomposition by ' + grouping_label + ':'
                # make figure and assign axis objects
                chart = _pie_with_bar(overall_values, overall_labels, within_values, within_labels, fig_title, bar_colours=list('bgrcmyk'), fontsize=15)
                chart[0].savefig(chart_filepath)
                dict_results['exported_files'].append(chart_filepath)

                # chart.close(chart)
                group_results['Gini_Charts'][item] = {'object': chart, 'filename': chart_filepath}

        elif decomp_chart == 'bar':
            group_results['Gini_Charts'] = {}
            colors = ['green', 'indigo', 'turquoise', 'brown', 'red', 'peru', 'lightgreen', 'olive', 'yellow', 'darkorange']
            for item in ['Gini_Sale_Prices', 'Gini_Assessments', 'Concentration_Index', 'Redistributive_Effect', 'Reranking_Effect', 'Kakwani_Index']:
                # prep data
                chart_filename = 'bar_chart' + '_' + grouping_label + '_' + item + '.png'
                chart_filepath = os.path.join(export_directory, chart_filename)
                values = []
                labels = []
                within_values = []
                within_labels = []
                values.append(round(population_ratios[0][item], 5))
                labels.append(item)
                if item in group_results['Gini_between_groups']:
                    values.append(round(group_results['Gini_between_groups'][item], 5))
                else:
                    values.append(round(group_results['Gini_between_groups']['Regressions'][item]['slope']['beta'], 5))
                labels.append('Between groups')

                for group in list_of_groups:
                    if group in group_results['Gini_within_groups']:
                        within_values.append(round(group_results['Gini_within_groups'][group]['Weighted_' + item], 5))
                        within_labels.append('Within ' + group)

                # summarized smaller effect' groups into "others groups"
                if len(list_of_groups) > 6:
                    summarized_groups = _summarize_others(within_values, within_labels, cutoff=6, reverse=True)
                    values += summarized_groups['values']
                    labels += summarized_groups['labels']

                else:
                    values += within_values
                    labels += within_labels

                values.append(round(group_results['Gini_residual'][item.replace('_', ' ')], 5))
                labels.append('Residual effects')

                fig, ax = plt.subplots(figsize=(16, 9))
                ax.barh(labels, values, color=colors)
                for s in ['top', 'bottom', 'left', 'right']:
                    ax.spines[s].set_visible(False)
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.xaxis.set_tick_params(pad=5)
                ax.yaxis.set_tick_params(pad=10)
                ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
                # ax.grid(b = True, color = 'grey', linestyle = '-.', linewidth = 0.5, alpha = 0.2)
                ax.invert_yaxis()
                for i in ax.patches:
                    plt.text(i.get_width() + 0.01, i.get_y() + 0.5, str(round((i.get_width()), 5)), fontsize=10, color='black')
                if item in group_results['Gini_between_groups']:
                    # ax.set_title(item.replace('_', ' ') + ' (' + str('{:,.5f}'.format(round(group_stats[item], 5))) + ') Decomposition by ' + grouping_label + ':', loc='left', )
                    ax.set_title(item.replace('_', ' ') + ' Decomposition by ' + grouping_label + ':', loc='left', )
                else:
                    ax.set_title(item.replace('_', ' ') + ' (' + str('{:,.5f}'.format(round(group_results['Gini_between_groups']['Regressions'][item]['slope']['beta'], 5))) + ') Decomposition by ' + grouping_label + ':', loc='left', )
                plt.savefig(chart_filepath)
                dict_results['exported_files'].append(chart_filepath)
                # plt.close(fig)
                group_results['Gini_Charts'][item] = {'object': plt, 'filename': chart_filepath}
        return group_results, dict_results
    except Exception as ex:
        raise RuntimeError('error in ratio study: gini_decomp: {ex}') from ex


# ---------------------------------------------------------------------------------------------------------------------

def _summarize_others(values, labels, cutoff=6, reverse=True):
    """
    Summarize smaller groups in a Gini decomposition into an "Others" bucket for charting.

    Parameters
    ----------
    values : list
        Numeric values for each group.
    labels : list
        Labels corresponding to each value in `values`.
    cutoff : int, default 6
        Number of items that still appear individually; items beyond
        this threshold are summed into a single "Others" entry.
    reverse : bool, default True
        Sort order used before applying the cutoff. True keeps the
        largest items represented individually and summarizes the
        smaller ones.

    Returns
    -------
    dict
        Dictionary with keys ``'values'`` and ``'labels'``, holding the
        (possibly summarized) lists.
    """
    try:
        output_values = []
        output_labels = []
        table = [*zip(values, labels)]
        table.sort(key=lambda tup: tup[0], reverse=reverse)
        if len(table) > cutoff:
            other_items = sum([pair[0] for pair in table[cutoff:]])
            table = table[:cutoff]
            table.append((other_items, 'Others'))
        for pair in table:
            output_values.append(pair[0])
            output_labels.append(pair[1])
        return {'values': output_values, 'labels': output_labels}
    except Exception as ex:
        raise RuntimeError('error in ratio study: summarize_others: {ex}') from ex

# ---------------------------------------------------------------------------------------------------------------------

def gini_factors(data, depvar_header, indvar_headers, unbiased=False):
    """
    Calculate Gini factor decomposition estimates based on Quintos (2021).

    Credit to Michael Boateng.

    Parameters
    ----------
    data : DataFrame
        pandas, polars, pyarrow, or narwhals DataFrame containing the
        dependent and independent variables.
    depvar_header : str
        Column name for the dependent variable (time-adjusted sale price).
    indvar_headers : list of str
        Column names for all independent variables used in the model.
    unbiased : bool, default False
        Whether to apply the small-sample bias correction when computing
        the concentration index used for Factor MKI.

    Returns
    -------
    pandas.DataFrame
        One row per independent variable, sorted descending by
        ``'% of Total MKI'``, with columns ``'Variable'``, ``'Mean'``,
        ``'Elasticities'``, ``'CI'``, ``'Coefficient'``,
        ``'Contribution'``, ``'Factor MKI'``, and ``'% of Total MKI'``.
        Returned as a plain pandas DataFrame regardless of the input
        DataFrame type (gating constraint of
        ``statsmodels.formula.api.ols``).
    """
    try:
        # ``smf.ols`` requires a real pandas DataFrame for the R-style
        # formula API, so coerce at the boundary.
        if not isinstance(data, pd.DataFrame):
            data = nw_utils.require_pandas(
                data, reason="gini_factors() uses statsmodels.formula.api.ols"
            )
        # Log transform the dependent variable
        data_log = data[indvar_headers + [depvar_header]].assign(tadj_amt=np.log(data[depvar_header]))
        # Check if independent variables exist in dataframe
        indvar_headers = [var for var in indvar_headers if var in data_log.columns]
        # Run a regression and preview summary results
        logmodel = smf.ols(f'{depvar_header} ~ ' + ' + '.join(indvar_headers), data=data_log).fit()
        # Extract beta values (coefficients) of our logform linear model into a dataframe
        data_log['pred_f'] = logmodel.fittedvalues

        logmodel_beta = logmodel.params.reset_index().rename(columns={'index': 'Variable', 0: 'Coefficient'}).query("Variable !=  'Intercept'")
        # Compute mean and CI (using price order) for all independent variables
        logmodel_meanCI = data_log.drop('pred_f', axis=1).agg(['mean', lambda x: _gini_wtd(x, rankvar=data_log[depvar_header], unbiased=unbiased)], axis=0). \
            T.rename(columns={'mean': 'Mean', '<lambda>': 'CI'}).rename_axis('Variable').reset_index().query("Variable !=  'tadj_amt'")
        # Bind regression beta & meanCI data together and compute the rest of necessary columns
        decomp_factors = pd.merge(logmodel_beta, logmodel_meanCI)
        decomp_factors['Elasticities'] = (decomp_factors['Coefficient'] * decomp_factors['Mean']) / data_log[depvar_header].mean()
        decomp_factors['Contribution'] = decomp_factors['Elasticities'] * decomp_factors['CI']
        decomp_factors['Factor MKI'] = decomp_factors['Contribution'] / _gini_dist(data_log[depvar_header], data_log['pred_f'], unbiased=unbiased, method='GINI_P')
        decomp_factors['% of Total MKI'] = decomp_factors['Factor MKI'] / decomp_factors['Factor MKI'].sum()
        decomp_factors = decomp_factors.reindex(columns=['Variable', 'Mean', 'Elasticities', 'CI', 'Coefficient', 'Contribution', 'Factor MKI', '% of Total MKI'])
        decomp_factors = decomp_factors.sort_values(by='% of Total MKI', ascending=False).reset_index(drop=True)
        return decomp_factors
    except Exception as ex:
        raise RuntimeError('error in ratio study: gini_factors: {ex}') from ex

def _split_list(input_list, items_per_sublist):
    sublists = []
    num_sublists = -(-len(input_list) // items_per_sublist)  # Ceiling division to calculate the number of sublists needed
    items_per_sublist_adjusted = -(-len(input_list) // num_sublists)  # Calculate the adjusted items per sublist

    for i in range(0, len(input_list), items_per_sublist_adjusted):
        sublists.append(input_list[i:i + items_per_sublist_adjusted])

    return sublists

# ---------------------------------------------------------------------------------------------------------------------

def _gini_dist(price
              , assessment
              , weights=None
              , unbiased=False
              , method=None
              , na_rm=True
              , n_samples=1000
              , boot_method="BCa"
              , random_state=42
              , conf_level=None
              , samples_only=False):
    """
    Compute one or more Gini-based inequality indices, optionally with bootstrap confidence intervals.

    Credit to Michael Boateng.

    Parameters
    ----------
    price : array-like
        Sale prices.
    assessment : array-like
        Assessed values.
    weights : array-like, optional
        Observation weights, assumed to have mean = 1.
    unbiased : bool, default False
        Apply the small-sample bias correction to the underlying Gini
        calculations.
    method : str, optional
        Which index(es) to compute, as a comma-separated string, or
        None for all of them. Recognized keys (case-insensitive):

        * 'GINI_P' -- Gini coefficient of prices.
        * 'GINI_A' -- Gini coefficient of assessments.
        * 'CI_A' -- Concentration index of assessments (ranked by price).
        * 'KI' -- Kakwani index (``CI_A - GINI_P``).
        * 'MKI' -- Modified Kakwani index (``CI_A / GINI_P``).
        * 'RE' -- Redistributive effect (``GINI_A - GINI_P``).
        * 'RR' -- Reranking effect (``GINI_A - CI_A``).
        * 'RRI' -- Reranking index (``(GINI_A - CI_A) / GINI_A``).
    na_rm : bool, default True
        Drop NaNs before computing the indices.
    n_samples : int, default 1000
        Number of bootstrap resamples, used only when `conf_level` is set.
    boot_method : str, default 'BCa'
        Bootstrap method passed to ``scipy.stats.bootstrap``.
    random_state : int, default 42
        Seed for the bootstrap resampling.
    conf_level : float, optional
        Confidence level for bootstrap confidence intervals. When None,
        no bootstrapping is performed and plain point estimates are
        returned.
    samples_only : bool, default False
        Not read within this function's own body; present for
        signature parity with `_gini_robust`, which this function calls
        internally with ``samples_only=True`` to collect raw bootstrap
        samples.

    Returns
    -------
    float or dict
        * If exactly one method is requested and `conf_level` is None,
          the scalar index value.
        * If exactly one method is requested and `conf_level` is set, a
          dict with keys ``'population_idx'``, ``'sampled_median_idx'``,
          ``'sampled_mean_idx'``, ``'sampled_lowci_idx'``,
          ``'sampled_highci_idx'``, ``'t_stat'``, and ``'p_value'``.
        * If multiple methods are requested, a dict keyed by method
          name, with values in either of the shapes above depending on
          `conf_level`.
    """
    # Input checking and error handling
    try:
        _check_inputs(price, assessment)
        # Set method to uppercase and provide default
        method = "ALL" if method is None else method.upper()

        # Prepare variables for all possible methods
        ginip = _gini_robust(price, price, weights, unbiased, na_rm)
        ginia = _gini_robust(assessment, assessment, weights, unbiased, na_rm)
        cia = _gini_robust(assessment, price, weights, unbiased, na_rm)

        # Prepare variables for bootstrapped samples if conf_level is provided
        if conf_level:
            ginip_samples = _gini_robust(price, price, weights, unbiased, na_rm, n_samples, boot_method, random_state, conf_level, samples_only=True)
            ginia_samples = _gini_robust(assessment, assessment, weights, unbiased, na_rm, n_samples, boot_method, random_state, conf_level, samples_only=True)
            cia_samples = _gini_robust(assessment, price, weights, unbiased, na_rm, n_samples, boot_method, random_state, conf_level, samples_only=True)

            samples = {'gini_p': ginip_samples, 'gini_a': ginia_samples, 'ci_a': cia_samples,
                       'ki': (cia_samples - ginip_samples), 'mki': (cia_samples / ginip_samples),
                       're': (cia_samples - ginip_samples), 'rr': (ginia_samples - cia_samples),
                       'rri': ((ginia_samples - cia_samples) / (ginia_samples))}

            # Create a dictionary of methods and their calculations
        ineq_methods = {
            "GINI_P": lambda: ginip,
            "GINI_A": lambda: ginia,
            "CI_A": lambda: cia,
            "KI": lambda: (cia - ginip),
            "MKI": lambda: (cia / ginip),
            "RE": lambda: (ginia - ginip),
            "RR": lambda: (ginia - cia),
            "RRI": lambda: ((ginia - cia) / ginia),
        }

        ineq = {}
        methods_to_run = list(ineq_methods.keys()) if method == "ALL" else [m.strip() for m in method.split(",")]
        for m in methods_to_run:
            if conf_level:
                if m.lower() not in samples:
                    samples[m.lower()] = _gini_robust(price, price, weights, unbiased, na_rm, n_samples, boot_method,
                                                     random_state, conf_level, samples_only=True)
                pop_gini = ineq_methods[m]()

                try:
                    from scipy.stats import bootstrap, ttest_1samp
                except ImportError:
                    raise ImportError("The scipy library is required to use the bootstrapping option")

                t_stat, p_value = ttest_1samp(samples[m.lower()], pop_gini)
                confint = bootstrap((samples[m.lower()],), np.median, n_resamples=n_samples,
                                    confidence_level=conf_level, method=boot_method).confidence_interval
                ineq[m] = {'population_idx': pop_gini, 'sampled_median_idx': np.median(samples[m.lower()]),
                           'sampled_mean_idx': np.mean(samples[m.lower()]), 'sampled_lowci_idx': confint.low,
                           'sampled_highci_idx': confint.high, 't_stat': t_stat, 'p_value': p_value}
            else:
                ineq[m] = ineq_methods[m]()

        if len(methods_to_run) == 1:
            return ineq[methods_to_run[0]]
        return ineq
    except Exception as ex:
        raise RuntimeError('error in ratio study: _gini_dist: {ex}') from ex


# ---------------------------------------------------------------------------------------------------------------------


def _gini_residuals(gini_no_groups, gini_groups):
    """
    Calculate the residual (interaction) effects in a Gini decomposition.

    Parameters
    ----------
    gini_no_groups : dict
        Gini statistics for the population without grouping (i.e.
        totals), as returned by `gini_stats`.
    gini_groups : dict
        Gini decomposition results with groups, including within-group
        and between-group components (the `group_results` produced by
        `gini_decomp`).

    Returns
    -------
    dict
        Residual values for each indicator, with keys
        ``'Gini Sale Prices'``, ``'Gini Assessments'``,
        ``'Concentration Index'``, ``'Modified Kakwani Index'``,
        ``'Kakwani Index'``, ``'Redistributive Effect'``,
        ``'Reranking Effect'``, and ``'Reranking Index'``.
    """
    try:
        result = {}
        result['Gini Sale Prices'] = gini_no_groups['Gini_Sale_Prices'] \
                                     - gini_groups['Gini_within_groups']['group_total']['Weighted_Gini_Sale_Prices'] \
                                     - gini_groups['Gini_between_groups']['Gini_Sale_Prices']
        result['Gini Assessments'] = gini_no_groups['Gini_Assessments'] \
                                     - gini_groups['Gini_within_groups']['group_total']['Weighted_Gini_Assessments'] \
                                     - gini_groups['Gini_between_groups']['Gini_Assessments']
        result['Concentration Index'] = gini_no_groups['Concentration_Index'] \
                                        - gini_groups['Gini_within_groups']['group_total']['Weighted_Concentration_Index'] \
                                        - gini_groups['Gini_between_groups']['Concentration_Index']

        # Paper Quintos (2021) Eq. 5: MKI is a scalar (CI_A / G_P);
        # KI is also a scalar difference (CI_A − G_P).  We therefore
        # read both as scalars from ``gini_no_groups`` (produced by
        # gini_stats), not from a regression dictionary.
        result['Modified Kakwani Index'] = gini_no_groups['Modified_Kakwani_Index'] \
                                           - gini_groups['Gini_within_groups']['group_total']['Weighted_Modified_Kakwani_Index'] \
                                           - gini_groups['Gini_between_groups']['corrected_MKI']

        result['Kakwani Index'] = gini_no_groups['Kakwani_Index'] \
                                  - gini_groups['Gini_within_groups']['group_total']['Weighted_Kakwani_Index'] \
                                  - gini_groups['Gini_between_groups']['Kakwani_Index']
        result['Redistributive Effect'] = gini_no_groups['Redistributive_Effect'] \
                                          - gini_groups['Gini_within_groups']['group_total']['Weighted_Redistributive_Effect'] \
                                          - gini_groups['Gini_between_groups']['Redistributive_Effect']
        result['Reranking Effect'] = gini_no_groups['Reranking_Effect'] \
                                     - gini_groups['Gini_within_groups']['group_total']['Weighted_Reranking_Effect'] \
                                     - gini_groups['Gini_between_groups']['Reranking_Effect']
        
        denom = 2 * result['Gini Assessments']
        result['Reranking Index'] = result['Reranking Effect'] / denom if denom > 0 else np.nan
        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: gini_residuals: {ex}') from ex

# ---------------------------------------------------------------------------------------------------------------------

def _pie_with_bar(overall_values
                 , overall_labels
                 , within_values
                 , within_labels
                 , fig_title
                 , bar_colours
                 , fontsize=15):
    """
    Create a combination pie chart + bar chart for Gini decomposition results.

    The pie chart shows overall within/between/residual shares; the bar
    chart decomposes the "within groups" share into its constituent
    groups.

    Parameters
    ----------
    overall_values : list
        Values for the pie chart slices (within, between, residual).
    overall_labels : list
        Labels for the pie chart slices.
    within_values : list
        Values for each group's contribution to the "within groups"
        share, used to build the adjoining bar chart.
    within_labels : list
        Labels for each group's bar segment.
    fig_title : str
        Title for the overall figure.
    bar_colours : list
        Colors to cycle through for the bar chart segments.
    fontsize : int, default 15
        Font size for the figure title.

    Returns
    -------
    list
        Two-element list ``[plt, fig]`` -- the ``matplotlib.pyplot``
        module and the generated ``Figure`` object.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
        fig.subplots_adjust(wspace=0)
        explode = [0.05] * len(overall_values)
        angle = -200 * overall_values[2]

        wedges, *_ = ax1.pie(overall_values, autopct='%1.1f%%', startangle=angle, labels=overall_labels
                             , explode=explode)

        bottom = 1
        width = .2

        # Adding from the top matches the legend.
        list_within = [*zip(within_values, within_labels)]
        list_within.sort(key=lambda tup: tup[0], reverse=True)
        # summarize into 'other groups' smaller groups
        if len(list_within) > 6:
            other_items = sum([pair[0] for pair in list_within[6:]])
            list_within = list_within[:6]
            list_within.append((other_items, 'Other groups'))
        for j, (height, label) in enumerate(list_within):
            bottom -= height
            bc = ax2.bar(0, height, width, bottom=bottom, color=bar_colours[j], label=label, alpha=0.5)
            ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        # ax2.set_title('Within Groups', loc = 'right')
        ax2.legend(loc='center right')
        ax2.axis('off')
        ax2.set_xlim(- 2 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(within_values)

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
        con.set_color((0, 0, 0))
        # con.set_color([0, 0, 0])
        con.set_linewidth(2)
        ax2.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        ax2.add_artist(con)
        con.set_linewidth(2)
        fig.suptitle(fig_title, fontsize=fontsize)
        # plt.show()
        return [plt, fig]
    except Exception as ex:
        raise RuntimeError('error in ratio study: pie_with_bar: {ex}') from ex

# ---------------------------------------------------------------------------------------------------------------------

def _format_gini_decomp(gini_dict, grouping, population_ratios):
    """
    Format Gini decomposition results into a display table, styled after Quintos (2021).

    Parameters
    ----------
    gini_dict : dict
        Gini decomposition results (the `group_results` produced by
        `gini_decomp`), containing within-group, between-group, and
        residual effects.
    grouping : str
        Description of the grouping variable(s), shown in the "Within"
        section label.
    population_ratios : dict
        Population-level Gini ratios without grouping (i.e. totals);
        index 0 holds the overall statistics used for the "Total" row.

    Returns
    -------
    pandas.DataFrame
        Formatted table with rows for Total, Within (one per group),
        Between, and Residual effects, and columns ``'Effect'``,
        ``'Group'``, ``'Sales Count'``, ``'Gini Sale Prices'``,
        ``'Gini Assessments'``, ``'Concentration Index'``,
        ``'Modified Kakwani Index'``, ``'Kakwani Index'``,
        ``'Redistributive Effect'``, ``'Reranking Effect'``, and
        ``'Reranking Index'``, with numeric values pre-formatted as
        strings.
    """
    try:
        output = []
        results_format = {'Sales Count': '{:,.0f}', 'Gini Sale Prices': '{:,.6f}'
            , 'Gini Assessments': '{:,.6f}', 'Concentration Index': '{:,.6f}', 'Modified Kakwani Index': '{:,.6f}'
            , 'Kakwani Index': '{:,.6f}', 'Redistributive Effect': '{:,.6f}', 'Reranking Effect': '{:,.6f}'
            , 'Reranking Index': '{:.4%}', 'All others': '{:,.6f}'}

        blank_row = {'Sales Count': math.nan, 'Gini Sale Prices': math.nan, 'Gini Assessments': math.nan
            , 'Concentration Index': math.nan, 'Modified Kakwani Index': math.nan, 'Kakwani Index': math.nan
            , 'Redistributive Effect': math.nan, 'Reranking Effect': math.nan, 'Reranking Index': math.nan}

        # add totals: first row of results
        output.append({'Effect': 'Total'
                          , 'Group': 'Sales Population'
                          , 'Sales Count': population_ratios[0]['Count']
                          , 'Gini Sale Prices': population_ratios[0]['Gini_Sale_Prices']
                          , 'Gini Assessments': population_ratios[0]['Gini_Assessments']
                          , 'Concentration Index': population_ratios[0]['Concentration_Index']
                          , 'Modified Kakwani Index': population_ratios[0]['Modified_Kakwani_Index']
                          , 'Kakwani Index': population_ratios[0]['Kakwani_Index']
                          , 'Redistributive Effect': population_ratios[0]['Redistributive_Effect']
                          , 'Reranking Effect': population_ratios[0]['Reranking_Effect']
                          , 'Reranking Index': population_ratios[0]['Reranking_Index']})
        output.append(blank_row)

        # add within groups
        output.append({'Effect': 'Within', 'Group': 'by: ' + grouping})

        for group_within in gini_dict['Gini_within_groups'].keys():
            output.append({'Effect': '', 'Group': group_within
                              , 'Sales Count': gini_dict['Gini_within_groups'][group_within]['n'] if group_within != 'group_total' else math.nan
                              , 'Gini Sale Prices': gini_dict['Gini_within_groups'][group_within]['Weighted_Gini_Sale_Prices']
                              , 'Gini Assessments': gini_dict['Gini_within_groups'][group_within]['Weighted_Gini_Assessments']
                              , 'Concentration Index': gini_dict['Gini_within_groups'][group_within]['Weighted_Concentration_Index']
                              , 'Modified Kakwani Index': gini_dict['Gini_within_groups'][group_within]['Weighted_Modified_Kakwani_Index'] if group_within == 'group_total' else math.nan
                              , 'Kakwani Index': gini_dict['Gini_within_groups'][group_within]['Weighted_Kakwani_Index']
                              , 'Redistributive Effect': gini_dict['Gini_within_groups'][group_within]['Weighted_Redistributive_Effect']
                              , 'Reranking Effect': gini_dict['Gini_within_groups'][group_within]['Weighted_Reranking_Effect']
                              , 'Reranking Index': gini_dict['Gini_within_groups'][group_within]['Weighted_Reranking_Index']})

        output.append(blank_row)
        # add between groups
        output.append({'Effect': 'Between', 'Group': 'Sales Population'
                          , 'Sales Count': population_ratios[0]['Count']
                          , 'Gini Sale Prices': gini_dict['Gini_between_groups']['Gini_Sale_Prices']
                          , 'Gini Assessments': gini_dict['Gini_between_groups']['Gini_Assessments']
                          , 'Concentration Index': gini_dict['Gini_between_groups']['Concentration_Index']
                          , 'Modified Kakwani Index': gini_dict['Gini_between_groups']['corrected_MKI']
                          , 'Kakwani Index': gini_dict['Gini_between_groups']['Kakwani_Index']
                          , 'Redistributive Effect': gini_dict['Gini_between_groups']['Redistributive_Effect']
                          , 'Reranking Effect': gini_dict['Gini_between_groups']['Reranking_Effect']
                          , 'Reranking Index': gini_dict['Gini_between_groups']['Reranking_Index']})

        output.append(blank_row)
        # add residual
        output.append(gini_dict['Gini_residual'])

        output = pd.DataFrame(output)
        output = custom_format(output, results_format.keys(), results_format, skip=['Effect', 'Group', 'temp_standards'], all_str=False)
        return output
    except Exception as ex:
        raise RuntimeError('error in ratio study: format_gini_decomp: {ex}') from ex

# ---------------------------------------------------------------------------------------------------------------------



# def _gini_wtd(x, rankvar, weights=None, unbiased=False):
#     """
#     Credit to Micheal Boateng
#     """
#     try:
#         x = np.asarray(x)
#         if weights is not None:
#             weights = np.asarray(weights)
#             sorted_indices = np.argsort(rankvar)
#             sorted_x = x[sorted_indices]
#             sorted_w = weights[sorted_indices]
#             # Force float dtype to avoid overflows
#             cumw = np.cumsum(sorted_w, dtype=float)
#             cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
#             n = len(x)
#             gini = (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1]))
#         else:
#             # sorted_x = np.sort(x)
#             sorted_x = np.array(sorted(zip(x, rankvar), key=lambda x: x[1]))[:, 0]
#             n = len(x)
#             cumx = np.cumsum(sorted_x, dtype=float)
#             # The above formula, with all weights equal to 1 simplifies to:
#             gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
#         if unbiased:
#             gini = n / (n - 1) * gini
#         return gini
#     except Exception as ex:
#         raise RuntimeError('error in ratio study: gini_wtd: {ex}') from ex
    

def _gini_wtd(x, rankvar, weights=None, *, unbiased=False) -> float:
    """
    Compute a weighted Gini coefficient, normalizing weights before delegating to `_gini_coeff`.

    Parameters
    ----------
    x : array-like
        Values whose distribution is measured.
    rankvar : array-like
        Variable that determines the ranking order.
    weights : array-like, optional
        Observation weights. Normalized internally to sum to
        ``len(weights)`` (mean = 1) before use.
    unbiased : bool, default False
        Apply the small-sample bias correction ``n / (n - 1)``.

    Returns
    -------
    float
        Weighted Gini coefficient.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    rv_arr = np.asarray(rankvar, dtype=np.float64)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        w_arr = w_arr / w_arr.sum() * len(w_arr)
    else:
        w_arr = None
    return _gini_coeff(x_arr, rv_arr, unbiased=unbiased, weights=w_arr)

# ---------------------------------------------------------------------------------------------------------------------

# def _gini_coeff(prices, rankvar, unbiased=False, between_groups_rank=False, weights=None):
#     """
#     based on code by David Zaslavsky: https://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
#     calculates the gini coefficient, using a values array and a ranking array, both must be the same lenght and
#     indexed equally.
#     requires all values in x to be zero or positive numbers, otherwise results are undefined
#     :param prices: numpy array List of values for calculation
#     :param rankvar: numpy array - Values for sorting order of x.
#      For sale prices, repeat values array for rankvar; for assessments, use sale prices array
#     :param unbiased: BOOL. True if estimate is to be ubiased, False (default) for no correction to be applied.
#     :param between_groups_rank: Array. optional, used for between groups gini calculations, where the ranking isn't either in
#     prices nor in rankvar.
#     :param weights: Optional. array of weights, assumed to have mean = 1.
#     return: Numeric gini index
#      """
#     try:
#         prices_reduced_size = prices / 1000  # reduce size of prices to avoid overflow errors
#         prices_reduced_size = prices_reduced_size.astype(np.float64)
#         n = len(prices_reduced_size)
#         s = prices_reduced_size.sum()
#         if n < 3 or s == 0:
#             # print('Warning: Gini coefficient cannot be calculated for less than 3 values or when sum of values is zero.')
#             return 0.0
#         if isinstance(between_groups_rank, pd.Series):
#             r = between_groups_rank
#         else:
#             r = np.argsort(np.argsort(-rankvar))  # calculates zero-based ranks
#         if weights is not None:
#             weights = np.asarray(weights)
#             sorted_indices = np.argsort(rankvar)
#             sorted_x = prices_reduced_size[sorted_indices]
#             sorted_w = weights[sorted_indices]
#             # Force float dtype to avoid overflows
#             cumw = np.cumsum(sorted_w, dtype=float)
#             cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
#             result = (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1]))
#         else:
#             result = 1 - (2.0 * (r * prices_reduced_size).sum() + s) / (n * s)
#         if unbiased == True:
#             result = result * (n / (n - 1))
#         return result
#     except Exception as ex:
#         raise RuntimeError('error in ratio study: gini_coeff:: {ex}') from ex

def _gini_coeff(
    prices: np.ndarray,
    rankvar: np.ndarray,
    *,
    unbiased: bool = False,
    between_groups_rank=None,
    weights=None,
) -> float:
    """
    Compute a Gini-type coefficient for `prices` ranked by `rankvar`.

    When ``prices is rankvar`` this yields the standard Gini coefficient.
    When `prices` are assessments and `rankvar` are sale prices, the
    result is a concentration index.

    Parameters
    ----------
    prices : np.ndarray
        Values whose distribution is measured (>= 0).
    rankvar : np.ndarray
        Variable that determines the ranking order.
    unbiased : bool, default False
        Apply small-sample correction ``n / (n - 1)``.
    between_groups_rank : np.ndarray, optional
        Pre-computed fractional ranks for between-groups decomposition.
    weights : np.ndarray, optional
        Observation weights (mean approx 1).

    Returns
    -------
    float
        Gini / concentration index. ``np.nan`` when ``n < MIN_SAMPLE_GINI``
        (3) or when the sum of `prices` is zero.
    """
    prices_scaled = _gini_rescale(np.asarray(prices, dtype=np.float64))
    n = len(prices_scaled)
    s = float(prices_scaled.sum())

    if n < MIN_SAMPLE_GINI or s == 0:
        return np.nan

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        order = np.argsort(rankvar)
        sorted_x = prices_scaled[order]
        sorted_w = w[order]
        cumw = np.cumsum(sorted_w, dtype=np.float64)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=np.float64)
        result = float(
            np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:])
            / (cumxw[-1] * cumw[-1])
        )
    elif between_groups_rank is not None:
        r = np.asarray(between_groups_rank, dtype=np.float64)
        result = 1.0 - (2.0 * (r * prices_scaled).sum() + s) / (n * s)
    else:
        r = np.argsort(np.argsort(-rankvar)).astype(np.float64)
        result = 1.0 - (2.0 * (r * prices_scaled).sum() + s) / (n * s)

    if unbiased:
        result *= n / (n - 1)

    return float(result)


gini_coeff = _gini_coeff
# ---------------------------------------------------------------------------------------------------------------------

def _gini_rescale(prices: np.ndarray) -> np.ndarray:
    """
    Rescale `prices` to unit mean to keep cumulative sums in a safe float64 range.

    Gini and concentration indices are scale-invariant, so dividing by
    any positive constant leaves the result unchanged. Dividing by the
    mean puts values in O(1), which keeps ``cumsum(prices * weights)``
    well under float64's exact-integer ceiling (2^53 ~ 9e15) and avoids
    precision loss on large datasets (e.g. 500K rows x $5M prices, where
    naive cumsum products can reach ~2.5e17 and silently lose digits).

    Parameters
    ----------
    prices : np.ndarray
        Values to rescale.

    Returns
    -------
    np.ndarray
        `prices` cast to float64 and divided by its mean. Returned
        with only the float64 cast applied (unscaled) when the mean is
        zero or non-finite -- that degenerate case is handled by the
        caller's ``n < MIN_SAMPLE_GINI`` / ``s == 0`` guard.
    """
    arr = np.asarray(prices, dtype=np.float64)
    mean = arr.mean()
    if not np.isfinite(mean) or mean == 0:
        return arr
    return arr / mean

# ---------------------------------------------------------------------------------------------------------------------

def gini_lorenz_plot(market_value, assessment, gini_sale_prices,
                      concentration_index, title, filepath=''):
    """
    Generate Lorenz curves for sale prices and assessments.

    Parameters
    ----------
    market_value : array-like
        Sale prices (numpy array, pandas/polars/narwhals Series, or list).
    assessment : array-like
        Assessed values (numpy array, pandas/polars/narwhals Series, or list).
    gini_sale_prices : float
        Gini coefficient of sale prices, shown in the legend label.
    concentration_index : float
        Concentration index, shown in the legend label.
    title : str
        Plot title.
    filepath : str, default ''
        Path to save the chart as a PNG. When empty, the figure object
        is returned instead of being saved.

    Returns
    -------
    str or matplotlib.figure.Figure
        The saved file path (with a ``.png`` extension appended if
        missing) when `filepath` is given; otherwise the
        ``matplotlib.figure.Figure`` object.
    """

    sa = nw_utils.to_numpy_1d(market_value).astype(np.float64)
    av = nw_utils.to_numpy_1d(assessment).astype(np.float64)
    order = np.argsort(sa)
    x_cum = np.cumsum(sa[order]) / sa.sum()
    y_cum = np.cumsum(av[order]) / av.sum()
    n = len(x_cum)

    _use_agg_if_headless()
    fig, ax = plt.subplots()
    xs = np.arange(n) / float(n - 1)
    ax.plot(xs, x_cum,
            label=f"Gini Sale Prices: {gini_sale_prices:.5f}")
    ax.plot(xs, y_cum,
            label=f"Concentration Index: {concentration_index:.5f}")
    ax.plot([0, 1], [0, 1], "k--", label="Equality Line")
    ax.legend(loc="lower right")
    ax.set_xlabel("Cumulative Proportion of Properties")
    ax.set_ylabel("Cumulative Proportion of Property Values")
    ax.set_title(title)

    if filepath:
        fp = filepath if filepath.endswith(".png") else filepath + ".png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        return fp
    plt.close(fig)
    return fig


def gini_stats(market_value
               , assessment
               , weights=None
               , unbiased=False
               , between_groups_rank=False
               , run_regression=True
               , plot_ref=''
               , gini_lorenz_plots=False
               , export_directory=''):
    """
    Calculate Gini equality indicators for sale prices and assessments.

    Parameters
    ----------
    market_value : array-like
        Numeric sale prices (numpy array, pandas series, or list).
    assessment : array-like
        Numeric assessed values (numpy array, pandas series, or list).
    weights : array-like, optional
        Observation weights, assumed to have mean = 1.
    unbiased : bool, default False
        Use True if sample size < 30, to apply the small-sample
        correction.
    between_groups_rank : array-like or False, default False
        For between-groups decomposition, provide a specific fractional
        rank array to use instead of the default within-sample ranking.
    run_regression : bool, default True
        Enables calculation of the Reranking Effect, Reranking Index,
        Kakwani Index, and Modified Kakwani Index (plus their
        supporting regressions and p-values) in addition to the basic
        Gini coefficients.
    plot_ref : str, default ''
        Text used in the plot title. Only used when `gini_lorenz_plots`
        is True.
    gini_lorenz_plots : bool, default False
        If True, generate a Lorenz curve chart via `gini_lorenz_plot`.
    export_directory : str, default ''
        Directory where the Lorenz curve chart is saved, when
        `gini_lorenz_plots` is True.

    Returns
    -------
    dict
        Dictionary of Gini indicators, including
        ``'Gini_Sale_Prices'``, ``'Gini_Assessments'``,
        ``'Concentration_Index'``, and ``'Redistributive_Effect'``.
        When `run_regression` is True, also includes ``'Regressions'``
        (per-indicator regression details), ``'Kakwani_Index'``,
        ``'Modified_Kakwani_Index'``, ``'Reranking_Effect'``, and
        ``'Reranking_Index'``. When `gini_lorenz_plots` is True, also
        includes ``'gini_lorenz_plot'`` with the saved chart's file
        path.
    """
    try:
        result = {}
        # Coerce inputs to numpy arrays; accept pandas/polars/narwhals Series
        # and lists transparently.
        sale_prices = nw_utils.to_numpy_1d(market_value).astype(np.float64)
        assessed_values = nw_utils.to_numpy_1d(assessment).astype(np.float64)
        if weights is not None:
            sample_weights = nw_utils.to_numpy_1d(weights).astype(np.float64)
        else:
            sample_weights = None

        if len(sale_prices) != len(assessed_values):
            raise ValueError('input error: values list and ranking variables have different lenghts:', len(sale_prices), 'and', len(assessed_values), 'respectively.')

        elif sample_weights is not None and len(sale_prices) != len(sample_weights):
            raise ValueError('input error: values list and weights variables have different lenghts:', len(sale_prices), 'and', len(sample_weights), 'respectively.')

        else:
            # Build a pandas DataFrame for the regression / dataframe-style
            # operations that follow.  This is cheap (a few hundred rows for
            # the per-call input) and matches the existing semantics.
            # Cast everything to float up front so that downstream
            # ``np.isnan``/``sm.OLS`` arithmetic stays in float64.
            df = pd.DataFrame({
                'sale_prices': np.asarray(sale_prices, dtype=np.float64),
                'assessed_values': np.asarray(assessed_values, dtype=np.float64),
            })
            if sample_weights is not None:
                df['weights'] = np.asarray(sample_weights, dtype=np.float64)
            # cleanup the data
            df = df.dropna().reset_index(drop=True)
            df = df[(df['sale_prices'] > 0) & (df['assessed_values'] > 0)]
            # run gini coefficient
            n = len(df)
            # build lorenz curves for sale prices and assessment values
            if isinstance(between_groups_rank, (pd.Series, np.ndarray)):
                df['Share_of_Rank'] = np.asarray(between_groups_rank)
            else:
                df['Share_of_Rank'] = 1 / n
            df.sort_values('sale_prices', inplace=True)
            df['Share_of_Prices'] = df['sale_prices'] / df['sale_prices'].sum()
            df['Share_of_Assessments'] = df['assessed_values'] / df['assessed_values'].sum()
            dfA = df.copy()
            if isinstance(between_groups_rank, (pd.Series, np.ndarray)):
                df['Cummulative_Share'] = np.asarray(between_groups_rank)
            else:
                df['Cummulative_Share'] = df['Share_of_Rank'].cumsum()
            df['C_SP'] = df['Share_of_Prices'].cumsum()
            df['C_SA'] = df['Share_of_Assessments'].cumsum()
            dfA.sort_values('assessed_values', inplace=True)
            if isinstance(between_groups_rank, (pd.Series, np.ndarray)):
                dfA['Cummulative_Share'] = np.asarray(between_groups_rank)
            else:
                dfA['Cummulative_Share'] = dfA['Share_of_Rank'].cumsum()
            dfA['C_SA'] = dfA['Share_of_Assessments'].cumsum()
            # Prep dependent Variables
            var_R = df['Cummulative_Share'].var(ddof=0)
            mean_SP = df['Share_of_Prices'].mean()
            mean_SA = df['Share_of_Assessments'].mean()
            df['Y_Prices'] = 2 * var_R * (df['Share_of_Prices'] / mean_SP)
            df['Y_Assessment'] = 2 * var_R * (df['Share_of_Assessments'] / mean_SA)
            df['Y_CI'] = 2 * var_R * (df['Share_of_Assessments'] / mean_SA)
            df['Y_KI'] = df['Y_CI'] - df['Y_Prices']
            df['Y_MKI_1'] = df['Share_of_Prices'] / mean_SP
            df['Y_MKI_2'] = df['Share_of_Assessments'] / mean_SA
            dfA['Y_Assessment'] = 2 * var_R * (dfA['Share_of_Assessments'] / mean_SA)
            # Gini and CI indexes
            w_arr = df['weights'].to_numpy() if 'weights' in df.columns else None
            result['Gini_Sale_Prices'] = gini_coeff(df['sale_prices'].to_numpy(), df['sale_prices'].to_numpy(), unbiased=unbiased, weights=w_arr)
            result['Gini_Assessments'] = gini_coeff(df['assessed_values'].to_numpy(), df['assessed_values'].to_numpy(), unbiased=unbiased, weights=w_arr)
            result['Concentration_Index'] = gini_coeff(df['assessed_values'].to_numpy(), df['sale_prices'].to_numpy(), unbiased=unbiased, weights=w_arr)
            result['Redistributive_Effect'] = result['Gini_Assessments'] - result['Gini_Sale_Prices']
            # Vertical Equity Regressions
            if run_regression:
                result['Regressions'] = {}
                _w_arg = df['weights'].to_numpy() if 'weights' in df.columns else None
                result['Regressions']['Gini_Sale_Prices'] = _simple_regression(df['Y_Prices'].to_numpy(), df['Cummulative_Share'].to_numpy(), weights=_w_arg)
                result['Regressions']['Gini_Assessments'] = _simple_regression(dfA['Y_Assessment'].to_numpy(), dfA['Cummulative_Share'].to_numpy(), weights=_w_arg)
                result['Regressions']['Concentration_Index'] = _simple_regression(df['Y_CI'].to_numpy(), df['Cummulative_Share'].to_numpy(), weights=_w_arg)
                result['Regressions']['Kakwani_Index'] = _simple_regression(df['Y_KI'].to_numpy(), df['Cummulative_Share'].to_numpy(), weights=_w_arg)
                # The original code computed a regression slope of Y_MKI_2 on the
                # fitted values of Y_MKI_1 ~ R as a stand-in for MKI. Per Quintos
                # (2021) Eq. (5), MKI is a *ratio* CI_A / G_P, not a regression
                # slope; the two produce materially different numbers (e.g. the
                # paper's worked Model A: paper MKI = 0.79155, regression slope ≈
                # 0.79x but not the same). The diagnostic regression is preserved
                # below for users who want the alternative formulation.
                reg_MKI_1 = _simple_regression(df['Y_MKI_1'].to_numpy(), df['Cummulative_Share'].to_numpy(), weights=_w_arg)
                result['Regressions']['Modified_Kakwani_Regression_Diagnostic'] = _simple_regression(
                    x=df['Y_MKI_2'].to_numpy(), y=reg_MKI_1['fitted_values'], weights=_w_arg
                )
                # Paper-compliant MKI: scalar ratio (Eq. 5)
                result['Regressions']['Modified_Kakwani_Index'] = (
                    result['Concentration_Index'] / result['Gini_Sale_Prices']
                    if result['Gini_Sale_Prices'] > 0 else np.nan
                )
                # Expose scalar (paper-compliant) KI and MKI at the top
                # level so downstream consumers (decomposition residuals,
                # _format_gini_decomp, ratio_study.py reporting) do not
                # need to navigate the regression dict.
                result['Kakwani_Index'] = result['Regressions']['Kakwani_Index']['slope']['beta']
                result['Modified_Kakwani_Index'] = result['Regressions']['Modified_Kakwani_Index']
                result['Reranking_Effect'] = result['Redistributive_Effect'] - result['Kakwani_Index']
                result['Reranking_Index'] = (result['Reranking_Effect']) / (2 * result['Gini_Assessments']) if ((result['Gini_Assessments'] > 0) and ((result['Reranking_Effect']) / (2 * result['Gini_Assessments']) < 100)) else 0
            if plot_ref and gini_lorenz_plots:
                result['gini_lorenz_plot'] = gini_lorenz_plot(market_value
                                                              , assessment
                                                              , result['Gini_Sale_Prices']
                                                              , result['Concentration_Index']
                                                              , title='Lorenz Curve - ' + plot_ref
                                                              , filepath=os.path.join(export_directory, 'Lorenz_Curve_' + plot_ref.replace(' ', '_') + '.png'))
        return result
    except Exception as ex:
        raise RuntimeError('error in ratio study: gini_stats:: {ex}') from ex

# ---------------------------------------------------------------------------------------------------------------------

def _kakwani_index(gini_prices: float,
                   concentration_index: float) -> float:
    """
    Compute the Kakwani progressivity index: K = C - Gp.

    Positive values indicate a progressive assessment; negative values
    indicate a regressive one.

    Parameters
    ----------
    gini_prices : float
        Gini coefficient of sale prices.
    concentration_index : float
        Concentration index of assessed values (ranked by sale price).

    Returns
    -------
    float
        Kakwani index (`concentration_index` minus `gini_prices`).
        ``np.nan`` when either input is NaN.
    """
    if np.isnan(gini_prices) or np.isnan(concentration_index):
        return np.nan
    return float(concentration_index - gini_prices)

# ---------------------------------------------------------------------------------------------------------------------
def _gini_robust(x, rankvar, weights=None, unbiased=False, na_rm=True,
                 n_samples=1000, boot_method="BCa", random_state=42,
                 conf_level=None, samples_only=False):
    _check_inputs(x, rankvar)
    x = np.asarray(x, dtype=np.float64)
    rankvar = np.asarray(rankvar, dtype=np.float64)
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)

    if na_rm:
        mask = ~(np.isnan(x) | np.isnan(weights) | np.isnan(rankvar))
        x, weights, rankvar = x[mask], weights[mask], rankvar[mask]

    if len(x) == 0 or np.any(x < 0):
        return np.nan

    def _i_gini(xv, rv, w, unbiased):
        w = w / w.sum()
        order = np.argsort(rv)
        xv, w = xv[order], w[order]
        f_hat = w / 2 + np.concatenate(([0], np.cumsum(w)[:-1]))
        x_mean = np.average(xv, weights=w)
        res = 2 / x_mean * np.sum(w * (xv - x_mean) * (f_hat - np.average(f_hat, weights=w)))
        if unbiased:
            res *= 1 / (1 - np.sum(w ** 2))
        return res

    pop_gini = _i_gini(x, rankvar, weights, unbiased)

    if conf_level is None:
        return pop_gini

    from scipy.stats import bootstrap, ttest_1samp
    rng = np.random.default_rng(random_state)
    n = len(x)
    samples = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        samples[i] = _i_gini(x[idx], rankvar[idx], weights[idx], unbiased)

    if samples_only:
        return samples

    t_stat, p_value = ttest_1samp(samples, pop_gini)
    ci = bootstrap((samples,), np.median, n_resamples=n_samples,
                   confidence_level=conf_level, method=boot_method).confidence_interval
    return {
        'population_idx': pop_gini, 'sampled_median_idx': float(np.median(samples)),
        'sampled_mean_idx': float(np.mean(samples)),
        'sampled_lowci_idx': float(ci.low), 'sampled_highci_idx': float(ci.high),
        't_stat': float(t_stat), 'p_value': float(p_value),
    }


# def _gini_robust(x
#                 , rankvar
#                 , weights=None
#                 , unbiased=False
#                 , na_rm=True
#                 , n_samples=1000
#                 , boot_method="BCa"
#                 , random_state=42
#                 , conf_level=None
#                 , samples_only=False):
#     """
#     Credit to Micheal Boateng
#     """
#     try:
#         _check_inputs(x, rankvar)
#         if weights is None:
#             weights = np.ones(len(x))
#         if na_rm:
#             mask = ~(np.isnan(x) | np.isnan(weights))
#             x = x[mask]
#             weights = weights[mask]
#             rankvar = rankvar[mask]
#         if any(np.isnan(x)) or any(x < 0):
#             return np.nan
#     except Exception as ex:
#         raise RuntimeError('error in ratio study: _gini_robust: {ex}') from ex

#     def _i_gini(x, rankvar, w=None, unbiased=False):
#         try:
#             if w is None:
#                 w = np.ones(len(x))
#             w = w / w.sum()
#             x = np.array(x)
#             rankvar = np.array(rankvar)
#             indices = np.argsort(rankvar)
#             x = x[indices]
#             w = w[indices]
#             f_hat = w / 2 + np.concatenate(([0], np.cumsum(w)[:-1]))
#             x_mean = np.average(x, weights=w)
#             res = 2 / x_mean * np.sum(w * (x - x_mean) * (f_hat - np.average(f_hat, weights=w)))
#             if unbiased:
#                 res *= 1 / (1 - np.sum(w ** 2))
#                 return res

#             if conf_level is None:
#                 res = _i_gini(x, rankvar, unbiased=unbiased)
#             else:
#                 try:
#                     from scipy.stats import bootstrap, ttest_1samp
#                 except ImportError:
#                     raise ImportError("The scipy library is required to use the bootstrapping option")
#                 pop_gini = _i_gini(x, rankvar, unbiased=unbiased)  # set population value for t_statistic & p-value
#                 samples = np.empty(n_samples)
#                 np.random.seed(random_state)
#                 for i in range(n_samples):
#                     sample = np.random.choice(range(len(x)), len(x), replace=True)
#                     x_sample = x[sample]
#                     rankvar_sample = rankvar[sample]
#                     samples[i] = _i_gini(x_sample, rankvar_sample, unbiased=unbiased)

#                 if samples_only:
#                     res = samples
#                 else:
#                     t_stat, p_value = ttest_1samp(samples, pop_gini)
#                     confint = bootstrap((samples,), np.median, n_resamples=n_samples, confidence_level=conf_level,
#                                         method=boot_method).confidence_interval
#                     res = {'population_idx': pop_gini, 'sampled_median_idx': np.median(samples),
#                            'sampled_mean_idx': np.mean(samples), 'sampled_lowci_idx': confint.low,
#                            'sampled_highci_idx': confint.high, 't_stat': t_stat, 'p_value': p_value}
#             return res
#         except Exception as ex:
#             raise RuntimeError('error in ratio study: _i_gini: {ex}') from ex
# ---------------------------------------------------------------------------------------------------------------------
