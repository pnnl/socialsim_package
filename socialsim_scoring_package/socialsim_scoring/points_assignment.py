import pandas as pd
import numpy as np
from .definitions import *

## ---------------------------------------------------------------------------------------------------------------------
## Points Assignment Strategies
## ---------------------------------------------------------------------------------------------------------------------


def assign_points(metric_series):
    """
    points assignment strategy used in cp3 (also used to identify best model in cp2)
    that assigns a point score of 1/(1+ rank) points to each rank
    """
    metric = metric_series.name[-1]
    ser = metric_series.copy().dropna() \
        .sort_values(ascending=metric not in BIGGER_IS_BETTER_METRICS)

    rank = (ser.diff().abs() > 0).cumsum()
    points = 1. / (1. + rank)

    return pd.Series(points, index=ser.index)


def assign_points_3_2_1(metric_series, *, max_points=3):
    """
    (deprecated) 3-2-1 points scoring assignment strategy (used in CP2) that assigns
    3 points to rank 1, 2 points to rank 2, 1 point to rank 3, and 0 points to rank 4 or higher
    """
    metric = metric_series.name[-1]
    ser = metric_series.copy().dropna() \
        .sort_values(ascending=metric not in BIGGER_IS_BETTER_METRICS)

    rank = (ser.diff().abs() > 0).cumsum()
    points = np.clip(max_points - rank, 0, max_points)

    return pd.Series(points, index=ser.index)


## ---------------------------------------------------------------------------------------------------------------------
## Results Calculations and Formatting
## ---------------------------------------------------------------------------------------------------------------------


def get_results(df):
    """
    Function to apply scoring to metrics results
    :param df: pandas dataframe representation of _metrics.json files
    :return: results dataframe after points assigned
    """
    df.scenario = df.scenario.astype(int).astype(str)
    df_metrics_best = df.groupby(NON_MEAS_BY_COLUMNS).mean() \
        .value.mean(level=NON_MEAS_BY_COLUMNS[:-1]) \
        .unstack(level=['team', 'identifier'])

    # apply 1/1+rank points to each rank
    df_points = df_metrics_best.apply(assign_points, axis=1).fillna(0)

    return df_points.mean(level=df_points.index.names[:-1]) \
        .groupby(level=['scenario', 'measurement_type']) \
        .apply(lambda x: x / float(len(x.groupby(['scale', 'measurement', 'platform']))) )


def proportion_possible_points_awarded_breakdown(df):
    """
    Function to provide a breakdown of the proportion (0 to 1) of possible points that were awarded to each model
    :param df: pandas dataframe representation of _metrics.json files

    returns dictionary where keys 'individual_measurements', 'scale_temporal_groups', 'measurement_type'
    """
    df.scenario = df.scenario.astype(int).astype(str)
    df_metrics_best = df.groupby(NON_MEAS_BY_COLUMNS).mean() \
        .value.mean(level=NON_MEAS_BY_COLUMNS[:-1]) \
        .unstack(level=['team', 'identifier'])

    # apply 1/1+rank points to each rank
    df_points = df_metrics_best.apply(assign_points, axis=1).fillna(0)

    df_points['weight_in_total_score'] = [1] * len(df_points)

    results = df_points.mean(level=df_points.index.names[:-1]) \
        .groupby(level=['scenario', 'measurement_type']) \
        .apply(lambda x: x / float(len(x.groupby(['scale', 'measurement', 'platform']))) )

    individual_measurements = results.copy()
    for c in individual_measurements.columns:
        if c == ('weight_in_total_score',''):
            continue
        else:
            individual_measurements[c] = individual_measurements[c]/individual_measurements[('weight_in_total_score','')]

    by_scale_temporal = results.copy().groupby(['scenario', 'measurement_type', 'scale', 'temporal']).sum()
    for c in by_scale_temporal.columns:
        if c == ('weight_in_total_score', ''):
            continue
        else:
            by_scale_temporal[c] = by_scale_temporal[c] / by_scale_temporal[('weight_in_total_score', '')]

    measurement_type = results.copy().groupby(['scenario', 'measurement_type']).sum()
    for c in measurement_type.columns:
        if c == ('weight_in_total_score', ''):
            continue
        else:
            measurement_type[c] = measurement_type[c] / measurement_type[('weight_in_total_score', '')]

    return {'individual_measurements': individual_measurements, 'scale_temporal_groups':by_scale_temporal,
            'measurement_type':measurement_type}



def percentage_possible_points_awarded_breakdown(df, *, scenario='1'):
    """
    Function to provide a breakdown of the percentage (0 to 100) of possible points that were awarded to each model
    :param df: pandas dataframe representation of _metrics.json files

    returns dictionary where keys 'individual_measurements', 'scale_temporal_groups', 'measurement_type'
    """
    df.scenario = df.scenario.astype(int).astype(str)
    df = df[df['scenario']==str(scenario)].copy()
    df_metrics_best = df.groupby(NON_MEAS_BY_COLUMNS).mean() \
        .value.mean(level=NON_MEAS_BY_COLUMNS[:-1]) \
        .unstack(level=['team', 'identifier'])

    # apply 1/1+rank points to each rank
    df_points = df_metrics_best.apply(assign_points, axis=1).fillna(0)

    df_points['weight_in_total_score'] = [1] * len(df_points)

    results = df_points.mean(level=df_points.index.names[:-1]) \
        .groupby(level=['scenario', 'measurement_type']) \
        .apply(lambda x: x / float(len(x.groupby(['scale', 'measurement', 'platform']))) )

    individual_measurements = results.copy()
    for c in individual_measurements.columns:
        if c == ('weight_in_total_score',''):
            continue
        else:
            individual_measurements[c] = 100 * individual_measurements[c]/individual_measurements[('weight_in_total_score','')]

    by_scale_temporal = results.copy().groupby(['scenario', 'measurement_type', 'scale', 'temporal']).sum()
    for c in by_scale_temporal.columns:
        if c == ('weight_in_total_score', ''):
            continue
        else:
            by_scale_temporal[c] = 100 * by_scale_temporal[c] / by_scale_temporal[('weight_in_total_score', '')]

    measurement_type = results.copy().groupby(['scenario', 'measurement_type']).sum()
    for c in measurement_type.columns:
        if c == ('weight_in_total_score', ''):
            continue
        else:
            measurement_type[c] = 100 * measurement_type[c] / measurement_type[('weight_in_total_score', '')]

    return {'individual_measurements': individual_measurements.drop(columns=[('weight_in_total_score','')]),
            'scale_temporal_groups':by_scale_temporal.drop(columns=[('weight_in_total_score','')]),
            'measurement_type':measurement_type.drop(columns=[('weight_in_total_score','')])}




def get_best_models_table(results):
    levels = ['scenario','measurement_type']
    return results.groupby(level=levels).apply(lambda x: x.sum().idxmax())


def get_n_best_models(results, *, teams=['usc', 'usf', 'uva', 'uiuc', 'ucf-garibay']):
    """
    Function measuring generalizability of best model(s) - i.e. for each team, how many of the team's models are
    needed to achieve "best performance" aross all measurement types
    :param results: metrics results (pandas dataframe that includes all teams)
    :param teams: list of teams to compute number of best models needed
    :return: pandas dataframe with columns: team, scenario, n_best models
    """
    scenarios = sorted(set(results.index.get_level_values('scenario')))
    team_list = []
    scenario_list = []
    n_models = []

    for t in teams:
        model_lists = []
        for s in scenarios:
            team_list.append(t)
            scenario_list.append(s)
            dat_to_use = results.loc[s, t]

            model_lists.extend(list(get_best_models_table(dat_to_use).values))
            n_models.append(get_best_models_table(dat_to_use).nunique())
        if len(scenarios) > 1:
            team_list.append(t)
            if len(scenarios) > 2:
                combined_s = 'All'
            else:
                combined_s = 'Both'
            scenario_list.append(combined_s)
            n_models.append(len(list(set(model_lists))))
    n_best_models = pd.DataFrame({'team': team_list, 'scenario': scenario_list, 'n_models': n_models})
    return n_best_models



def relative_performance_of_best_models(all_models_df, all_models_results):
    """

    :param all_models_df:
    :param all_models_results:
    :return:
    """
    ### calculate results for each best model by scenario
    best_models = {}
    res_per_best_model = {}
    scenarios = set(all_models_df['scenario'])
    for s in scenarios:
        best_model_per_team = all_models_results.loc[s].groupby('measurement_type') \
            .mean().sum() \
            .groupby(level=0, group_keys=False).apply(lambda x: x.idxmax()[1])

        best_models[s] = best_model_per_team.to_dict()

        all_models_df_s = all_models_df[all_models_df['scenario'] == s].copy()
        mask = all_models_df_s.identifier.apply(set(best_model_per_team).__contains__)

        results_per_best_model = get_results(all_models_df_s[mask])
        res_per_best_model[s] = results_per_best_model
    return res_per_best_model




def get_best_teams_table(results):
    levels = ['scale', 'temporal', 'measurement_type']
    return results.groupby(level=levels).apply(lambda x: x.sum().idxmax()) \
        .unstack() \
        .sort_index(axis=0) \
        .sort_index(axis=1)


def get_teams_best_models_for_scenario(scenario_results):
    levels = ['scale', 'temporal', 'measurement_type']
    best_model_table = scenario_results.groupby(level=levels).apply(lambda x: x.sum()).unstack().fillna('--')
    return best_model_table




