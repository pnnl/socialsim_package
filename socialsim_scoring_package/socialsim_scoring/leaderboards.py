from .load import *
from .points_assignment import *


import warnings
from matplotlib.colors import rgb2hex


def stack_points(results, scenario, min_points):
    def rgb(*args):
        return rgb2hex(np.array(args) / 255)

    meas_colors = {
        'cross-platform_spread': rgb(230, 25, 75),
        'scale_of_spread': rgb(255, 159, 0),
        'structure_of_spread': rgb(244, 9, 255),
        'social_network_of_spread': rgb(57, 145, 247)
    }

    stacked = results.loc[scenario].sum(level='measurement_type').T
    total = stacked.sum(axis=1).sort_values()
    order = total[total > min_points].index

    listed_values = stacked.columns.values
    listed_colors = [meas_colors[v]
                     for v in listed_values]

    return stacked.loc[order], listed_values, listed_colors


def get_leaderboard_input(stacked, y):
    if y == 'team_and_model':
        pltdat = stacked
        ylabel = 'Team, Model Identifier'
    elif y == 'team':
        pltdat = stacked.reset_index()
        pltdat = pltdat.drop(columns=['identifier'])
        pltdat['team'] = [x.upper() for x in pltdat['team']]
        pltdat = pltdat.set_index('team')
        ylabel = 'Team'
    elif y == 'model':
        pltdat = stacked.reset_index()
        pltdat = pltdat.drop(columns=['team'])
        pltdat = pltdat.set_index('identifier')
        ylabel = 'Model Identifier'

    return pltdat, ylabel


def get_leaderboard_summary_table(results, scenario, *, min_points=0, by='team_and_model'):
    stacked, listed_values, listed_colors = stack_points(results, scenario, min_points)

    pltdat, ylabel = get_leaderboard_input(stacked, by)

    meas_cols = [x for x in pltdat.columns if x not in ['measurement_type']]
    pltdat2 = pltdat.copy()
    pltdat2['total'] = 0
    for c in meas_cols:
        pltdat2['total'] = pltdat2['total'] + pltdat2[c]
    return pltdat2.sort_values(by='total', ascending=False)


def leaderboard_table_from_metrics(fmt_df, *, scenario=None):
    if scenario is None:
        warnings.warn("You need to specify a scenario.")
        return None
    if scenario not in fmt_df['scenario'].unique():
        warnings.warn(f"There is no data for scenario {scenario} in the dataframe passed")
        return None

    # subset to only data for specified scenario
    fmt_df = fmt_df[fmt_df['scenario'] == str(scenario)].copy()
    # use 1/1+rank points scoring to get results for all models across all teams
    results = get_results(fmt_df)
    # generate leaderboard summary table for all models across all teams
    leaderboard_table = get_leaderboard_summary_table(results, str(scenario))
    return leaderboard_table

def best_models_leaderboard_table_from_metrics(fmt_df, scenario=None):
    if scenario is None:
        warnings.warn("You need to specify a scenario.")
        return None
    if scenario not in fmt_df['scenario'].unique():
        warnings.warn(f"There is no data for scenario {scenario} in the dataframe passed")
        return None

    # subset to only data for specified scenario
    fmt_df = fmt_df[fmt_df['scenario'] == str(scenario)].copy()
    # use 1/1+rank points scoring to get results for all models across all teams
    results = get_results(fmt_df)
    # identify best model per team
    best_model_per_team = results.loc[str(scenario)].groupby('measurement_type') \
        .mean().sum() \
        .groupby(level=0, group_keys=False).apply(lambda x: x.idxmax()[1])
    best_models = best_model_per_team.to_dict()
    # subset results dataframe to only include best models per team
    best_model_results = results[list(best_models.items())].copy()
    # generate leaderboard summary table for best models
    leaderboard_table = get_leaderboard_summary_table(best_model_results, str(scenario))
    return leaderboard_table



def add_metrics_for_scenario(breakdown, scenario, metrics):
    metrics[scenario] = dict()

    indiv = breakdown['individual_measurements']
    indiv = indiv.reset_index().melt(id_vars=list(indiv.index.names))
    scale_temporal = breakdown['scale_temporal_groups']
    scale_temporal = scale_temporal.reset_index().melt(id_vars=list(scale_temporal.index.names))
    indiv = indiv[indiv['team'] != 'weight_in_total_score']
    scale_temporal = scale_temporal[scale_temporal['team'] != 'weight_in_total_score']

    categories = scale_temporal[['measurement_type', 'scale', 'temporal']].drop_duplicates()
    for _, cat_row in categories.iterrows():
        meas_type, scale, temporal = cat_row['measurement_type'], cat_row['scale'], cat_row['temporal']

        scale_temporal_dat = scale_temporal[(scale_temporal.measurement_type == meas_type) &
                                            (scale_temporal.scale == scale) &
                                            (scale_temporal.temporal == temporal) &
                                            (scale_temporal.scenario == scenario)]

        category_measurements = indiv[(indiv.measurement_type == meas_type) &
                                      (indiv.scale == scale) & (indiv.temporal == temporal) &
                                      (indiv.scenario == scenario)]
        if not meas_type in metrics[scenario]:
            metrics[scenario][meas_type] = dict()
        if not scale in metrics[scenario][meas_type]:
            metrics[scenario][meas_type][scale] = dict()
        if not temporal in metrics[scenario][meas_type][scale]:
            metrics[scenario][meas_type][scale][temporal] = list()
        for _, team_row in scale_temporal_dat.iterrows():
            team = team_row['team']
            if team != 'weight_in_total_score':
                model_identifier = team_row['identifier']
                cat_total_value = team_row['value']

                team_model_cat_measurements = category_measurements[(category_measurements.team == team) &
                                                                    (
                                                                                category_measurements.identifier == model_identifier)]
                measurements = team_model_cat_measurements[['measurement', 'value']].to_dict(orient='records')
                type_scale_temporal_data = {'team': team, 'identifier': model_identifier,
                                            'scale': scale, 'temporal': temporal, 'measurement_type': meas_type,
                                            'value': cat_total_value, 'scenario': scenario, 'measurements': measurements}
                metrics[scenario][meas_type][scale][temporal].append(type_scale_temporal_data)

    return metrics