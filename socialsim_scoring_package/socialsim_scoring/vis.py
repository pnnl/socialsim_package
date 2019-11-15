from matplotlib.colors import ListedColormap, hex2color, rgb_to_hsv
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display

from .leaderboards import *



## ---------------------------------------------------------------------------------------------------------------------
## Visualizing Results
## ---------------------------------------------------------------------------------------------------------------------

def plot_leaderboard(stacked_data, listed_colors, *, xlabel='Points', ylabel='Team'):
    plt.figure(figsize=(8, 6.5))
    stacked_data.plot.barh(stacked=True,
                           colormap=ListedColormap(listed_colors),
                           ax=plt.subplot(111))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14,
               fancybox=False, shadow=False, frameon=False, ncol=1)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def show_leaderboard_summary_table(results, scenario, *, min_points=0, by='team_and_model', title=''):
    leaderboard_summary_table = get_leaderboard_summary_table(results, scenario, min_points=min_points,
                                                              title=title, by=by)
    display(leaderboard_summary_table)


def show_leaderboard_from_metrics(fmt_df, scenario, *, only_show_best_models=False,
                                        min_points=0, title='', y='team_and_model', show_leaderboard_summary_table=False):

    # subset to only data for specified scenario
    fmt_df = fmt_df[fmt_df['scenario']==str(scenario)].copy()
    # use 1/1+rank points scoring to get results for all models across all teams
    results = get_results(fmt_df)

    if only_show_best_models:
        # identify best model per team
        best_model_per_team = results.loc[str(scenario)].groupby('measurement_type') \
            .mean().sum() \
            .groupby(level=0, group_keys=False).apply(lambda x: x.idxmax()[1])
        best_models = best_model_per_team.to_dict()
        # subset results dataframe to only include best models per team
        results = results[list(best_models.items())]

    stacked, listed_values, listed_colors = stack_points(results, scenario, min_points)

    pltdat, ylabel = get_leaderboard_input(stacked, y)

    if show_leaderboard_summary_table:
        show_leaderboard_summary_table(results, scenario, min_points=0, title='', y='team_and_model')

    plot_leaderboard(pltdat, listed_colors, xlabel='Score (higher is better)', ylabel=ylabel)
    plt.title(f'Scenario {scenario} {title}', fontweight='bold', fontsize=24)


def show_leaderboard_from_results(results, scenario, *, min_points=0, title='', y='team_and_model',
                                  show_points_per_table=False):
    stacked, listed_values, listed_colors = stack_points(results, scenario, min_points)

    pltdat, ylabel = get_leaderboard_input(stacked, y)

    if show_points_per_table:
        show_leaderboard_summary_table(results, scenario, min_points=0, title='', y='team_and_model')

    plot_leaderboard(pltdat, listed_colors, xlabel='Score (higher is better)', ylabel=ylabel)
    plt.title(f'Scenario {scenario} {title}', fontweight='bold', fontsize=24)


def stringify_team_name(tup):
    if tup is np.NaN:
        return ''
    t, m = tup
    return t.upper()


def style_team_name(s):
    s = s.lower()
    c = TEAM_VIS_ATTRIBUTES[s]['color'] if s in TEAM_VIS_ATTRIBUTES else '#ffffff'
    hsv = rgb_to_hsv(hex2color(c))
    fc = 'black' if hsv[2] > .6 else 'white'

    return f'background-color: {c}; color: {fc}; text-align: center; height: 40px'


def show_best_teams_table(results):
    return get_best_teams_table(results) \
        .applymap(stringify_team_name) \
        .style.applymap(style_team_name)


def show_unique_best_models_across_measurement_types_heatmap(n_best_models, *, subtitle=''):
    sns.set_context("talk", font_scale=1.2)
    plt.figure(figsize=(20, 5))
    n_best_models['team'] = [x.upper() for x in n_best_models['team']]
    hm = pd.pivot_table(n_best_models, index='scenario', columns='team', values='n_models')
    sns.heatmap(hm, cmap='Reds', annot=True, cbar=False, vmax=10)
    plt.title(f'Unique Best Models Across Measurement Types {subtitle}')
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.xlabel('')
    plt.ylabel('Scenario')