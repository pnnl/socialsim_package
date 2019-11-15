import pandas as pd


BIGGER_IS_BETTER_METRICS = ['r2', 'rbo', 'spearman']

NON_MEAS_BY_COLUMNS = [
    'scenario',
    'measurement_type',
    'platform',
    'scale',
    'temporal',
    'measurement',
    'metric',
    'team',
    'identifier',
    'node',
]

S1_MEASUREMENT_CATEGORIES = pd.read_csv('socialsim_scoring/cp3_s1_measurement_categories.csv')
S2_MEASUREMENT_CATEGORIES = pd.read_csv('socialsim_scoring/cp3_s2_measurement_categories.csv')

# scale_of_spread, structure_of_spread, social_network_of_spread, cross_platform_spread
meas_type_to_class = {'multi_platform':'scale_of_spread',
                      'evolution':'social_network_of_spread',
                      'cross_platform':'cross-platform_spread',
                      'social_structure':'social_network_of_spread',
                      'recurrence':'structure_of_spread',
                      'persistent_groups':'structure_of_spread',
                      'information_cascades':'structure_of_spread'
                      }



TEAM_VIS_ATTRIBUTES = {
    'ucf-garibay': {
        'color': '#9730F2',
        'name': 'UCF - Garibay',
    },
    'uiuc': {
        'color': '#FF452F',
        'name': 'UIUC',
    },
    'usc': {
        'color': '#F8E71C',
        'name': 'USC',
    },
    'usf': {
        'color': '#0040FF',
        'name': 'USF',
    },
    'uva': {
        'color': '#096001',
        'name': 'UVA',
    },
}
