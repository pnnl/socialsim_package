import sys

import socialsim as ss

simulation   = ss.load_data('test_data/simulation.json')

configuration = {}
metadata      = None

for platform in ['twitter','github','reddit']:

    print('\n')
    print('-'*80)
    print('Running tests on '+platform)

    simulation_subset   = simulation[simulation['platform']==platform]

    print('Subsets assigned.')
    print('-'*80)

    simulation_measurements   = ss.CascadeMeasurements(simulation_subset,
                                                       configuration)

    print('Cascade measurements initialized.')

    simulation_measurements   = ss.NetworkMeasurements(simulation_subset,
                                                       configuration)

    print('Network measurements initialized.')

    simulation_measurements   = ss.InfospreadMeasurements(simulation_subset,
                                                          configuration,
                                                          metadata)

    print('Infospread measurements initialized.')
