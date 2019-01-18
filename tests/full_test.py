import sys

import socialsim as ss

simulation   = ss.load_data('test_data/simulation.json')
ground_truth = ss.load_data('test_data/ground_truth.json')

print(simulation.head())

configuration = {}
metadata      = None

for platform in ['twitter','github','reddit']:

    print('\n\n\n')
    print('-'*80)
    print('Running tests on '+platform)
    print('-'*80)

    simulation_subset   = simulation[simulation['platform']==platform]
    ground_truth_subset = ground_truth[ground_truth['platform']==platform]

    print('Subsets assigned.')
    print('-'*80)

    simulation_measurements   = ss.CascadeMeasurements(simulation_subset, configuration)
    ground_truth_measurements = ss.CascadeMeasurements(ground_truth_subset, configuration)

    print('Cascade measurements initialized.')
    print('-'*80)

    """
    simulation_measurements   = ss.NetworkMeasurements(simulation_subset, configuration)
    ground_truth_measurements = ss.NetworkMeasurements(ground_truth_subset, configuration)

    print('Network measurements initialized.')
    print('-'*80)

    simulation_measurements   = ss.InfospreadMeasurements(simulation_subset, configuration, metadata)
    ground_truth_measurements = ss.InfospreadMeasurements(ground_truth_subset, configuration, metadata)
    """
