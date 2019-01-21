import sys

import socialsim as ss

ground_truth  = ss.load_data('test_data/twitter_data_sample.json')
configuration = {}
metadata      = None
platform      = 'twitter'

print('\n')
print('-'*80)
print('Running tests on '+platform)

simulation_subset = ground_truth

print('Subsets assigned.')
print('-'*80)

try:
    simulation_measurements = ss.CascadeMeasurements(simulation_subset, configuration)
    print('Cascade measurements initialized.')
except Exception as e:
    print(e)
    print('Cascade measurements failed')

try:
    simulation_measurements = ss.NetworkMeasurements(simulation_subset, configuration)
    print('Network measurements initialized.')
except Exception as e:
    print(e)
    print('Network measurements failed.')

try:
    simulation_measurements = ss.InfospreadMeasurements(simulation_subset, configuration, metadata, platform)
    print('Infospread measurements initialized.')
except Exception as e:
    print(e)
    print('Infospread measurements failed.')
