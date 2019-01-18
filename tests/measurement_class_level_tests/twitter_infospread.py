import socialsim as ss

import json

dataset = ss.load_data('../test_data/simulation.json')
dataset = dataset[dataset['platform']=='twitter']

with open('../configuration_files/twitter_infospread.json') as f:
    configuration = json.load(f)

print(configuration)

measurements = ss.InfospreadMeasurements(dataset, configuration)

results, logs = measurements.run(timing=True)

print(results)
print(logs)
