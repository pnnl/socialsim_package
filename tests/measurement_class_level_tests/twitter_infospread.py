import socialsim as ss

import json

dataset = ss.load_data('../test_data/simulation.json')
dataset = dataset[dataset['platform']=='twitter']

with open('../configuration_files/twitter_infospread.json') as f:
    configuration = json.load(f)

print(configuration)

metadata = None

measurements = ss.InfospreadMeasurements(dataset, configuration, metadata)

results, logs = measurements.run(timing=True)

print(results)
print(logs)

for scale in logs.keys():
    for measurement in logs[scale].keys():
        if 'Error' in logs[scale][measurement].keys():
            print(logs[scale][measurement]['Error'])
