import socialsim as ss

import json

dataset = ss.load_data('../test_data/twitter_data_sample.json')

configuration_file = '../configuration_files/twitter/twitter_cascade_config.json'
with open(configuration_file) as f:
    configuration = json.load(f)

measurements  = ss.CascadeMeasurements(dataset, configuration)
results, logs = measurements.run(timing=True, verbose=True)

for scale in logs.keys():
    for measurement in logs[scale].keys():

        print(logs[scale][measurement])


for scale in results.keys():
    for measurement in results[scale].keys():
        print(type(results[scale][measurement]))
