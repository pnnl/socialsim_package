import socialsim as ss

import numpy as np

import json

dataset = ss.load_data('../test_data/twitter_data_sample.json')

configuration_file = '../configuration_files/twitter/twitter_cascade_config.json'

with open(configuration_file) as f:
    configuration = json.load(f)

metadata=None
platform='twitter'

measurements  = ss.CascadeMeasurements(dataset, configuration, metadata, platform)
results, logs = measurements.run(timing=True, verbose=True)

for scale in logs.keys():
    for measurement in logs[scale].keys():

        print(logs[scale][measurement])


for scale in results.keys():
    for measurement in results[scale].keys():

        if type(results[scale][measurement]) is np.float64:
            print(results[scale][measurement])
        elif type(results[scale][measurement]) is float:
            print(results[scale][measurement])
        else:
            print(type(results[scale][measurement]))


