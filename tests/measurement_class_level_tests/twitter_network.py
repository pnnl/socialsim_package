import socialsim as ss

import json

dataset = ss.load_data('../test_data/twitter_data_sample.json')

with open('../configuration_files/twitter_network.json') as f:
    configuration = json.load(f)

measurements = ss.NetworkMeasurements(dataset, configuration)

results, logs = measurements.run(timing=True)

print(results)
print(logs)
