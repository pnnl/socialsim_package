import socialsim as ss

import json

dataset = ss.load_data('../test_data/twitter_data_sample.json')

with open('../configuration_files/twitter_infospread.json') as f:
    configuration = json.load(f)

metadata = None

print(configuration)

measurements = ss.InfospreadMeasurements(dataset, configuration, metadata, 'twitter')

results, logs = measurements.run(timing=True)


for scale in logs.keys():
    print(scale)
    for measurement in logs[scale].keys():
        print('    '+measurement+logs[scale][measurement]['status'])



print('-'*80)

print(results)
