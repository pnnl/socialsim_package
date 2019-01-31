import socialsim as ss
import json

dataset = ss.load_data('../test_data/twitter_data_sample.json')

with open('../configuration_files/twitter_network.json') as f:
    configuration = json.load(f)

metadata = None

measurements = ss.NetworkMeasurements(dataset, configuration, metadata, 'twitter')

results, logs = measurements.run(timing=True, verbose=True, save=True,
    save_directory='./output/', save_format='pickle')
