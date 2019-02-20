import socialsim as ss
import json

user_metadata_location    = '/data/socialsim/december_2018/challenge/measurements_input_files/scenario1/crypto/github/repo_meta_data_dec18_challenge.csv' 
content_metadata_location = '/data/socialsim/december_2018/challenge/measurements_input_files/scenario1/crypto/github/user_meta_data_dec18_challenge.csv' 

metadata = ss.MetaData(user_metadata_location, content_metadata_location)
dataset  = ss.load_data('/data/socialsim/december_2018/processed_challenge/ground_truth_data/scenario1/crypto/github/github_data_dec2018_crypto_20171201_to_20171214.json')

with open('december_2018/configs/new/1_github_crypto_config.json') as f:
    configuration = json.load(f)

configuration = configuration['github']['infospread']

measurement = ss.InfospreadMeasurements(dataset, configuration, metadata, 'github')

results, logs = measurement.run(timing=True, verbose=True, save=True, 
    save_directory='./output/', save_format='pickle')

for s in results.keys():
    for m in results[s].keys():
        print(type(results[s][m]))

print('done.')
