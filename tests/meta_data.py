import socialsim as ss
import json

user_metadata_location    = '/data/socialsim/december_2018/challenge/measurements_input_files/scenario1/crypto/github/repo_meta_data_dec18_challenge.csv' 
content_metadata_location = '/data/socialsim/december_2018/challenge/measurements_input_files/scenario1/crypto/github/user_meta_data_dec18_challenge.csv' 

metadata = ss.MetaData(user_metadata_location, content_metadata_location)

dataset  = ss.load_data('/data/socialsim/december_2018/processed_challenge/ground_truth_data/scenario1/crypto/github/github_data_dec2018_crypto_20171201_to_20171214.json')

with open('december_2018/configs/new/1_github_crypto_config.json') as f:
    configuration = json.load(f)

measurement = ss.InfospreadMeasurements(dataset, configuration, metadata, 'github')

print('done.')
