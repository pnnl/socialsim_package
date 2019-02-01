import socialsim as ss

user_metadata_location    = '/data/socialsim/december_2018/challenge/measurements_input_files/scenario1/crypto/github/repo_meta_data_dec18_challenge.csv' 
content_metadata_location = '/data/socialsim/december_2018/challenge/measurements_input_files/scenario1/crypto/github/user_meta_data_dec18_challenge.csv' 

metadata = ss.MetaData(user_metadata_location, content_metadata_location)

print('done.')
