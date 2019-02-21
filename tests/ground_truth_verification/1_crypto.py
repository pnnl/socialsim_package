import socialsim as ss

import json
import os

for platform in ['twitter', 'reddit', 'github']:
    config_file       = '../december_2018/configs/new/'+'1_'+platform+'_crypto_config.json'
    output_directory  = 'output/'+platform+'/'
    dataset_directory = '/data/socialsim/december_2018/processed_challenge/ground_truth_data/scenario1/crypto/'+platform+'/'
    
    for filename in os.listdir(dataset_directory):
        dataset_location = dataset_directory+filename

    dataset = ss.load_data(dataset_location)

    with open(config_file) as f:
        configuration = json.load(f)

    timing=True
    verbose=True
    save=True
    save_directory=output_directory
    save_format='pickle'

    results, logs = ss.run_measurements(dataset, configuration, timing, verbose, save, save_directory, save_format)
