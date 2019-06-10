import socialsim as ss

# Load the example dataset
# dataset = 'data/test_dataset.txt'
dataset = '/media/sf_Jun19-train/Jun19-train_Coins_.json'
dataset = ss.load_data(dataset)

# Load the configuration file
config = 'data/persistent_groups_configuration.json'
config = ss.load_config(config)

# Subset the configuration for the given task
config = config['multi_platform']['persistent_groups']

# Get metadata
metadata = ss.MetaData(community_directory='data/communities/')

# Define the measurement object
persistent_measurements = ss.PersistentGroupsMeasurements(dataset, config, metadata=metadata, plot=True,
                                                          save_groups=True)

# Run all measurements in the config file
results = persistent_measurements.run(verbose=True)