import socialsim as ss

# Load the example dataset
dataset = 'data/test_dataset.txt'
dataset = ss.load_data(dataset)

# Load the configuration file
config = 'data/cp3_s1_configuration.json'
config = ss.load_config(config)

# Subset the configuration for the given task 
config = config['multi_platform']['multi_platform']

# Get metadata
metadata = ss.MetaData()

# Define the measurement object
multi_platform_measurements = ss.MultiPlatformMeasurements(dataset, config, 
    metadata=metadata)

# Run all measurements in the config file
results = multi_platform_measurements.run(verbose=True)