import socialsim as ss

# Load the example dataset
dataset = 'data/example_dataset.txt'
dataset = ss.load_data(dataset)

# Subset the dataset to a particular platform 
dataset = dataset[dataset['platform']=='github'].head(n=2000)

# Load the configuration file
config = 'example_configuration.json'
config = ss.load_config(config)

# Subset the configuration for the given task 
config = config['github']['social_activity']

# Define the measurement object
social_activity_measurements = ss.SocialActivityMeasurements(dataset, config, None, 'github')

# Run all measurements in the config file
results = social_activity_measurements.run(verbose=True)
