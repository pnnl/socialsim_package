import socialsim as ss

# Load the example dataset
dataset = 'example_dataset.txt'
dataset = ss.load_data(dataset)

# Subset the dataset to a particular platform 
dataset = dataset[dataset['platform']=='reddit'].head(n=2000)

# Load the configuration file
config = 'example_configuration.json'
config = ss.load_config(config)

# Subset the configuration for the given task 
config = config['reddit']['information_cascades']

# Define the measurement object
information_cascade_measurements = ss.InformationCascadeMeasurements(dataset, config, None, 'reddit')

# Run all measurements in the config file
results = information_cascade_measurements.run(verbose=True)