import socialsim as ss

# Load the example dataset
dataset = 'data/test_dataset.txt'
dataset = ss.load_data(dataset)

# Load the configuration file
config = 'data/cp1_configuration.json'
config = ss.load_config(config)

# Subset the configuration for the given task 
config = config

# load metadata
metadata = ss.MetaData()

# Define the measurement object
recurrence_measurements = ss.RecurrenceMeasurements(dataset, config, 
    metadata=metadata)

# Run all measurements in the config file
results = recurrence_measurements.run(verbose=True)
