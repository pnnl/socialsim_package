import socialsim as ss

# Load the example dataset
dataset = 'data/test_dataset.txt'
dataset = ss.load_data(dataset)

# Load the configuration file
config = 'data/cp3_s1_configuration.json'
config = ss.load_config(config)

# Subset the configuration for the given task 
config = config
config = config['multi_platform']['recurrence']

# load metadata
metadata = ss.MetaData() 

# Define the measurement object
recurrence_measurements = ss.RecurrenceMeasurements(dataset, 
    configuration=config, metadata=metadata, id_col='nodeID', 
    userid_col='nodeUserID', timestamp_col='nodeTime', 
    content_col='informationID', time_granularity='H')

# Run all measurements in the config file
results = recurrence_measurements.run(verbose=True)


