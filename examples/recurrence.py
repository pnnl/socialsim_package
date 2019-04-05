import socialsim as ss

# Load the example dataset
dataset = 'data/test_dataset.txt'
dataset = ss.load_data(dataset)

# Load the configuration file
config = 'data/recurrence_configuration.json'
config = ss.load_config(config)

# Subset the configuration for the given task 
config = config
config = config['recurrence']

# load metadata
# metadata = ss.MetaData(community_directory='data/communities', content_data=True) 

# Define the measurement object
# recurrence_measurements = ss.RecurrenceMeasurements(dataset, config['recurrence'], id_col='nodeID', userid_col='nodeUserID', timestamp_col='nodeTime', content_col='informationID')
recurrence_measurements = ss.RecurrenceMeasurements(dataset, config['recurrence'], id_col='nodeID', userid_col='nodeUserID', timestamp_col='nodeTime', content_col='platform', time_granularity='H')

# Run all measurements in the config file
results = recurrence_measurements.run(verbose=True)


