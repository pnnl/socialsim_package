import socialsim as ss

# Load metadata
metadata = ss.MetaData()

# Load dataset file
dataset = ''
dataset = ss.load_data()

# Load configuration file
configuration = ''
configuration = ss.load_configuration(configuration)

# instantiate cross platform measurements object 
cross_platform_measurements = ss.CrossPlatformMeasurements(dataset, metadata)

# Run all cross platform measurements 
results = cross_platform_measurements.run()