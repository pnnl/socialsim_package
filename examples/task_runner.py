from pprint import pprint
import socialsim as ss

# Load the simulation data
simulation = 'data/test_dataset.json'
simulation = ss.load_data(simulation, ignore_first_line=True, verbose=False)

# Load the ground truth data
ground_truth = 'data/test_dataset.json'
ground_truth = ss.load_data(ground_truth, ignore_first_line=True, verbose=False)

# Load the configuration file 
config = 'examples/data/cp3_s1_configuration.json'
config = ss.load_config(config)

# Get metadata
metadata = ss.MetaData()

# Instantiate the task runner 
task_runner = ss.TaskRunner(ground_truth, config, metadata=metadata)

# Run measurements and metrics on the simulation data
results, logs = task_runner(simulation, verbose=True)

# Print metrics
pprint(results['metrics'])

