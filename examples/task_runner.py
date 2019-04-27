from pprint import pprint
import socialsim as ss

# Load the simulation data
simulation = 'data/test_dataset.txt'
simulation = ss.load_data(simulation, ignore_first_line=True, verbose=False)

# Load the ground truth data
ground_truth = 'data/test_dataset.txt'
ground_truth = ss.load_data(ground_truth, ignore_first_line=True, verbose=False)

# Load the configuration file 
config = 'data/cp2_configuration.json'
config = ss.load_config(config)

# Get metadata
metadata = ss.MetaData(community_directory='data/communities/',
                       node_file='data/node_list.txt')

# Instantiate the task runner 
task_runner = ss.TaskRunner(ground_truth, config, metadata=metadata, test=True)

# Run measurements and metrics on the simulation data
results, logs = task_runner(simulation, verbose=True)

# Print metrics
pprint(results['metrics'])

