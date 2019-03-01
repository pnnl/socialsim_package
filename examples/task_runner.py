import socialsim as ss

# Load the simulation data
simulation = 'data/debug_dataset.txt'
simulation = ss.load_data(simulation)

# Load the ground truth data
ground_truth = 'data/debug_dataset.txt'
ground_truth = ss.load_data(ground_truth)

# Load the configuration file 
config = 'data/example_configuration.json'
config = ss.load_config(config)

# Instantiate the task runner 
task_runner = ss.TaskRunner(ground_truth, config, test=True)

# Run measurements and metrics on the simulation data
results = task_runner(simulation, verbose=True)
