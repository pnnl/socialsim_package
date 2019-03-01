import socialsim as ss

# Load the simulation data
simulation = ''
simulation = ss.load_data(dataset)

# Load the ground truth data
ground_truth = ''
ground_truth = ss.load_data(ground_truth)

# Instantiate the task runner 
task_runner = ss.TaskRunner(ground_truth)

# Run measurements and metrics on the simulation data
results = task_runner(simulation)
