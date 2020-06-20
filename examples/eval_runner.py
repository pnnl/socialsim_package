import socialsim as ss

# Load the configuration file
config = 'data/cp4_configuration.json'
config = ss.load_config(config)

# Get metadata
metadata = ss.MetaData()

# Instantiate the task runner with the specified ground truth
ground_truth_filepath = 'data/test_dataset.json'
ground_truth = ss.load_data(ground_truth_filepath, ignore_first_line=True, verbose=False)
eval_runner = ss.EvaluationRunner(ground_truth, config, metadata=metadata)

# Evaluate a series of submissions that contain submission metadata as the first line of the submission file\
submission_filepaths = ['data/test_dataset.json']
for simulation_filepath in submission_filepaths:
    # Run measurements and metrics on the simulation data
    results, logs = eval_runner(simulation_filepath, verbose=True, submission_meta=True)
