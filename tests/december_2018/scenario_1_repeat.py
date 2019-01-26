import socialsim as ss

domain = 'cyber'

scenario_1_config = 'filepath'

configuration = load_configuration(scenario_1_config)
task_runner   = ss.Task_runner(configuration)
ground_truth  = ss.load_data()

for submission in submissions:    
    dataset = ss.load_data()
    results, logs = task_runner(dataset)







