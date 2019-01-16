import socialsim as ss

simulation   = ss.load_data('test_data/simulation.json')
ground_truth = ss.load_data('test_data/ground_truth.json')

print(simulation.head())

simulation   = simulation[simulation['Platform']=='twitter']
ground_truth = ground_truth[ground_truth['platform']=='twitter']

simulation_measurements   = CascadeMeasurements(simulation)
ground_truth_measurements = CascadeMeasurements(ground_truth)
