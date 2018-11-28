import sys
import os.path

socialsim_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              os.path.pardir))

sys.path.append(socialsim_path)

import socialsim as ss

data_directory = 'test_datasets/'

data_file = 'reddit/reddit_data_cyber_20180101_to_20180107_1week.json'

dataset = ss.load_data(data_directory+data_file)

print(dataset.head())
print(dataset.meta_info)

simulation_measurements   = ss.BaselineMeasurements(dataset)

sys.exit()

ground_truth_measurements = ss.BaselineMeasurements(dataset)

simulation_results   = ss.run_all_measurements(simulation_measurements)
ground_truth_results = ss.run_all_measurements(ground_truth_measurements)

final_results = ss.run_all_metrics(simulation_results, ground_truth_results)
