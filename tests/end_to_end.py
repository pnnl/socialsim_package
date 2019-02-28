import socialsim as ss
import pandas as pd
import json
import sys

def subset_for_test(dataset, n=1000):
    platforms = dataset['platform'].unique()

    subsets = []
    for platform in platforms:
        subset = dataset[dataset['platform']==platform]
        subset = subset.head(n=n)
        subsets.append(subset)

    subset = pd.concat(subsets, axis=0)

    return subset

dataset_directory  = '/data/socialsim/dataset.txt'
configuration_file = '/home/newz863/pic_drive/socialsim/repositories/stash/'
configuration_file = configuration_file + 'socialsim_package/tests/'
configuration_file = configuration_file + 'configuration_files/end_to_end.json'

ground_truth = ss.load_data(dataset_directory, verbose=True)
ground_truth = subset_for_test(ground_truth)

simulation = ground_truth.copy()
#simulation = ss.load_data(dataset_directory, verbose=True)
#simulation = subset_for_test(simulation)

with open(configuration_file) as f:
    configuration = json.load(f)

task_runner = ss.TaskRunner(ground_truth, configuration, test=False)

print('|'*100)
print('|'*100)
print('|'*100)

results, logs = task_runner(simulation, verbose=True)

#print(results[0])
