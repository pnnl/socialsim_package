import socialsim as ss
import json

dataset_directory  = '/data/socialsim/dataset.txt'
configuration_file = '/home/newz863/pic_drive/socialsim/repositories/stash/socialsim_package/tests/configuration_files/end_to_end.json'

ground_truth = ss.load_data(dataset_directory, verbose=False)
simulation   = ss.load_data(dataset_directory, verbose=False)

with open(configuration_file) as f:
    configuration = json.load(f)

task_runner = ss.TaskRunner(ground_truth, configuration, test=True)

print('|'*100)
print('|'*100)
print('|'*100)

results, logs = task_runner(simulation, verbose=True)


#print(results[0])
