import socialsim as ss
import pandas    as pd 

import random 

n = 25

dataset = 'test_dataset.txt'
dataset = ss.load_data(dataset)

dataset = dataset.drop_duplicates(subset='informationID')

sampled_nodes = dataset['informationID'].sample(n=50)

sampled_nodes.to_csv('node_list.txt',index=False)
