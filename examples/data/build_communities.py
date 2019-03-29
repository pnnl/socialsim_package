import socialsim as ss
import pandas    as pd 

import random 

n = 50

dataset = 'test_dataset.txt'
dataset = ss.load_data(dataset)

dataset = dataset['informationID'].unique().tolist()

random.shuffle(dataset)

print(len(dataset))

for i in range(n):
    s = len(dataset) / n

    s = round(s)

    community = dataset[s*i:s*(i+1)]

    print(s*i, s*(i+1), len(dataset))

    filepath = 'communities/community_'+str(i)+'.txt'
    with open(filepath, 'a+') as f:
        for item in community:
            f.write(item+'\n')
