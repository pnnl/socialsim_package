import csv
import sys
import os
import json

import pandas as pd

def subset_for_test(dataset, n=1000):
    platforms = dataset['platform'].unique()

    subsets = []
    for platform in platforms:
        subset = dataset[dataset['platform']==platform]
        subset = subset.head(n=n)
        subsets.append(subset)

    if len(subsets) > 0:
        subset = pd.concat(subsets, axis=0)
    else:
        return dataset

    return subset

def add_communities_to_dataset(dataset, communities_directory, communities=None):
    """
    Description: Makes a new dataset with the community information integrated 
        into it.

    Input:
        :dataset:
        :communities_directory:

    Output:
        :communities_dataset:

    """

    community_dataset = []

    if communities is None:
        for community in os.listdir(communities_directory):
            community_file = communities_directory+community

            with open(community_file) as f:
                community_data = [line.rstrip() for line in f]

            community_data = pd.DataFrame(community_data, columns=['informationID'])
            community_data['community'] = community.split('.')[0].replace('community_','')
            community_data = community_data.drop_duplicates()
        
            community_dataset.append(community_data)
    else:
        for key,value in communities.items():
            community_data = pd.DataFrame(value,columns=['informationID'])
            community_data['community'] = key
            community_data = community_data.drop_duplicates()

            community_dataset.append(community_data)

    community_dataset = pd.concat(community_dataset)
    community_dataset = community_dataset.replace(r'\n','', regex=True) 
    
    dataset = dataset.merge(community_dataset, how='outer', on='informationID')
    dataset = dataset.dropna(subset=['actionType'])

    return dataset

def get_community_contentids(communities_directory: str) -> dict:
    '''Get a list of nodeIDs for all communities from the communities directory
    '''
    community_contentids = {}
    for community_fname in sorted(next(os.walk(communities_directory))[2]):
        with open(os.path.join(communities_directory, community_fname)) as fhandle:
            community_contentids[os.path.splitext(community_fname)[0]] = [x.strip() for x in fhandle.readlines()]
    return community_contentids

