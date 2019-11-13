import csv
import sys
import os
import json

import pandas as pd
import numpy as np
import warnings

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

    # casefold informationID columns
    dataset['informationID'] = dataset['informationID'].str.lower()
    community_dataset['informationID'] = community_dataset['informationID'].str.lower()

    dataset = dataset.merge(community_dataset, how='outer', on='informationID')
    dataset = dataset.dropna(subset=['actionType'])

    return dataset

def get_community_contentids(communities_directory: str) -> dict:
    '''
    Get a list of nodeIDs for all communities from the communities directory
    '''
    community_contentids = {}

    for community_fname in sorted(next(os.walk(communities_directory))[2]):
        with open(os.path.join(communities_directory, community_fname)) as fhandle:
            community_contentids[os.path.splitext(community_fname)[0]] = [x.strip() for x in fhandle.readlines()]

    return community_contentids


def gini(x):
    """
    Gini Coefficient calculated using the relative mean difference form.
    For more details, see: https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    :param x: list of frequencies per item in frequency distribution
    (can be list or array, will be converted to np.array format for calculation).

    e.g. number of times a user participated in a cascade per user where [1,1,1,1]
         would indicate 4 users who each participated once.

    :return: gini coefficient (float)
    """
    if len(x) == 0:
        warnings.warn('Cannot compute gini, no values passed (empty list)')
        return None
    x = np.asarray(sorted(x))
    i = np.arange(1, len(x)+1)
    n = len(x)
    gini = sum(((2 * i) - n - 1)*x) / float(n * sum(x))
    return gini


def palma_ratio(values):
    """
    Palma Ratio - Ratio of the frequency share of the 10% most active to 40% least active in the frequency distribution
    :param values: list of frequencies per item in frequency distribution
    (can be list or array, will be converted to np.array format for calculation).

    e.g. number of times a user participated in a cascade per user where [1,1,1,1]
         would indicate 4 users who each participated once.
    :return:
    """
    if len(values) == 0:
        warnings.warn('Cannot compute palma ratio, no values passed (empty list)')
        return None
    sorted_values = np.sort(np.array(values))
    percent_nodes = np.arange(1, len(sorted_values) + 1) / float(len(sorted_values))
    xvals = np.linspace(0, 1, 10)
    percent_nodes_interp = np.interp(xvals, percent_nodes, sorted_values)
    top_10_pct = float(percent_nodes_interp[-1])
    bottom_40_pct = float(np.sum(percent_nodes_interp[0:4]))
    try:
        palma_ratio = top_10_pct / bottom_40_pct
    except ZeroDivisionError:
        return None
    return palma_ratio
