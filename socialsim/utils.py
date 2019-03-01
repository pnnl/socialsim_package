import csv
import json

import pandas as pd

def subset_for_test(dataset, n=1000):
    platforms = dataset['platform'].unique()

    subsets = []
    for platform in platforms:
        subset = dataset[dataset['platform']==platform]
        subset = subset.head(n=n)
        subsets.append(subset)

    subset = pd.concat(subsets, axis=0)

    return subset