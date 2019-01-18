import pandas as pd

import json
import csv
import os
import sys

def load_data(filepath, ignore_first_line=True, name_mappings=None, verbose=True):
    """
    Description:

    Input:
        :filepath: (str) The filepath to the submission file.
        :name_mappings: (dict) A dictionary where the keys are existing names
            and the values are new names to replace the existing names.

    Output:
        :dataset: (pandas dataframe) The loaded dataframe object.
    """

    filetype = filepath[-4:]

    if filetype=='.csv':
        if meta_data:
            raise ERROR

        dataset = _load_csv(filepath, ignore_first_line)

    elif filetype=='json':

        dataset = _load_json(filepath, ignore_first_line, verbose)

    dataset = convert_datetime(dataset, verbose)

    return dataset

def _load_csv(filepath, ignore_first_line):
    """
    Description: Loads a dataset from a csv file.

    Input:
        :filepath: (str) The filepath to the submission file. The submission
            file should have a header.
        :ignore_first_line: (bool) A True/False value. If True the first line
            is skipped.

    Output:
        :dataset: (pandas dataframe) The loaded dataframe object.
    """

    return dataset

def _load_json(filepath, ignore_first_line, verbose):
    """
    Description: Loads a dataset from a json file.

    Input:
        :filepath: (str) The filepath to the submission file.
        :ignore_first_line: (bool) A True/False value. If True the first line
            is skipped.

    Output:
        :dataset: (pandas dataframe) The loaded dataframe object.
    """

    dataset = []

    if verbose:
        print('Loading dataset at '+filepath)
        total_line_numbers = _count_number_of_lines(filepath)

    with open(filepath, 'r') as file:
        for line_number, line in enumerate(file):

            if line_number==0 and ignore_first_line:
                continue

            if verbose:
                print(line_number / total_line_numbers, end='\r')

            dataset.append(json.loads(line))

    if verbose:
        print(' '*100, end='\r')
        print(line_number / total_line_numbers)

    dataset = pd.DataFrame(dataset)

    return dataset

def validate_dataset(filepath):
    """
    Description: Checks a json submission file and for required fields.

    Input:
        :filepath: (str) The filepath to the submission file.

    Output:
        :check: (bool) A True/False value indicating the success or failure of
                the validation.
    """

    return check

def convert_datetime(dataset, verbose):
    """
    NOTE: THIS FUNCTION SHOULD BE OPTIMIZED WITH DASK

    Description:

    Input:

    Output:
    """

    if verbose:
        print('Converting strings to datetime objects...', end='', flush=True)

    dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'])

    if verbose:
        print('Done')

    return dataset

def _count_number_of_lines(filepath):
    count = -1
    for line in open(filepath):
        count += 1
    return count

    return count
