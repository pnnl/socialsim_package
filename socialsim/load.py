import pandas as pd

import json
import csv
import os
import sys

def load_config(filepath):
    """
    Description: Loads a json file. Really just a wrapper for the json package.

    Input:
        :filepath: (str) The filepath to the configuration file. 
    Output:
        :config: (dict) The configuration file loaded as a python dictionary. 

    """
    with open(filepath) as f:
        config = json.load(f)

    return config

def load_measurements(filepath):
    """
    Description:

    Input:
        :filepath:

    Output:
        :measurements:
    """

    measurements = None

    return results, logs

def load_data(filepath, ignore_first_line=True, name_mappings=None, verbose=True, short=False):
    """
    Description:

    Input:
        :filepath: (str) The filepath to the submission file.
        :name_mappings: (dict) A dictionary where the keys are existing names
            and the values are new names to replace the existing names.

    Output:
        :dataset: (pandas dataframe) The loaded dataframe object.
    """
    dataset = _load_json(filepath, ignore_first_line, verbose, short)

    dataset = convert_datetime(dataset, verbose)
    check   = validate_dataset(dataset, verbose)

    if check:
        return dataset
    else:
        return 'Dataset validation failed.'

def _load_json(filepath, ignore_first_line, verbose, short):
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

        if short:
            total_line_numbers = 1000
        else:
            total_line_numbers = _count_number_of_lines(filepath)

    with open(filepath, 'r') as file:
        for line_number, line in enumerate(file):

            if line_number==0 and ignore_first_line:
                continue

            if verbose:
                print(100.0*(line_number / total_line_numbers), end='\r')

            dataset.append(json.loads(line))

            if short:
                if len(dataset)==1000:
                    break

    if verbose:
        print(' '*100, end='\r')
        print(int(100.0*(line_number / total_line_numbers)))

    dataset = pd.DataFrame(dataset)

    return dataset

def validate_dataset(filepath, verbose):
    """
    Description: Checks a json submission file and for required fields.

        Note: placeholder function, currently no validation actions are taken.

    Input:
        :filepath: (str) The filepath to the submission file.

    Output:
        :check: (bool) A True/False value indicating the success or failure of
                the validation.
    """
    check = True

    return check

def convert_datetime(dataset, verbose):
    """
    Description:

    Input:

    Output:
    """

    if verbose:
        print('Converting strings to datetime objects...', end='', flush=True)


    try:
        dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'], unit='s')
    except:
        try:
            dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'], unit='ms')
        except:
            dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'])
        
    dataset['nodeTime'] = dataset['nodeTime'].dt.tz_localize(None)

    dataset['nodeTime'] = dataset['nodeTime'].dt.tz_localize(None)

    if verbose:
        print(' Done')

    return dataset


def _count_number_of_lines(filepath):
    count = -1
    for _ in open(filepath):
        count += 1
    return count
