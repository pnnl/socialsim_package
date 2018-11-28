# External imports
import pandas as pd
import numpy  as np
import json
import csv
import os

from ast import literal_eval

# Internal imports

import socialsim.measurements

from .dataset import Dataset
from .record  import RecordKeeper

def load_data(json_file):
    """
    Description:

    Inputs:

    Outputs:
    """
    with open(json_file, 'r') as f:
        json_object = json.loads(f.read())


    data = json_object['data']

    meta_info = ['team', 'scenario', 'domain', 'platform']
    meta_info = {key:json_object[key] for key in meta_info}

    # This instantiates our dataset object
    dataset = Dataset(data, meta_info)

    dataset.sort_index(axis=1, inplace=True)
    dataset = dataset.replace('', np.nan)

    # This converts the column names to ascii
    mapping = {name:str(name) for name in dataset.columns.tolist()}
    dataset = dataset.rename(index=str, columns=mapping)

    # This converts the row names to ascii
    dataset = dataset.reset_index(drop=True)

    # This converts the cell values to ascii
    json_df = dataset.applymap(str)

    return dataset

def run_measurement(simulation, ground_truth, measurement_name,
                    save_to_file=True, output_directory=None):
    """
    Description:

    Inputs:

    Outputs:

    """
    return result

def run_all_measurements(simulation, ground_truth, save_to_file=True,
                         output_directory=None):
    """
    Description:

    Inputs:

    Outputs:

    """
    return results

def run_metric(simulation_measurement, ground_truth_measurement, metric_name,
               save_to_file=True, output_directory=None):
    """
    Description:

    Inputs:

    Outputs:

    """
    return result

def run_all_metrics(simulation_measurements, ground_truth_measurements,
                    save_to_file=True, output_directory=None):
    """
    Description:

    Inputs:

    Outputs:

    """
    return results

def csv_to_json(csv_location, team_name, scenario, domain, platform,
                output_location=None):
    """
    Description: This file takes in a simulation CVS file and outputs a
                 submission JSON file.

    An example of the output of this file is available to all teams on
    confluence

    Inputs:
        csv_location    (string) The location of the csv file to load.
        output_location (string) The location to output the JSON file
        team_name       (string) Your team name
        scenario        (string) The scenario this is simulating. Should be
                                   1 or 2.
        domain          (string) The domain this is simulating. Should be
                                   'crypto', 'cyber', or 'CVE'.
        platform        (string) The platform this is simulating. Should be
                                   'github', 'twitter', or 'reddit'.
    Output:
        JSON_file - saves a json file of the following form:
                {"team"     : team_name,
                 "scenario" : scenario,
                 "domain"   : domain,
                 "platform" : platform,
                 "data":[JSON_datapoint,
                         JSON_datapoint,
                            :
                            :
                            :
                         JSON_datapoint,
                         JSON_datapoint]
                }
    """

    if output_location==None:
        output_location=csv_location[:-3]+'json'

    dataset = []
    with open(csv_location, "rb") as f:
        reader = csv.reader(f, delimiter=',')
        for i,line in enumerate(reader):
            if i==0:
                labels = line
            else:
                datapoint={}
                for j in range(len(labels)):
                    if labels[j]!='nodeAttributes':
                        datapoint.update({labels[j]:line[j]})
                    else:
                        temp_dict=literal_eval(line[j])
                        datapoint.update(temp_dict)
                dataset.append(datapoint)

    submission = {'team'     : team_name,
                  'scenario' : scenario,
                  'domain'   : domain,
                  'platform' : platform,
                  'data'     : dataset}

    with open(output_location,'w') as f:
        json.dump(submission, f)
