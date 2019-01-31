import csv
import json

def print_status(log):

    return None

def load_config(filepath):

    with open(filepath) as f:

    return config


def config_count(filepath, level):

    if level=='measurement':
        for platform in


    return count


def csv_to_json(csv_location, meta_data, output_location=None):
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
