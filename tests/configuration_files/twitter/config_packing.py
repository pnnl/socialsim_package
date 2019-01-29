import json
import os
import sys

"""
This is a one-off script for unpacking the twitter configuration files.
"""

def open_config(filename):

    with open(filename, 'r') as f:
        config = json.load(f)

    return config

node       = open_config('twitter_cascade_metrics_node.json')
community  = open_config('twitter_cascade_metrics_community.json')
population = open_config('twitter_cascade_metrics_population.json')

config = {'node':{}, 'community':{}, 'population':{}}

for item in node:
    config['node'].update(item)

for item in community:
    config['community'].update(item)

for item in population:
    config['population'].update(item)

for scale in config.keys():
    for measurement in config[scale].keys():
        print(scale, measurement)

with open('twitter_cascade_config.json', 'w+') as f:
    json.dump(config, f)




