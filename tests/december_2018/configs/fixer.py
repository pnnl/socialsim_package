import json
import os
import sys

def load_config(filepath):
    json_data = open(filepath).read()
    config    = json.loads(json_data)
    return config

def save_config(config, filepath):
    with open('new/'+filepath, 'w') as f:
        json.dump(config, f)
    return None

def delete_cascades(config):
    if 'cascade' in list(config.keys()):
        del config['cascade']
    return config

def add_network(config):
    network = load_config('network_metrics_config.json')
    config.update(network)
    return config


if __name__=='__main__':

    for filename in os.listdir():
        if filename not in ['new', 'fixer.py', '.fixer.py.swp', 'network_metrics_config.json']:

            print('-'*80)
            print(filename)

            platform = filename[2]

            config = load_config(filename)

            config = add_network(config)
            
            if platform=='g':
                config = delete_cascades(config)

            save_config(config, filename) 
