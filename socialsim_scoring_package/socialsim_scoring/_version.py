import json 

with open('socialsim_scoring/_version.json') as f:
    config = json.load(f)
    __version__ = config["version"]
