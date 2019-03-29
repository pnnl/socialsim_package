## Install Instructions
It is highly recommended that you install the SocialSim measurements package in a conda environment.

##### Step 1: Create a conda environment with required python packages 
``` bash
conda create --name myenv --file requirements.txt python=3.6
```

##### Step 2: Install Snap

Download the OS-specific distribution file here: http://snap.stanford.edu/snappy/release/beta/ 
``` bash
tar -xvzf snap_distribution.tar.gz
cd snap_distribution.tar.gz
python setup.py install
```
##### Step 3: Install the SocialSim package using pip
``` bash
pip install socialsim-0.1.0.tar.gz
```
## Examples of the SocialSim Package API

```python
import socialsim as ss

# Load the simulation data
simulation = 'data/test_dataset.txt'
simulation = ss.load_data(simulation)

# Load the ground truth data
ground_truth = 'data/test_dataset.txt'
ground_truth = ss.load_data(ground_truth)

# Load the configuration file
# There are configuration files provided for CP1 and CP2 measurements 
config = 'data/cp2_configuration.json'
config = ss.load_config(config)

# Get metadata
metadata = ss.MetaData(community_directory='data/communities/',
		       node_file='data/node_lists.txt')

# Instantiate the task runner 
task_runner = ss.TaskRunner(ground_truth, config, metadata=metadata, test=True)

# Run measurements and metrics on the simulation data
results, logs = task_runner(simulation, verbose=True)

# Get simulation measurements
results["simulation_results"]
# Get ground truth measurements
results["ground_truth_results"]
# Get metrics
results["metrics"]
```

## Generating test data and example community and node files

There are several scripts to generate example test data and community and node files.  These files are meant to illustrate the format for the input data and do not contain realistic interaction structure.

To generate example input data:
``` bash
python data/generate_test_data.py
```

To generate example community files:
``` bash
python data/build_communities.py
```

To generate an examples node list file:
``` bash
python data/build_communities.py
```


_______________________________________________________________________________

## Social Network Representation Files

Dataset files are made of individual json dictionaries on each line with a
header json as the first entry. 

The minimal format is shown below. Some measurements may require additional 
fields.

```python
{'identifier': identifier, 'team': team, 'scenario': scenario}
{'nodeID': value, 'nodeUserID': value, 'actionType': value, 'nodeTime': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'actionType': value, 'nodeTime': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'actionType': value, 'nodeTime': value, 'platform': platform}
                .                                                               .
                .                                                               .
                .                                                               .
{'nodeID': value, 'nodeUserID': value, 'actionType': value, 'nodeTime': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'actionType': value, 'nodeTime': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'actionType': value, 'nodeTime': value, 'platform': platform}
```

## Notes to contributors

All functions should follow the format shown below:

```python
def f(x,y):
    """
    Description: A plain English description of what this function does and
                 how it fits into the global package structure.
    Inputs:
        :x: (int) Plain English description of x.
        :y: (int) Plain English description of y.
    Outputs:
        :z: (int) Plain English description of z.
    """
    z = x**2 + y**2

    return z
```

On line length: Lines have a hard limit of 80 characters.

On everything else: Try to follow PEP8 where possible. Always follow local
conventions.
