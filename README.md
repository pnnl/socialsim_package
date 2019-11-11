## Install Instructions
It is highly recommended that you install the SocialSim measurements package in a conda environment. SocialSim TA1 performers, please see the Wiki for information on how to [run evaluation as a self-contained service](https://wiki.socialsim.info/x/PADUAQ).

#### Step 1: Create and activate a conda environment
``` bash
conda create --name myenv python=3
source activate myenv 
```

#### Step 2: Install Snap and iGraph

Download the OS-specific distribution file here: http://snap.stanford.edu/snappy/release/beta/ 
``` bash
# Snap installation
tar -xvzf snap_distribution.tar.gz
cd snap_distribution.tar.gz
python setup.py install

# iGraph installation
# Note: The iGraph installation process may vary from system to system
conda install -c conda-forge/label/gcc7 python-igraph 

# rtree installation
conda install rtree

# pysal installation
conda install pysal

# For Mac OSX, you may to need install matplotlib with conda rather than 
# relying on the pip install below
conda install matplotlib
```

#### Step 3: Install the SocialSim package using pip
``` bash
python setup.py install 
```

You can ignore the following user warnings, if they appear:

``` 
UserWarning: You need pandana and urbanaccess to work with segregation's network module
You can install them with  `pip install urbanaccess pandana` or `conda install -c udst pandana urbanaccess`
  "You need pandana and urbanaccess to work with segregation's network module\n"

UserWarning: The `dill` module is required to use the sqlite backend fully.
  from .sqlite import head_to_sql, start_sql
  
UserWarning: SNAP import failed. Using igraph version of code instead.
  warnings.warn('SNAP import failed. Using igraph version of code instead.')

``` 

## Examples of the SocialSim Package API

```python
import socialsim as ss

# Load the simulation data
simulation = 'data/test_dataset.txt'
simulation = ss.load_data(simulation, ignore_first_line=True, verbose=False)

# Load the ground truth data
ground_truth = 'data/test_dataset.txt'
ground_truth = ss.load_data(ground_truth, ignore_first_line=True, verbose=False)

# Load the configuration file
# There are configuration files provided for CP1 and CP2 measurements 
config = 'data/cp2_configuration.json'
config = ss.load_config(config)

# Get metadata
metadata = ss.MetaData(community_directory='data/communities/',
                       node_file='data/node_list.txt')

# Instantiate the task runner 
task_runner = ss.TaskRunner(ground_truth, config, metadata=metadata, test=False)

# Run measurements and metrics on the simulation data
results, logs = task_runner(simulation, verbose=True)

# Get simulation measurements
results['simulation_results']
# Get ground truth measurements
results['ground_truth_results']
# Get metrics
results['metrics']
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
python data/build_node_list.py
```


_______________________________________________________________________________

## Social Network Representation Files

Dataset files are made of individual json dictionaries on each line with a
header json as the first entry. 

The minimal format is shown below. Some measurements may require additional 
fields.

```python
{"identifier": identifier, "team": team, "scenario": scenario}
{"nodeID": value, "nodeUserID": value, "actionType": value, "nodeTime": value, "platform": platform}
{"nodeID": value, "nodeUserID": value, "actionType": value, "nodeTime": value, "platform": platform}
{"nodeID": value, "nodeUserID": value, "actionType": value, "nodeTime": value, "platform": platform}
                .                                                               .
                .                                                               .
                .                                                               .
{"nodeID": value, "nodeUserID": value, "actionType": value, "nodeTime": value, "platform": platform}
{"nodeID": value, "nodeUserID": value, "actionType": value, "nodeTime": value, "platform": platform}
{"nodeID": value, "nodeUserID": value, "actionType": value, "nodeTime": value, "platform": platform}
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

_______________________________________________________________________________

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<p align="center">
PACIFIC NORTHWEST NATIONAL LABORATORY<br/>
<i>operated by<br/>
BATTELLE<br/>
<i>for the<br/>
UNITED STATES DEPARTMENT OF ENERGY<br/>
<i>under Contract DE-AC05-76RL01830
</p>

