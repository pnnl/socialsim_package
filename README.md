## Examples of the SocialSim Package API

```python
metadata = ss.load_metadata(path)

infospread_measurements    = ss.InfoSpreadMeasurements(dataset, metadata, configuration)
cascade_measurements       = ss.CascadeMeasurements(dataset, configuration)
network_measurements       = ss.NetworkMeasurements(dataset, configuration)
crossplatform_measurements = ss.CrossPlatformMeasurements(dataset, configuration)

measurements = ss.AllMeasurements(dataset, metadata, configuration)

measurements_output = measurements.run()











task_runner = ss.TaskRunner(ground_truth, metadata, configuration)

task_runner(dataset, configuration)
```


































#### Measurements

Run a single measurement:

```python
import socialsim as ss

dataset            = ss.load(directory)
measurement_object = ss.MeasurementObject(Dataset, config_file)
measurement        = measurement_object.run(measurement_name)
```

Run all measurements

```python
import socialsim as ss

dataset            = ss.load(directory)
measurement_object = ss.MeasurementObject(dataset, config_file)
measurements       = measurement_object.run()
```

Get a list of available measurements

```python
import socialsim as ss

dataset            = ss.load(directory)
measurement_object = ss.MeasurementObject(dataset)
measurement_names  = measurement_object.measurements()
```

#### Metrics

Run metrics

```python
import socialsim as ss

simulation_dataset            = ss.load(simulation_directory)
simulation_measurement_object = ss.MeasurementObject(simulation_dataset)
simulation_measurements       = measurement_object.run()

ground_truth_dataset            = ss.load(ground_truth_directory)
ground_truth_measurement_object = ss.MeasurementObject(ground_truth_dataset)
ground_truth_measurements       = ground_truth_measurement_object.run()

# run a single metric on a single measurement
metric  = run_metrics(simulation_measurement, ground_truth_measurement, measurement_name, metric_name)

# run a single metric on all the measurements
metric  = run_metrics(simulation_measurement, ground_truth_measurement, metric=metric_name)

# run all valid metrics on a single measurement
metric  = run_metrics(simulation_measurement, ground_truth_measurement, measurement=measurement_name)

# run all metrics (valid) on all measurements
run_metrics(simulation_measurement, ground_truth_measurement)
```

#### Analysis of results

Create measurement plots

```python
import socialsim as ss

ss.plot(measurements)
```

Create text file report for various outputs

```python
import socialsim as ss

ss.produce_report(measurements, metrics)
ss.produce_report(measurements)
ss.produce_report(metrics)
```
_______________________________________________________________________________

## Social Network Representation Files

Submission files are made of individual json dictionaries on each line with a
header json as the first entry. The format is as follows:

```python
{'identifier': identifier, 'team': team, 'scenario': scenario, 'domain': domain}
{'nodeID': value, 'nodeUserID': value, 'rootID': value, 'parentID': value, 'nodeTime': value, 'actionType': value, 'actionSubType': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'rootID': value, 'parentID': value, 'nodeTime': value, 'actionType': value, 'actionSubType': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'rootID': value, 'parentID': value, 'nodeTime': value, 'actionType': value, 'actionSubType': value, 'platform': platform}
                .                                                               .
                .                                                               .
                .                                                               .
{'nodeID': value, 'nodeUserID': value, 'rootID': value, 'parentID': value, 'nodeTime': value, 'actionType': value, 'actionSubType': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'rootID': value, 'parentID': value, 'nodeTime': value, 'actionType': value, 'actionSubType': value, 'platform': platform}
{'nodeID': value, 'nodeUserID': value, 'rootID': value, 'parentID': value, 'nodeTime': value, 'actionType': value, 'actionSubType': value, 'platform': platform}
```

## Function Documentation

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

Measurements list

    Baseline Measurements
        Node
        Community
        Population

    Network Measurements
        Population

    Cascade Measurements
        Node
        Community
        Population

    Cross-Platform Measurements
        Node
        Community
        Population
