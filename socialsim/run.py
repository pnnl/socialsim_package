# External imports
import pandas as pd
import numpy  as np

from ast import literal_eval

# Internal imports
from .metrics import metrics

from .measurements import InfospreadMeasurements
from .measurements import CascadeMeasurements
from .measurements import NetworkMeasurements
from .measurements import GroupFormationMeasurements
from .measurements import CrossPlatformMeasurements

from .load   import load_measurements
from .record import RecordKeeper

class TaskRunner:
    def __init__(self, ground_truth, metadata, configuration):
        """
        Description: Initializes the TaskRunner object. Stores the metadata and
            ground_truth objects and defines all measurements and metrics
            specified by the configuration dictionary.

        Inputs:
            :ground_truth:
            :metadata:
            :configuration:
        Outputs:
            None
        """

        if ground_truth is str:
            self.ground_truth_results = _load_measurements(ground_truth)
        else:
            self.ground_truth  = self.run(ground_truth)

        self.ground_truth  = ground_truth
        self.metadata      = metadata
        self.configuration = configuration

    def __call__(self, dataset, measurements_subset=None, run_metrics=True):
        """
        Description: Allows the class to be called as a function. Takes in a
            dataset runs the specified measurements and metrics on the dataset
            given the ground truth and metadata that initialized the
            TaskRunner.

        Inputs:
            :dataset: (pd.DataFrame) The dataset to run the given measurements
                and metrics on.
            :configuration: (dict) The configuration information in the form of
                a nested dictionary.

        Outputs:
            :report: (dict) A summary of the run including all results and
                status on the success or failure of individual function calls.
        """

        return report

    def run(self):
        """
        Description: This function runs the measurements and metrics code at
            accross all measurement types. It does not deal with multiple
            platforms.

        """
        self.simulation_results, self.simulation_logs = run_measurements(dataset, configuration)

        # Get the ground truth measurement results
        ground_truth_results = self.ground_truth_results

        # Run metrics to compare simulation and ground truth results
        metrics, metric_logs = run_metrics(self.simulation_results, self.ground_truth_results, configuration)

        # Log results at the task level
        results = [simulation_results, ground_truth_results, metrics]
        logs    = [simulation_logs, ground_truth_logs, metrics_logs]

        return results, logs

def run_measurements(dataset, configuration):
    """
    Description: Takes in a dataset and a configuration file and runs the
        specified measurements.

    Input:

    Output:
    """
    measurements = {
        'infospread'      : InfospreadMeasurements,
        'cascade'         : CascadeMeasurements,
        'network'         : NetworkMeasurements,
        'group_formation' : GroupFormationMeasurements
    }

    results = {}
    logs    = {}

    # Loop over platforms
    for platform in configuration.keys():
        platform_results = {}
        platform_logs    = {}

        # Loop over measurement types
        for measurement_type in configuration[platform].keys():
            if measurement_type=='infospread':
                Measurement = InfospreadMeasurements
            elif measurement_type=='cascade':
                Measurement = CascadeMeasurements
            elif measurement_type=='network':
                Measurement = NetworkMeasurements
            elif measurement_type=='cross_platform':
                Measurement = CrossPlatformMeasurements

            # Get data and configuration subset
            configuration_subset = configuration[platform][measurement_type]
            dataset_subset = dataset[dataset['platform']==platform]

            # Instantiate measurement object
            measurement = Measurement(dataset_subset, configuration_subset)

            # Run the specified measurements
            results, logs = measurement.run()

            # Log the results at the measurement type level
            platform_results.update({measurement_type:results})
            platform_logs.update({measurement_type:logs})

        # Log the results at the platform level
        results.update({platform:platform_results})
        logs.update({platform:platform_logs})

    return results, logs

def run_metrics(simulation, ground_truth, configuration):
    """
    Description: Takes in simulation and ground truth measurement results and a
        configuration file and runs all the specified metrics on the
        measurements.

        TODO: Add error handling at each level of loop

    Input:
        :simulation_results:
        :ground_truth_results:
        :configuration:

    Output:
        :results:
        :logs:
    """

    metrics_object = ss.metrics(simulation, ground_truth, configuration)

    results, logs = metrics_object.run()

    return results, logs

# ALMOST CERTAINLY MOVING TO metrics.py
def _evaluate_metric_list(simulation_result, ground_truth_result, configuration):
    """
    Description:

    Input:

    Output:
    """

    # Loop over metrics
    for metric in configuration['metrics'].keys():

        # Get metric name and arguments
        metric_name = configuration['metrics'][metric]['metric']
        metric_args = configuration['metrics'][metric]['metric_args']

        metric_function = getattr(metrics, metric_name)

        try:
            result = metric_function(**metric_args)
        except Exception as error:
            result = function_name+' failed to run.'

        if timing:

    return result, log
