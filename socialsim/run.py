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

from .load import load_measurements

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
            self.ground_truth = _load_measurements(ground_truth)
        else:
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

    def run(self, dataset, configuration):
        """
        Description: This function runs the measurements and metrics code at
            accross all measurement types. It does not deal with multiple
            platforms.

        """
        measurements = {
            'infospread'      : InfospreadMeasurements,
            'cascade'         : CascadeMeasurements,
            'network'         : NetworkMeasurements,
            'group_formation' : GroupFormationMeasurements
        }

        simulation_results = {}
        simulation_logs    = {}

        for platform in configuration.keys():
            platform_results = {}
            platform_logs    = {}

            for measurement_type in configuration[platform].keys():
                if measurement_type=='infospread':
                    Measurement = InfospreadMeasurements
                elif measurement_type=='cascade':
                    Measurement = CascadeMeasurements
                elif measurement_type=='network':
                    Measurement = NetworkMeasurements
                elif measurement_type=='cross_platform':
                    Measurement = CrossPlatformMeasurements

                configuration_subset = configuration[platform][measurement_type]
                dataset_subset = dataset[dataset['platform']==platform]
                measurement = Measurement(dataset_subset, configuration_subset)

                results, logs = measurement.run()

                platform_results.update({measurement_type:results})
                platform_logs.update({measurement_type:logs})

            simulation_results.update({platform:platform_results})
            simulation_logs.update({platform:platform_logs})

        ground_truth_results = self.ground_truth_results

        metrics, metric_logs = run_metrics(simulation_results,
            ground_truth_results, configuration)

        results = [simulation_results, ground_truth_results, metrics]
        logs    = [simulation_logs, ground_truth_logs, metrics_logs]

        return results, logs

def _run_metrics(simulation_results, ground_truth_results, configuration):
    """
    Description: Takes in simulation and ground truth measurement results and a
        configuration file and runs all the specified metrics on the
        measurements.

    Input:
        :simulation_results:
        :ground_truth_results:
        :configuration:

    Output:
        :results
    """

    return results, logs

def _run_metric(simulation_result, ground_truth_result, configuration):
    """
    Description:

    Input:

    Output:
    """

    return result, log
