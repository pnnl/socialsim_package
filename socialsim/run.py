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
        pass

    def __call__(self, dataset, configuration):
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

    def _run_measurements_and_metrics(self, dataset, configuration):



        return results, logs
