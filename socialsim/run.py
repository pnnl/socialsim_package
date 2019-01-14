# External imports
import pandas as pd
import numpy  as np

from ast import literal_eval

# Internal imports
from .record import RecordKeeper

class TaskRunner:
    def __init__(self, ground_truth, metadata, configuration):
        pass

    def __call__(self, dataset, configuration):
        """
        Description: Allows the class to be called as a function. Takes in a
            dataset runs the specified measurements and metrics on the dataset
            given the ground truth and metadata that initialized the
            TaskRunner.

        Inputs:
            :dataset:
            :configuration:

        Outputs:
            :report:
        """
        return report
