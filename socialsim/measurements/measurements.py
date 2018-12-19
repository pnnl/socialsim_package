class MeasurementsBaseclass:
    def __init__(self, dataset):
        """
        Description:

        Inputs:

        Outputs:

        """

        if not self.verify_dataset(dataset):
            raise Exception('Dataset failed verification.')

    def names(self):
        """
        Description:

        Inputs:

        Outputs:

        """

        raise NotImplementedError

    def verify_dataset(self, dataset):
        """
        Description: This verifies that a dataset has the required fields.

        Inputs:
            dataset: (pandas dataframe) The dataframe represenation of the
                     submission.

        Outputs:
            check: (bool) A True/False value indicating the success or failer
                   of the test.

        """

        check = True

        return check

class Measurement:
    def __init__(self):
        """
        Description:

        Inputs:

        Outputs:
        """

        self.scale = None
        self.name  = 'measurement base class'

    def __call__(self, *args, **kwargs):
        """
        Description: Allows the measurement class to be called like a function.
                     See self.run().

        Inputs:
            Same as the run function

        Outputs:
            Same as the run function
        """

        self.result = self.run(*args, **kwargs)

        return self.result

    def run(self, dataset):
        """
        Description: Runs the measurement on the given dataset.

        Inputs:
            :dataset: (pd.DataFrame) The dataset object.

        Outputs:
        """

        raise NotImplementedError
