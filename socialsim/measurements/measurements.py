from ..record import RecordKeeper

class MeasurementsBaseClass:
    def __init__(self, dataset, configuration, log_file='measurements_log.txt'):
        """
        Description: Base class for all measurements classes. Contains methods
            necessary for running all measurements.

        Input:
            :dataset: (pd.DataFrame) The dataset to run measurements on.
            :configuration: (dict) A configuration dictionary specifying what
                measurements to run and what arguments to use.

        Output:
        """
        self.dataset       = dataset
        self.configuration = configuration
        self.record_keeper = RecordKeeper('measurements_log.txt')

        self.measurements  = []
        for scale in configuration.keys():
            for name in configuration[scale].keys():
                self.measurements.append(name)

    def run(self, measurements_subset=None, timing=False, verbose=False):
        """
        Description: Runs a measurement or a set of measurements on the given
            dataset.

        Input:
            :measurements_subset: (list) A list of strings specifying which
                measurements to run.
            :timing: (bool) If true then timing is run and included in the log
                output.

        Output:
            :result: The output of the measurement functions.
            :log: The log indicating the status of results and timing.

        """
        results = {}
        logs    = {}

        for scale in self.configuration.keys():
            scale_results = {}
            scale_logs    = {}

            for name in self.configuration[scale].keys():
                if verbose:
                    print('SOCIALSIM MEASUREMENTS | Running '+scale+' '+name)
        


                result, log = self._evaluate_measurement(
                    self.configuration[scale][name], timing)

                scale_results.update({name:result})
                scale_logs.update({name:log})

            results.update({scale:scale_results})
            logs.update({scale:scale_logs})

        return results, logs

    def _evaluate_measurement(self, configuration, timing):
        """
        Description: Evaluates a single measurement given the configuration
            information.

        Input:
            :configuration: (dict) Contains the measurement name and arguments
                to be used when running the measurement.
            :timing: (bool) If true then timing is run and included in the log
                output.

        Output:
            :result: The output of the measurement function.
            :log: The log indicating the status of the result and timing.
        """
        log = {}

        # unpack the configuration dictionary
        function_name      = configuration['measurement']
        function_arguments = configuration['measurement_args']

        # get the requested method from the instantiated measurement class
        function = getattr(self, function_name)

        # Evaluate the function with the given arguments
        if timing:
            self.record_keeper.tic(1)

        try:
            result = function(**function_arguments)
            log.update({'status' : 'success'})

            self.record_keeper.update(function_name+' complete.')
        except Exception as error:
            result = function_name+' failed to run.'
            log.update({'status' : 'failure'})
            log.update({'error'  : error})

            self.record_keeper.update(function_name+' exited with error: '+error)

        if timing:
            delta_time = self.record_keeper.toc(1)
            log.update({'run_time': delta_time})

        return result, log
