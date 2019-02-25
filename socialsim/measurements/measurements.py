import pickle as pkl

import traceback

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

        self.measurement_type = 'baseclass'

        self.measurements  = []
        for scale in configuration.keys():
            for name in configuration[scale].keys():
                self.measurements.append(name)

    def run(self, measurements_subset=None, verbose=False, save=False, 
        save_directory='./', save_format='json'):
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
                    message = 'SOCIALSIM MEASUREMENTS | Running '
                    message = message+self.measurement_type+' '+scale+' '+name
                    message = message+'... '
                    print(message, end='', flush=True)

                result, log = self._evaluate_measurement(
                    self.configuration[scale][name], verbose)

                if verbose:
                    delta_time = log['run_time']
                    message = 'Done. ({0} seconds.)'.format(delta_time)
                    print(message, flush=True)

                if save:
                    filepath = save_directory+self.measurement_type
                    filepath = filepath+'_'+scale+'_'+name
                    self.save_measurement(result, filepath, save_format)

                scale_results.update({name:result})
                scale_logs.update({name:log})

            results.update({scale:scale_results})
            logs.update({scale:scale_logs})

        return results, logs

    def _evaluate_measurement(self, configuration, verbose):
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
        function_name = configuration['measurement']

        if 'measurement_args' in configuration.keys():
            function_arguments = configuration['measurement_args']
        else:
            function_arguments = {}

        # get the requested method from the instantiated measurement class
        try:
            function = getattr(self, function_name)
        except Exception as error:
            result = None
            log.update({'status' : 'failure'})
            log.update({'error'  : error})

            if verbose:
                print('')
                print('-'*80)
                trace = traceback.format_exc()
                print(trace)

            return result, log

        # Evaluate the function with the given arguments
        self.record_keeper.tic(1)

        try:
            result = function(**function_arguments)
            log.update({'status' : 'success'})

            self.record_keeper.update(function_name+' complete.')
        except Exception as error:
            result = None
            log.update({'status' : 'failure'})
            log.update({'error'  : error})

            if verbose:
                print('')
                print('-'*80)
                trace = traceback.format_exc()
                print(trace)

        delta_time = self.record_keeper.toc(1)
        log.update({'run_time': delta_time})

        return result, log

    def save_measurement(self, result, filepath, format='json'):
        """
        Description:

        Input:

        Output:
        """

        if format=='pickle':
            self._save_measurement_to_pickle(result, filepath)
        elif format=='json':
            self._save_measuremnt_to_json(result, filepath)

        return None

    def _save_measurement_to_pickle(self, result, filepath):
        """
        Description: Saves the result of a measurement to a pickle file.

        Input:
            :result: Type varies.
            :filepath: (str)

        Output:
            None
        """

        filepath = filepath+'.pkl'

        with open(filepath, 'wb') as f:
            pkl.dump(result, f)

        return None

    def _save_measuremnt_to_json(self, result, filepath):
        """
        Description:

        Input:
            :result:
            :filepath:

        Output:
            None
        """

        filepath = filepath+'.json'

        return None

    def _raw_to_json(self, result):
        """
        Description:

        Input:
            :result:

        Output:
            :result:
        """

        return result

    def _json_to_raw(self, result):
        """
        Description:

        Input:
            :result:

        Output:
            :result:
        """

        return result
