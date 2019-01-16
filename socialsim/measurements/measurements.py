class MeasurementsBaseClass:
    def __init__(self, dataset, configuration, log_file='measurements_log.txt'):
        """
        Description:

        Input:
            :dataset:
            :configuration:

        Output:
        """

        self.dataset       = dataset
        self.configuration = configuration
        self.timer = RecordKeeper('measurements_log.txt')

    def run(self, measurements_subset=None, timing=False):
        """
        Description:

        Input:
            :measurements_subset:
            :timing:

        Output:
            :results:
            :logs:

        """
        results = {}
        logs    = {}

        for scale in configuration.keys():
            scale_results = {}
            scale_logs    = {}

            for name in configuration[scale].keys():
                result, log = _evaluate_measurement(configuration[scale][name],
                    timing)

                scale_results.update({'name':result})
                scale_logs.update({'name':log})

            results.update({scale:scale_results})
            logs.update({scale:scale_logs})

        return results, logs

    def _evaluate_measurement(self, configuration, stiming):
        """
        Description: Evaluates

        Input:
            :configuration:
            :timing:

        Output:
            :result:
            :log:
        """

        function_name      = configuration['measurement']
        function_arguments = configuration['measurement_args']

        self.timer.tic(1)

        try:
            result = self.measurements[function_name](**function_arguments)
        except:
            result = function_name+' failed to run.'

        delta_time = self.timer.toc(1)

        log = {'run_time': delta_time}

        return result, log
