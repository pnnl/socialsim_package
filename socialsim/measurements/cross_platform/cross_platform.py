from ..measurements import MeasurementsBaseClass

class CrossPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration,
        log_file='cross_platform_measurements_log.txt'):
        super(GroupFormationMeasurements, self).__init__(main_df, configuration,
            log_file=log_file)
