from ..measurements import MeasurementsBaseClass

class GroupFormationMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration,
        log_file='group_formation_measurements_log.txt'):
        super(CrossPlatformMeasurements, self).__init__(main_df, configuration,
            log_file=log_file)
