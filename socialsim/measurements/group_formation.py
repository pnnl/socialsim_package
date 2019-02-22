from .measurements import MeasurementsBaseClass

class GroupFormationMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration,
        log_file='group_formation_measurements_log.txt'):
        super(GroupFormationMeasurements, self).__init__(dataset, configuration,
            log_file=log_file)
