from .measurements import MeasurementsBaseClass

class PersistentGroupsMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration,
        log_file='group_formation_measurements_log.txt'):
        super(PersistentGroupsMeasurements, self).__init__(dataset, 
            configuration, log_file=log_file)
