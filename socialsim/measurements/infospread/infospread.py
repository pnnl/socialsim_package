from .infospread_node       import InfospreadNode
from .infospread_population import InfospreadPopulation
from .infospread_community  import InfospreadCommunity

class InfospreadMeasurements(InfospreadNode, InfospreadPopulation,
    InfospreadCommunity):

    def __init__(self, dataset, configuration, metadata=None):
        """
        Description:

        Input:
            :dataset:
            :configuration:
            :metadata:

        Output:
            None
        """


        super(InfospreadNode, self).__init__(dataset, configuration, metadata)
        super(InfospreadPopulation, self).__init__(dataset, configuration,
            metadata)
        super(InfospreadCommunity, self).__init__(dataset, configuration,
            metadata)
