from .measurements import Measurements

from .user_centric_measurements      import UserCentricMeasurements
from .community_centric_measurements import CommunityCentricMeasurements
from .content_centric_measurements   import ContentCentricMeasurements
from .te_measurements                import TEMeasurements

class BaselineMeasurements(UserCentricMeasurements, ContentCentricMeasurements,
                           TEMeasurements, CommunityCentricMeasurements):
    def __init__(self, dataset, baseline_data='baseline_data/',
                 content_node_ids=[], user_node_ids=[], meta_content_data=False,
                 meta_user_data=False):
        """
        Description:

        Inputs:

        Outputs:

        """
        super(BaselineMeasurements, self).__init__()
