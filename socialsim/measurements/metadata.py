import pandas as pd

class MetaData:
    def __init__(self, content_data=False, user_data=False):
        """
        Description:

        Input:

        Output:

        """
        if content_data!=False:
            self.use_content_data = True
            self.content_data = pd.read_csv(content_data)
        else:
            self.use_content_data = False

        if user_data != False:
            self.use_user_data = True
            self.user_data = pd.read_csv(user_data)
        else:
            self.use_user_data = False

    def build_communities(self, content_data, user_data):
        """
        Description:

        Input:

        Output:
        """

        return communities
