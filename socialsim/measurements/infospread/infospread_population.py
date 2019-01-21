class InfospreadPopulation:
    def __init__(self):
        """
        Description: Intentially does nothing. Infospread parent classes are
            just containers for functions. That is all.

        Input:
            None

        Output:
            None
        """
        pass


    def getUserUniqueContent(self, selectedUsers=False, eventTypes=None,
        content_field="root"):
        """
        Description: This method returns the number of unique repos that a
            particular set of users contributed too

        Question #17

        Inputs:
            :selectedUsers: (bool or list) A list of users of interest or a
                boolean indicating whether to subset to the node-level
                measurement users.
            :eventTypes: (None or list) A list of event types to include in the
                data.
            :content_field: (str) Column which contains the content ID
                (e.g. nodeID, parentID, or rootID)
        Output:
            :data: A dataframe with the user id and the number of repos
                contributed to.
        """

        data = self.determineDf(selectedUsers, eventTypes)
        data = data.groupby('user')
        data = data[content_field].nunique().reset_index()

        data.columns = ['user','value']

        return data
