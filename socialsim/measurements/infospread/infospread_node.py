import pandas as pd

from ..measurements import MeasurementsBaseClass

class InfospreadNode:
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

    def determineDf(self, users, eventTypes):
        """
        This function selects a subset of the full data set for a selected set of users and event types.
        Inputs: users - A boolean or a list of users.  If it is list of user ids (login_h) the data frame is subset on only this list of users.
                        If it is True, then the pre-selected node-level subset is used.  If False, then all users are included.
                eventTypes - A list of event types to include in the data set

        Output: A data frame with only the selected users and event types.
        """
        if users==True:
            df = self.selectedUsers
        elif users!=False:
            df = df[df.user.isin(users)]
        else:
            df = self.main_df

        if eventTypes!=None:
            df = df[df.event.isin(eventTypes)]

        return df

    def getUserActivityTimeline(self, selectedUsers=True, time_bin='1d',
        cumSum=False, eventTypes=None):
        """
        This method returns the timeline of activity of the desired user over
        time, either in raw or cumulative counts.

        Question #19

        Inputs:
            :selectedUsers: (list or bool) A list of users of interest or a
                boolean indicating whether to subset to node-level measurement
                users.
            :time_bin: Time frequency for calculating event counts
            :cumSum: (bool) Boolean indicating whether to calculate the
                cumulative activity counts
            :eventTypes: (list) List of event types to include in the data
        Output:
            :measurements: A dictionary with a data frame for each user with
                two columns: data and event counts
        """

        df = self.determineDf(selectedUsers, eventTypes)

        df['value'] = 1

        if cumSum:
            df['cumsum'] = df.groupby('user').value.transform(pd.Series.cumsum)
            df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).max().reset_index()
            df['value'] = df['cumsum']
            df = df.drop('cumsum',axis=1)
        else:
            df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).sum().reset_index()

        data = df.sort_values(['user', 'time'])

        measurements = {}
        for user in data['user'].unique():
            user_df = data[data['user'] == user]
            idx = pd.date_range(min(user_df.time), max(user_df.time))
            user_df = user_df.set_index('time')
            user_df = user_df.reindex(idx)
            user_df.index.names = ['time']
            user_df['user'].ffill(inplace=True)
            user_df['value'].fillna(0,inplace=True)

            measurements[user] = user_df.reset_index()

            del measurements[user]['user']

        return measurements
