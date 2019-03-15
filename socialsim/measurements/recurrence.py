
import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import burst_detection as bd
import os
from .measurements import MeasurementsBaseClass


class RecurrenceMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset_df, configuration_subset=None, metadata=None, 
        id_col='nodeID', timestamp_col="nodeTime", 
        userid_col="nodeUserID", platform_col="platform", configuration={}, communities=None, 
        log_file='recurrence_measurements_log.txt', node_list=None, 
        community_list=None):
        """
        :param dataset_df: dataframe containing all posts for a single coin in all platforms
        :param timestamp_col: name of the column containing the time of the post
        :param id_col: name of the column containing the post id
        :param userid_col: name of the column containing the user id
        """
        super(RecurrenceMeasurements, self).__init__(
            dataset_df, configuration, log_file=log_file)
        self.dataset_df = dataset_df
        self.community_set = communities
        self.timestamp_col = timestamp_col
        self.id_col = id_col
        self.userid_col = userid_col
        self.platform_col = platform_col
        self.measurement_type = 'recurrence'
        self.get_per_epoch_counts()
        self.detect_bursts()
        self._time_between_bursts_distribution = None

    def get_per_epoch_counts(self, time_granularity='Min'):
        '''
        group activity by provided time granularity and get size and unique user counts per epoch
        time_granularity: s, Min, 10Min, 30Min, H, D, ...
        '''
        self.dataset_df[self.timestamp_col] = pd.to_datetime(self.dataset_df[self.timestamp_col])
        self.counts_df = self.dataset_df.set_index(self.timestamp_col).groupby(pd.Grouper(
            freq=time_granularity))[[self.id_col, self.userid_col]].nunique().reset_index()

    def detect_bursts(self, s=2, gamma=0.5):
        '''
        detect intervals with bursts of activity: [begin_timestamp, end_timestamp)
        :param s: multiplicative distance between states (input to burst_detection library)
        :param gamma: difficulty associated with moving up a state (input to burst_detection library)
        burst_detection library: https://pypi.org/project/burst_detection/
       '''
        r = self.counts_df[self.id_col].values
        n = len(r)
        d = np.array([sum(r)] * n, dtype=float)
        q = bd.burst_detection(r, d, n, s, gamma, 1)[0]
        bursts_df = bd.enumerate_bursts(q, 'burstLabel')
        index_date = pd.Series(
            self.counts_df[self.timestamp_col].values, index=self.counts_df.index).to_dict()
        bursts_df['begin_timestamp'] = bursts_df['begin'].map(index_date)
        bursts_df['end_timestamp'] = bursts_df['end'].map(index_date)
        time_granularity = index_date[1] - index_date[0]
        self.burst_intervals = [(burst['begin_timestamp'], burst['end_timestamp'] +
                                 time_granularity) for _, burst in bursts_df.iterrows()]
        self.update_with_burst()

    def update_with_burst(self):
        '''update dataset_df with burst index'''
        for idx, burst_interval in enumerate(self.burst_intervals):
            self.dataset_df.loc[self.dataset_df[self.timestamp_col].between(
                burst_interval[0], burst_interval[1], inclusive=False), 'burst_index'] = idx
        self.grouped_bursts = self.dataset_df.dropna().groupby('burst_index')

    @property
    def time_between_bursts_distribution(self):
        if self._time_between_bursts_distribution is None:
            self._time_between_bursts_distribution = [(start_j - end_i).total_seconds() / (60*60*24) for (_, end_i), (start_j, _) in zip(self.burst_intervals[:-1], self.burst_intervals[1:])]
        return self._time_between_bursts_distribution


    def number_of_bursts(self):
        '''
        How many renewed bursts of activity are there?
        '''
        return len(self.burst_intervals)

    def time_between_bursts(self):
        '''
        How much time elapses between renewed bursts of activity on average?
        Time granularity: days
        '''
        return np.mean(self.time_between_bursts_distribution)

    def average_size_of_each_burst(self):
        '''
        How many times is the information shared per burst on average?
        '''
        return self.grouped_bursts.size().reset_index(name='size')['size'].mean()

    def average_number_of_users_per_burst(self):
        '''
        How many users are reached by the information during each burst on average?
        '''
        return self.grouped_bursts[[self.userid_col]].nunique().reset_index()[self.userid_col].mean()

    def burstiness_of_burst_timing(self):
        '''Do multiple bursts of renewed activity tend to cluster together?'''
        std = np.std(self.time_between_bursts_distribution)
        mean = np.mean(self.time_between_bursts_distribution)
        return (std - mean) / (std + mean) if std + mean > 0 else 0

    def new_users_per_burst(self):
        '''
        How many new users are reached by the information during each burst on average?
        First burst is also counted.
        '''
        users = set()
        num_new_users = []
        for _, single_burst_df in self.grouped_bursts:
            old_len = len(users)
            users.update(single_burst_df[self.userid_col].unique())
            num_new_users.append(len(users) - old_len)
#             print(old_len)
#         print(num_new_users)
        return np.mean(num_new_users)

    def lifetime_of_each_burst(self):
        '''
        How long does each burst last on average?
        Time granularity: minutes
        '''
        return np.mean([(end_i - start_i).total_seconds() / 60 for (start_i, end_i) in self.burst_intervals])

    def average_proportion_of_top_platform_per_burst(self):
        '''
        Do individual bursts tend to occur on a single platform or are they distributed among platforms?
        '''
        return np.mean([single_burst_df[self.platform_col].value_counts().max()/len(single_burst_df) for _, single_burst_df in self.grouped_bursts])


class RecurrenceCommunityMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset_df, configuration_subset=None, metadata=None,
    id_col='id_h', timestamp_col="nodeTime", userid_col="nodeUserID", platform_col="platform", configuration={}, communities=None, log_file='recurrence_measurements_log.txt', node_list=None, community_list=None):
        """
        :param dataset_df: dataframe containing all posts for all communities (Eg. coins for scenario 2) in all platforms
        :param timestamp_col: name of the column containing the time of the post
        :param id_col: name of the column containing the post id
        :param userid_col: name of the column containing the user id
        """
        super(RecurrenceCommunityMeasurements, self).__init__(dataset_df, configuration, log_file=log_file)
        self.dataset_df         = dataset_df
        self.community_set      = communities
        self.timestamp_col      = timestamp_col
        self.id_col             = id_col
        self.userid_col         = userid_col
        self.platform_col       = platform_col
        self.measurement_type   = 'recurrence'
        self.metadata           = metadata
        self.get_percommunity_recurrence_measurements()

    def get_percommunity_recurrence_measurements(self):
        self.percommunity_recurrence_measurements = {}
        for community, community_ids in self.metadata.communities.items():
            community_df = self.dataset_df[self.dataset_df[self.id_col].isin(community_ids)]
            self.percommunity_recurrence_measurements[community] = RecurrenceMeasurements(dataset_df=community_df, id_col=self.id_col, timestamp_col=self.timestamp_col, userid_col=self.userid_col, platform_col=self.platform_col, configuration=self.configuration)

    def run_for_all_communities(self, measurement_name):
        return [getattr(percommunity_recurrence_measurements, measurement_name)() for community, percommunity_recurrence_measurements in self.percommunity_recurrence_measurements.items()]


    def distribution_of_number_of_bursts(self):
        '''
        How does the number of renewed bursts of activity vary across different pieces of information?
        '''
        return self.run_for_all_communities('number_of_bursts')


    def distribution_of_time_between_bursts(self):
        '''
        How does the time elapsed between renewed bursts of activity vary across different pieces of information?
        '''
        return self.run_for_all_communities('time_between_bursts')


    def distribution_of_average_burst_size(self):
        '''
        How does the number times is the information shared per burst vary across different pieces of information?
        '''
        return self.run_for_all_communities('average_size_of_each_burst')


    def distribution_of_average_number_of_users_per_burst(self):
        '''
        How does the number of users reached during each burst vary across different pieces of information?
        '''
        return self.run_for_all_communities('average_number_of_users_per_burst')


    def distribution_of_burst_timing_burstiness(self):
        '''
        How does the burstiness of burst timing vary across different pieces of information?
        '''
        return self.run_for_all_communities('burstiness_of_burst_timing')


    def distribution_of_new_users_per_burst(self):
        '''
        How does the number of new users reached during each burst vary across different pieces of information?
        '''
        return self.run_for_all_communities('new_users_per_burst')


    def distribution_of_burst_lifetime(self):
        '''
        How does the burst length vary across different pieces of information?
        '''
        return self.run_for_all_communities('lifetime_of_each_burst')


    def distribution_of_burst_platform_proportion(self):
        '''
        How does the prominence of a single platform vary across different pieces of information?
        '''
        return self.run_for_all_communities('average_proportion_of_top_platform_per_burst')




