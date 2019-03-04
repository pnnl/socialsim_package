
import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import burst_detection as bd

from .measurements import MeasurementsBaseClass


class RecurrenceMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset_df, id_col='id_h', timestamp_col="nodeTime", userid_col="nodeUserID", platform_col="platform", configuration={}, meatadata=None, communities=None, community_col="community", log_file='recurrence_measurements_log.txt', node_list=None, community_list=None):
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
        self.community_col = community_col
        self.platform_col = platform_col
        self.measurement_type = 'recurrence'
        self.get_per_epoch_counts()
        self.detect_bursts()
        self.time_between_bursts_distribution = None

    def get_per_epoch_counts(self, time_granularity='Min'):
        '''
        group activity by provided time granularity and get size and unique user counts per epoch
        time_granularity: s, Min, 10Min, 30Min, H, D, ...
        '''
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

