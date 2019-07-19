import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import burst_detection as bd
import matplotlib.pyplot as plt
import os
import joblib
from tsfresh import extract_features

from .measurements import MeasurementsBaseClass
from ..utils import get_community_contentids
import pprint
from .model_parameters.selected_features import selected_features

import re


class RecurrenceMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset_df, configuration={}, metadata=None,
                 id_col='nodeID', timestamp_col="nodeTime", userid_col="nodeUserID", platform_col="platform",
                 content_col="informationID", communities=None, log_file='recurrence_measurements_log.txt',
                 selected_content=None, selected_communties=None, time_granularity='12H', 
                 plot=False, show=False, save_plots=False, plot_dir='./'):
        """
        :param dataset_df: dataframe containing all posts for all communities (Eg. coins for scenario 2) in all platforms
        :param timestamp_col: name of the column containing the time of the post
        :param id_col: name of the column containing the post id
        :param userid_col: name of the column containing the user id
        :param content_col: name of the column containing the content the simulation was done for eg. coin name
        """
        super(RecurrenceMeasurements, self).__init__(dataset_df, configuration, log_file=log_file)
        self.dataset_df = dataset_df
        self.community_set = communities
        self.timestamp_col = timestamp_col
        self.id_col = id_col
        self.userid_col = userid_col
        self.platform_col = platform_col
        self.content_col = content_col
        self.measurement_type = 'recurrence'
        self.metadata = metadata
        self.selected_content = selected_content
        self.selected_communties = selected_communties
        self.community_contentids = None
        self.time_granularity = time_granularity
        self.plot = plot
        self.save_plots = save_plots

        self.plot_dir = plot_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.gammas = {k: {p: None for p in self.dataset_df[self.platform_col].unique()} for k in self.dataset_df[self.content_col].unique()}

        self.min_date = self.dataset_df[self.timestamp_col].min()
        self.max_date = self.dataset_df[self.timestamp_col].max()

        if not self.metadata is None:
            if hasattr(self.metadata, 'community_directory'):
                if not self.metadata.community_directory is None:
                    self.community_contentids = get_community_contentids(self.metadata.community_directory)
            if self.metadata.use_info_data and 'gamma' in self.metadata.info_data.columns:
                
                for i, row in self.metadata.info_data[[self.content_col, self.platform_col, 'gamma']].iterrows():
                    if row[self.content_col] in self.gammas.keys():
                        self.gammas[row[self.content_col]][row[self.platform_col]] = row['gamma'] 

        self.initialize_recurrence_measurements(ever_show=show)

    def list_measurements(self):
        count = 0
        for f in dir(self):
            if not f.startswith('_'):
                func = getattr(self, f)
                if callable(func):
                    doc_string = func.__doc__
                    if not doc_string is None and 'Measurement:' in doc_string:
                        desc = re.search('Description\:([\s\S]+?)Input', doc_string).groups()[0].strip()
                        #print('{}) {}: {}\n'.format(count + 1, f, desc))
                        print('{}) {}\n'.format(count + 1, f))
                        count += 1

    def initialize_recurrence_measurements(self,ever_show=False):
        self.content_recurrence_measurements = {}
        max_plots_to_show = 5
        show = False
        n_ids = self.dataset_df[self.content_col].nunique()
        num_plots = 0
        for content_id, content_df in self.dataset_df.groupby(self.content_col):
            if num_plots < max_plots_to_show and ever_show:
                show = True
            else:
                show = False
            if not self.plot:
                show = False
            self.content_recurrence_measurements[content_id] = ContentRecurrenceMeasurements(dataset_df=content_df,
                                                                                             id_col=self.id_col,
                                                                                             metadata=self.metadata,
                                                                                             timestamp_col=self.timestamp_col,
                                                                                             userid_col=self.userid_col,
                                                                                             platform_col=self.platform_col,
                                                                                             content_col=self.content_col,
                                                                                             configuration=self.configuration,
                                                                                             content_id=content_id,
                                                                                             time_granularity=self.time_granularity,
                                                                                             gamma=self.gammas[
                                                                                                 content_id],
                                                                                             min_date=self.min_date,
                                                                                             max_date=self.max_date,
                                                                                             plot_flag=self.plot,
                                                                                             show=show,
                                                                                             plot_dir=self.plot_dir,
                                                                                             save_plots = self.save_plots)
            num_plots += 1

    def run_content_level_measurement(self, measurement_name, scale='node',
                                      selected_content=None, **kwargs):
        # determine selected nodes in order of priority from the argument to this function, the selected nodes from the metadata, and all nodes
        if scale == 'node':
            if self.metadata is None:
                metadatanodelist = None
            else:
                metadatanodelist = self.metadata.node_list
            selected_content = next(x for x in [selected_content, self.selected_content, metadatanodelist,
                                                self.content_recurrence_measurements.keys()] if
                                    x is not None)
        else:
            selected_content = self.dataset_df[self.content_col].unique()

        contentid_value = {
            content_id: getattr(self.content_recurrence_measurements[content_id], measurement_name)(**kwargs) for
            content_id
            in selected_content if content_id in self.content_recurrence_measurements.keys()}
        contentid_value = {k: v for k, v in contentid_value.items() if not v is None and not np.isnan(v)}
        if scale == 'node':
            return contentid_value
        elif scale == 'population':
            return pd.DataFrame(list(contentid_value.items()), columns=[self.content_col, 'value'])

    def run_community_level_measurement(self, measurement_name, selected_communties=None, **kwargs):
        if self.community_contentids is None:
            print('No communities provided')
            return
        selected_communities = next(
            x for x in [selected_communties, self.selected_communties, self.community_contentids.keys()] if
            x is not None)

        meas = {}
        for community in selected_communities:
            meas[community] = pd.DataFrame(list(self.run_content_level_measurement(measurement_name,
                                                                                   selected_content=
                                                                                   self.community_contentids[community],
                                                                                   **kwargs).items()),
                                           columns=[self.content_col, 'value'])

        return meas

    def node_number_of_bursts(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='number_of_bursts')

    def node_time_between_bursts(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='time_between_bursts')

    def node_average_size_of_each_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='average_size_of_each_burst')

    def node_average_number_of_users_per_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='average_number_of_users_per_burst')

    def node_burstiness_of_burst_timing(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='burstiness_of_burst_timing')

    def node_new_users_per_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='new_users_per_burst')

    def node_lifetime_of_each_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='lifetime_of_each_burst')

    def node_average_proportion_of_top_platform_per_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='average_proportion_of_top_platform_per_burst')

    def community_distribution_of_number_of_bursts(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='number_of_bursts')

    def community_distribution_of_time_between_bursts(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='time_between_bursts')

    def community_distribution_of_average_burst_size(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='average_size_of_each_burst')

    def community_distribution_of_average_number_of_users_per_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='average_number_of_users_per_burst')

    def community_distribution_of_burst_timing_burstiness(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='burstiness_of_burst_timing')

    def community_distribution_of_new_users_per_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='new_users_per_burst')

    def community_distribution_of_burst_lifetime(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='lifetime_of_each_burst')

    def community_distribution_of_burst_platform_proportion(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_community_level_measurement(measurement_name='average_proportion_of_top_platform_per_burst')

    def population_distribution_of_number_of_bursts(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='number_of_bursts',scale='population')

    def population_distribution_of_time_between_bursts(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='time_between_bursts',scale='population')

    def population_distribution_of_average_burst_size(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='average_size_of_each_burst',scale='population')

    def population_distribution_of_average_number_of_users_per_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='average_number_of_users_per_burst',scale='population')

    def population_distribution_of_burst_timing_burstiness(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='burstiness_of_burst_timing',scale='population')

    def population_distribution_of_new_users_per_burst(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='new_users_per_burst',scale='population')

    def population_distribution_of_burst_lifetime(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='lifetime_of_each_burst',scale='population')

    def population_distribution_of_burst_platform_proportion(self):
        """
        Measurement:

        Description:

        Input:
        """
        return self.run_content_level_measurement(measurement_name='average_proportion_of_top_platform_per_burst',scale='population')

class BurstDetection():
    def __init__(self, dataset_df, metadata, id_col='nodeID', timestamp_col="nodeTime",
                 platform_col="platform", time_granularity='D',
                 min_date=None, max_date=None, content_id=''):
        self.dataset_df = dataset_df
        self.metadata = metadata
        self.timestamp_col = timestamp_col
        self.id_col = id_col
        self.platform_col = platform_col
        self.time_granularity = time_granularity
        self.min_date = min_date
        self.max_date = max_date
        if not min_date is None:
            self.min_date = pd.Timestamp(min_date)
        if not max_date is None:
            self.max_date = pd.Timestamp(max_date)
        self.content_id = content_id

    def detect_bursts(self, gamma=None):
        '''
        detect bursts for each platform separately and then merge overlapping bursts
        time_granularity: s, Min, 10Min, 30Min, H, D, ...
        '''

        def merge_bursts(all_bursts_df):
            '''
            merge bursts from timeseries of different platforms
            '''
            burst_intervals = []
            all_bursts_df = all_bursts_df.sort_values('start_timestamp').reset_index(drop=True)
            for idx, burst in all_bursts_df.iterrows():
                if idx == 0:
                    burst_intervals.append((burst['start_timestamp'], burst['end_timestamp']))
                    continue
                if can_be_merged(burst_intervals[-1], burst['start_timestamp']):
                    burst_intervals[-1] = (min(burst_intervals[-1][0], burst['start_timestamp']),
                                           max(burst_intervals[-1][1], burst['end_timestamp']))
                else:
                    burst_intervals.append((burst['start_timestamp'], burst['end_timestamp']))

            return burst_intervals

        def can_be_merged(earlier_interval, later_interval_start):
            '''
            check if two burst intervals can be merged
            '''
            if earlier_interval[0] <= later_interval_start <= earlier_interval[1]:
                return True

        all_bursts_dfs = []
        for platform, platform_df in self.dataset_df.groupby(self.platform_col):
            counts_df = platform_df.set_index(self.timestamp_col).groupby(pd.Grouper(freq=self.time_granularity))[
                [self.id_col]].count()

            if type(gamma) == dict:
                g = gamma[platform]
            else:
                g = gamma

            # make sure the time series covers the full range
            if not self.min_date is None and counts_df.index.min() > self.min_date:
                counts_df.loc[self.min_date] = 0
            if not self.max_date is None and counts_df.index.max() < self.max_date:
                counts_df.loc[self.max_date] = 0

            # fill in missing time stamps with a zero count
            counts_df = counts_df.resample(self.time_granularity).mean().fillna(0)

            counts_df = counts_df.reset_index()

            bursts_df = self.detect_bursts_of_a_timeseries(counts_df, gamma=g, platform=platform)
            if bursts_df is None:
                continue
            bursts_df['platform'] = platform
            all_bursts_dfs.append(bursts_df)

        if len(all_bursts_dfs) > 0:
            all_bursts_df = pd.concat(all_bursts_dfs).reset_index(drop=True)
            return merge_bursts(all_bursts_df)
        else:
            return []

    def detect_bursts_of_a_timeseries(self, timeseries_df, gamma=None, platform='all'):
        '''
        detect intervals with bursts of activity: [start_timestamp, end_timestamp)
        :param ts_df: timeseries_df for a single platform
        :param s: multiplicative distance between states (input to burst_detection library)
        :param gamma: difficulty associated with moving up a state (input to burst_detection library)
        burst_detection library: https://pypi.org/project/burst_detection/
        '''
        if len(timeseries_df) < 2:
            return None
        r = timeseries_df[self.id_col].values
        n = len(r)
        d = np.array([sum(r)] * n, dtype=float)
        if gamma is None and np.max(r) >= 5:
            gamma = self.predict_gamma_for_timeseries(timeseries_df)
            with open('predicted_gammas.csv', 'a') as f:
                f.write(self.content_id + ',' + platform + ',' + str(gamma) + '\n')
        elif np.max(r) < 5:
            return None

        q = bd.burst_detection(r, d, n, s=2, gamma=gamma, smooth_win=1)[0]
        bursts_df = bd.enumerate_bursts(q,
                                        'burstLabel')  # returns a df with 'begin' and 'end' columns for a burst where both begin and end indices are included.
        index_date = pd.Series(
            timeseries_df[self.timestamp_col].values, index=timeseries_df.index).to_dict()
        time_granularity = index_date[1] - index_date[0]
        bursts_df['start_timestamp'] = bursts_df['begin'].map(index_date)
        bursts_df['end_timestamp'] = bursts_df['end'].map(index_date)
        bursts_df['end_timestamp'] = bursts_df['end_timestamp'] + time_granularity
        if len(bursts_df) > 0:
            return bursts_df

    def predict_gamma_for_timeseries(self, timeseries_df):
        '''
        Predict the best gamma based on time series properties
        '''
        timeseries_df[
            'dummy_col'] = 'dummy'  # the library requires an id column, but all ids are the same for our timeseries, so adding a dummy id column
        try:
            features_df = extract_features(timeseries_df.rename(columns={self.id_col: 'value'}),
                                           column_id='dummy_col', column_sort=self.timestamp_col,
                                           disable_progressbar=True)[selected_features].fillna(0)
            features_df = features_df.replace(np.inf, 0)
            features_df = features_df.replace(-np.inf, 0)
            gamma = self.metadata.estimator.predict(features_df)[0]
        except:
            gamma = 1.0
        return gamma


class ContentRecurrenceMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset_df, configuration={},
                 metadata=None,
                 id_col='nodeID', timestamp_col="nodeTime",
                 userid_col="nodeUserID", platform_col="platform",
                 content_col="informationID", communities=None,
                 log_file='recurrence_measurements_log.txt', content_id=None,
                 time_granularity='D', gamma=None,
                 min_date=None, max_date=None, plot_flag=True, show=False, plot_dir='./', save_plots=False):
        """
        :param dataset_df: dataframe containing all posts for a single coin in all platforms
        :param timestamp_col: name of the column containing the time of the post
        :param id_col: name of the column containing the post id
        :param userid_col: name of the column containing the user id
        :param content_col: name of the column containing the content the simulation was done for eg. coin name
        """
        super(ContentRecurrenceMeasurements, self).__init__(
            dataset_df, configuration, log_file=log_file)
        self.dataset_df = dataset_df.copy()
        self.metadata = metadata
        self.community_set = communities
        self.timestamp_col = timestamp_col
        self.id_col = id_col
        self.userid_col = userid_col
        self.platform_col = platform_col
        self.content_col = content_col
        self.measurement_type = 'recurrence'
        self.content_id = content_id
        self.time_granularity = time_granularity

        self.plot_dir = plot_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.save_plots = save_plots

        burstDetection = BurstDetection(dataset_df=self.dataset_df, metadata=self.metadata, id_col=self.id_col,
                                        timestamp_col=self.timestamp_col, platform_col=self.platform_col,
                                        time_granularity=self.time_granularity,
                                        min_date=min_date,
                                        max_date=max_date,
                                        content_id=content_id)
        self.burst_intervals = burstDetection.detect_bursts(gamma)
        self.update_with_burst()
        if plot_flag:
            plot_df = self.dataset_df.copy()
            plot_df.set_index(self.timestamp_col, inplace=True)
            new_df = plot_df.groupby(pd.Grouper(freq=self.time_granularity))[[self.content_col]].count()
            new_df.reset_index(inplace=True)
            plt.figure()
            plt.plot(new_df[self.timestamp_col], new_df[self.content_col])
            content_name = self.dataset_df[content_col].iloc[0]
            if len(self.burst_intervals) > 0:
                for idx, burst in enumerate(self.burst_intervals):
                    plt.axvspan(xmin=burst[0], xmax=burst[1], color="gray", alpha=0.25)

            plt.title("Recurrence with bursts - {}".format(str(content_name)))
            plt.xticks(fontsize=10, rotation=30)
            if self.save_plots:
                plt.savefig(self.plot_dir + str(content_name) + "_recurrence_with_bursts.png", bbox_inches='tight')

            if show:
                plt.show()
        self._time_between_bursts_distribution = None

    def list_measurements(self):
        count = 0
        for f in dir(self):
            if not f.startswith('_'):
                func = getattr(self, f)
                if callable(func):
                    doc_string = func.__doc__
                    if not doc_string is None and 'Measurement:' in doc_string:
                        desc = re.search('Description\:([\s\S]+?)Input', doc_string).groups()[0].strip()
                        print('{}) {}: {}'.format(count + 1, f, desc))
                        count += 1

    def update_with_burst(self):
        '''update dataset_df with burst index'''
        self.dataset_df.loc['burst_index', :] = None
        if len(self.burst_intervals) == 0:
            self.grouped_bursts = None
            return
        for idx, burst_interval in enumerate(self.burst_intervals):
            self.dataset_df.loc[self.dataset_df[self.timestamp_col].between(
                burst_interval[0], burst_interval[1], inclusive=True), 'burst_index'] = idx

        if 'burst_index' in self.dataset_df.columns:
            self.grouped_bursts = self.dataset_df.dropna(subset=['burst_index']).groupby('burst_index')
        else:
            self.grouped_burst = None

    @property
    def time_between_bursts_distribution(self):
        """
        Measurement:

        Description:

        Input:

        Output:
        """
        if self._time_between_bursts_distribution is None:
            self._time_between_bursts_distribution = [(start_j - end_i).total_seconds() for (_, end_i), (start_j, _) in
                                                      zip(self.burst_intervals[:-1], self.burst_intervals[1:])]
        return self._time_between_bursts_distribution

    def time_granularity_scaling(self, time_granularity):

        if time_granularity == 'M':
            return 60.0
        elif time_granularity == 'H':
            return 60.0 * 60.0
        elif time_granularity == 'D':
            return 60.0 * 60.0 * 24.0
        elif time_granularity == 'W':
            return 60.0 * 60.0 * 24.0 * 7.0
        elif time_granularity == 'm':
            return 60.0 * 60.0 * 24.0 * 30.5

    def number_of_bursts(self):
        """
        Measurement: number_of_bursts (population_distribution_of_number_of_bursts, community_distribution_of_number_of_bursts, node_number_of_bursts)

        Description: How many renewed bursts of activity are there?

        Input:

        Output:
        """
        return len(self.burst_intervals)

    def time_between_bursts(self, time_granularity='D'):
        """
        Measurement: time_between_bursts (population_distribution_of_time_between_bursts, community_distribution_of_time_bwtween_bursts, node_time_between_bursts)

        Description: How much time elapses between renewed bursts of activity on average?
        Time granularity: days

        Input:

        Output:
        """
        delta_t = np.mean(self.time_between_bursts_distribution)

        return delta_t / self.time_granularity_scaling(time_granularity)

    def average_size_of_each_burst(self):
        """
        Measurement: average_size_of_each_burst (population_distribution_of_average_size_of_each_burst, community_distribution_of_average_size_of_each_burst, node_average_size_of_each_burst)

        Description: How many times is the information shared per burst on average?

        Input:

        Output:
        """
        if self.grouped_bursts is None:
            return None

        return self.grouped_bursts.size().reset_index(name='size')['size'].mean()

    def average_number_of_users_per_burst(self):
        """
        Measurement: average_number_of_users_per_burst (population_distribution_average_number_of_users_per_burst, community_average_number_of_users_per_burst, node_average_number_of_users_per_burst)

        Description: How many users are reached by the information during each burst on average?

        Input:

        Output:
        """
        if self.grouped_bursts is None:
            return None

        return self.grouped_bursts[[self.userid_col]].nunique().reset_index()[self.userid_col].mean()

    def burstiness_of_burst_timing(self):
        """
        Measurement: burstiness_of_burst_timing (population_distribution_burstiness_of_burst_timing, community_distribution_burstiness_of_burst_timing, node_burstiness_of_burst_timing)

        Description: Do multiple bursts of renewed activity tend to cluster together?

        Input:

        Output:
        """
        if len(self.time_between_bursts_distribution) > 2:
            std = np.std(self.time_between_bursts_distribution)
            mean = np.mean(self.time_between_bursts_distribution)
            return (std - mean) / (std + mean) if std + mean > 0 else 0
        else:
            return None

    def new_users_per_burst(self):
        """
        Measurement: new_users_per_burst (population_distriubtion_new_users_per_burst, community_distribution_new_users_per_burst, node_new_users_per_burst)

        Description: How many new users are reached by the information during each burst on average?
        First burst is also counted.

        Input:

        Output:
        """
        if self.grouped_bursts is None:
            return None

        users = set()
        num_new_users = []
        for _, single_burst_df in self.grouped_bursts:
            old_len = len(users)
            users.update(single_burst_df[self.userid_col].unique())
            num_new_users.append(len(users) - old_len)
        return np.mean(num_new_users)

    def lifetime_of_each_burst(self, time_granularity='H'):
        """
        Measurement: lifetime_of_each_burst (population_distribution_lifetime_of_each_burst, community_distribution_lifetime_of_each_burst, node_lifetime_of_each_burst)

        Description: How long does each burst last on average?
        Time granularity: minutes

        Input:

        Output:
        """
        return np.mean(
            [(end_i - start_i).total_seconds() / self.time_granularity_scaling(time_granularity) for (start_i, end_i) in
             self.burst_intervals])

    def average_proportion_of_top_platform_per_burst(self):
        """
        Measurement: average_proportion_of_top_platform_per_burst (population_distribution_average_proportion_of_top_platform_per_burst, community_distribution_average_proportion_of_top_platform_per_burst, node_average_proportion_of_top_platform_per_burst)

        Description: Do individual bursts tend to occur on a single platform or are they distributed among platforms?

        Input:

        Output:
        """
        if self.grouped_bursts is None:
            return None

        return np.mean(
            [single_burst_df[self.platform_col].value_counts().max() / len(single_burst_df) for _, single_burst_df in self.grouped_bursts])
