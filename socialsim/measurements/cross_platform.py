
import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from itertools import combinations
import pprint

from .measurements import MeasurementsBaseClass


class CrossPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, metadata=None, 
        platform_col="platform", timestamp_col="nodeTime", 
        user_col="nodeUserID", content_col="informationID", 
        community_col="community", node_list=None, community_list=None,
        log_file='cross_platform_measurements_log.txt'):

        """
        :param dataset: dataframe containing all pieces of content and associated data, sorted by time
        :param configuration:
        :param platform_col: name of the column containing the platforms
        :param timestamp_col: name of the column containg the time of the content
        :param user_col: name of the column containing the user ids of each piece of content
        :param content_col: name of the column where the content can be found
        :param community_col: name of the column containing the subset of content in a community
        :param log_file:
        """
        super(CrossPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)
        self.dataset            = dataset
        self.timestamp_col      = timestamp_col
        self.user_col           = user_col
        self.platform_col       = platform_col
        self.content_col        = content_col
        self.community_col      = community_col

        self.dataset[self.community_col] = np.random.choice(['A','B','C'],len(self.dataset))

        print(self.dataset)

        self.measurement_type = 'cross_platform'

        if metadata is None:
#            self.community_set = None
            self.community_set = self.dataset
        else:
            self.community_set = metadata.communities

        if node_list == "all":
            self.node_list = self.dataset[self.content_col].tolist()
        elif node_list is not None:
            self.node_list = node_list
        else:
            self.node_list = []

        if self.community_set is not None:
            if community_list == "all":
                self.community_list = self.community_set[self.community_col].unique()
            elif community_list is not None:
                self.community_list = community_list
            else:
                self.community_list = []
        else:
            self.community_list = []

    def select_data(self, nodes=None, communities=None):
        """
        Subset the data based on the given communities or pieces of content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: New DataFrame with the select communities/content only
        """

        if nodes is None:
            nodes = self.node_list
        if communities is None:
            communities = self.community_list

        if len(nodes) > 0:
            data = self.dataset.loc[self.dataset[self.content_col].isin(nodes)]
        elif len(communities) > 0:
            data = self.community_set[self.community_set[self.community_col].isin(communities)]
        else:
            data = self.dataset.copy()
        return data

    def time_granularity(self,time_diff,granularity):

        if granularity.lower() == 's':
            return time_diff.seconds
        elif granularity.lower() == 'm':
            return time_diff.minutes
        elif granularity.lower() == 'h':
            return time_diff.hours
        elif granularity.lower() == 'd':
            return time_diff.days

    def order_of_spread(self, nodes=None, communities=None):
        """
        Determine the order of spread between platforms of a community/content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a dictionary mapping each platform to an array of percent freq. in each rank
                If community, a nested dictionary mapping each community to a dictionary mapping each platform
                    to an array of percent freq. in each rank
                Else, a dictionary mapping between the content to the ranked list of platforms
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        data = self.select_data(nodes, communities)
        platforms = sorted(data[self.platform_col].unique())

        def platform_order(diction):
            plat_diction = {p: np.zeros((len(platforms))) for p in platforms}
            for _, plat in diction.items():
                for p in platforms:
                    pos, = np.where(plat == p)
                    plat_diction[p][pos] += 1
            for k, v in plat_diction.items():
                if v.sum(axis=0) == 0:
                    plat_diction[k] = [0]*len(platforms)
                else:
                    plat_diction[k] = v / v.sum(axis=0)
            # for k, v in plat_diction.items():
            #     plat_diction[k] = v.tolist()
            return plat_diction

        if len(communities) > 0:
            data.drop_duplicates(subset=[self.community_col, self.content_col, self.platform_col], inplace=True)
            data = data.groupby(self.community_col).apply(
                lambda x: x.groupby(self.content_col).apply(lambda y: y[self.platform_col].values).to_dict())
            community_platform_order = data.to_dict()
            keywords_to_order = {}
            for comm, content_diction in community_platform_order.items():
                keywords_to_order[comm] = platform_order(content_diction)
            return keywords_to_order
        else:
            data.drop_duplicates(subset=[self.content_col, self.platform_col], inplace=True)
            data = data.groupby(self.content_col).apply(lambda x: x[self.platform_col].tolist())
            n_platforms = data.apply(len)
            data = data[n_platforms > 1] 
            keywords_to_order = data.to_dict()
            if len(nodes) == 0:
                plt_1, val = [], []
                keywords_to_order = platform_order(keywords_to_order)
                for k, v in keywords_to_order.items():
                    plt_1.extend([k]*len(v))
                    val.extend(v)
                return pd.DataFrame({"platform": plt_1, "value": val})
            return keywords_to_order


    def time_delta(self, time_granularity="s", nodes=None, communities=None):
        """
        Determine the amount of time it takes for a community/content to appear on another platform
        :param time_granularity: Unit of time to calculate {s=seconds, m=minutes, h=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a DataFrame with columns platform_1, platform_2, and value
                If community, a dictionary mapping a community to a DataFrame with columns
                    platform_1, platform_2, and value
                Else, a dictionary mapping a content to a list (equal in length to the number of platforms)
                of the time passed since the first time that content was observed. This list is sorted alphabetically
                to preserve the same ordering of platforms in all cases. This can causes negative times.
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        # elif communities == "all":
        #     communities = self.dataset[self.platform_col].tolist()


        data = self.select_data(nodes, communities)


        if len(communities) == 0:
            group_col = [self.content_col, self.platform_col]
        else:
            group_col = [self.content_col, self.community_col, self.platform_col]

        data.drop_duplicates(subset=group_col, inplace=True)
        data.sort_values(by=[self.timestamp_col], inplace=True, ascending=True)
                
        print(data[group_col + [self.timestamp_col]])

        group_col = [c for c in group_col if c != self.platform_col]

        #get all pairs of timestamps in each group
        data_combinations = data.groupby(group_col)[self.timestamp_col].apply(combinations,2).apply(list).apply(pd.Series).stack().apply(pd.Series)
        data_combinations.columns = [self.timestamp_col,'next_platform_timestamp']
        #get time differences for each pair of time stamps
        data_combinations['time_diff'] = data_combinations['next_platform_timestamp'] - data_combinations[self.timestamp_col]
        data_combinations = data_combinations.reset_index()
 
        #merge to get first platform
        data_combinations = data_combinations.merge(data[group_col + [self.platform_col,self.timestamp_col]],
                                                    on=group_col + [self.timestamp_col],how='left')

        #merge to get second platform
        data = data_combinations.merge(data[group_col + [self.platform_col,self.timestamp_col]],
                                       right_on=group_col + [self.timestamp_col],
                                       left_on=group_col + ['next_platform_timestamp'],
                                       how='left',suffixes=('_first','_second'))

        data = data[group_col + ['platform_first','platform_second','time_diff']]
        data = data.rename(columns={'time_diff':'value'})

        data['value'] = data['value'].apply(lambda x: self.time_granularity(x,time_granularity))

        if len(nodes) > 0:
            data = dict(tuple(data.groupby(self.content_col)))
        elif len(communities) > 0:
            data = dict(tuple(data[[self.community_col,'platform_first','platform_second','value']].groupby(self.community_col)))
        else:
            data = data[['platform_first','platform_second','value']]
            
        return(data)


    def overlapping_users(self, nodes=None, communities=None):
        """
        Calculate the percentage of users common to all platforms (that share in a community/content)
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix of percentages of common users to any pair of platforms.
                Else, a dictionary mapping a community/content to a matrix of the percentages of common users that
                share in that community/content across all pairs of platforms
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        data = self.select_data(nodes, communities)
 
        platforms = sorted(data[self.platform_col].unique())
        
        data['values'] = 1
        data = data.drop_duplicates(subset=[self.user_col, self.platform_col])
        cols = [self.user_col, self.platform_col, 'values']
        index_cols = [self.user_col]

        if len(communities) > 0:
            cols = [self.user_col, self.platform_col, self.community_col, 'values']
            index_cols = [self.user_col, self.community_col]
            group_col = self.community_col
        elif len(nodes) > 0:
            cols = [self.user_col, self.platform_col, self.content_col, 'values']
            index_cols = [self.user_col, self.content_col]
            group_col = self.content_col
        else:
            group_col = []

        user_platform = data[cols].pivot_table(index=index_cols,
                                               columns=self.platform_col,
                                               values='values').fillna(0)
        user_platform = user_platform.astype(bool)
        
        print(user_platform)

        def get_meas(grp):
            pl_1, pl_2, val = [], [], []
            for i, p1 in enumerate(platforms):
                for j, p2 in enumerate(platforms):
                    if p1 != p2:
                        x = (grp[p1] & grp[p2]).sum()
                        total = float(grp[p1].sum())
                        if total > 0:
                            x = x / total
                        else:
                            x = 0
                        pl_1.append(p1)
                        pl_2.append(p2)
                        val.append(x)

            meas = pd.DataFrame({"platform_1": pl_1, "platform_2": pl_2, "value": val})
            meas = meas.drop_duplicates(subset=["platform_1", "platform_2"])
            return meas

        if len(nodes) != 0 or len(communities) != 0:
            meas = {}
            for i, grp in user_platform.groupby(group_col):
                meas[i] = get_meas(grp)
        else:
            meas = get_meas(user_platform)
        return meas

    def size_of_audience(self, nodes=None, communities=None):
        """
        Determine the ranking of audience sizes on each platform
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a ranked list of the platforms with the largest audience sizes
                Else, a dictionary mapping the community/content to a ranked list of the platforms with the largest
                    audience sizes.
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        if len(communities) > 0:
            group_col = self.community_col
        elif len(nodes) > 0:
            group_col = self.content_col
        else:
            group_col = self.platform_col

        def audience(grp):
            aud = grp.groupby(self.platform_col).apply(lambda x: len(x[self.user_col].unique())).to_dict()
            return [item[0] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]

        if len(nodes) == 0 and len(communities) == 0:
            return audience(data)
        else:
            return data.groupby(group_col).apply(audience).to_dict()

    def speed_of_spread(self, nodes=None, communities=None):

        """
        Determine the speed at which the information is spreading
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a DataFrame with columns platform and value
                If community, a dictionary mapping a community to a a DataFrame with columns platform and value
                Else, a dictionary mapping each content to the ranked list of platforms on which it spreads the fastest
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        group_col = [self.content_col]
        if len(communities) > 0:
            group_col = [self.content_col,self.community_col]

        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds / 60.0 / 60.0
            if time == 0:
                speed = -1
            else:
                speed = len(grp) / time
            return speed

        speeds = data.groupby(self.platform_col).apply(lambda x: x.groupby(group_col).apply(get_speed)).reset_index()
        
        if 0 not in speeds.columns:
            speeds = pd.melt(speeds,id_vars=self.platform_col)

        speeds.columns = [self.platform_col] + group_col + ['value']
        speeds = speeds[speeds['value'] != -1]
        
        if len(nodes) > 0:
            speeds = dict(tuple(speeds.groupby(self.content_col)))
            speeds = {k:v[[self.platform_col,'value']] for k,v in speeds.items()}
        elif len(communities) > 0:
            speeds = dict(tuple(speeds.groupby(self.community_col)[[self.platform_col,'value']]))
            speeds = {k:v[[self.platform_col,'value']] for k,v in speeds.items()}
        else:
            speeds = speeds[[self.platform_col,'value']]

        return speeds

    def size_of_shares(self, nodes=None, communities=None):
        """
        Determine the number of shares per platform
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a ranked list of platforms based on total activity
                Else, a dictionary mapping the community/content to a ranked list of platforms based on activity
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        data = self.select_data(nodes, communities)


        group_col = [self.content_col]
        if len(communities) > 0:
            group_col += [self.community_col]

        share_counts = data.groupby([self.platform_col] + group_col)[self.timestamp_col].count().reset_index()
        share_counts.columns = [self.platform_col] + group_col + ['value']
        share_counts = share_counts.sort_values('value',ascending=False)

        if len(nodes) > 0:
            share_counts = dict(tuple(share_counts.groupby(self.content_col)))
            share_counts = {k:v[[self.platform_col,'value']] for k,v in share_counts.items()}
        elif len(communities) > 0:
            share_counts = dict(tuple(share_counts.groupby([self.platform_col,self.community_col])['value'].mean().reset_index().groupby(self.community_col)))
            share_counts = {k:v[[self.platform_col,'value']] for k,v in share_counts.items()}
        else:
            share_counts = share_counts.groupby(self.platform_col)['value'].mean().reset_index()

        return share_counts

    def temporal_correlation(self, measure="share", time_granularity="D", nodes=None, communities=None):
        """
        Calculates the correlation between the activity over time between all pairs of platforms
                Github | Reddit | Twitter
        ---------------------------------
        Github | 1.0
        ---------------------------------
        Reddit |           1.0
        ---------------------------------
        Twitter|                   1.0
        ---------------------------------
        :param measure: What to measure: number of shares or audience growth?
        :param time_granularity: The scale on which to aggregate activity {S=seconds, M=minutes, H=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix of pearson correlations between platforms.
                    Else, a dictionary mapping a community/content to the matrix of correlations
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        def get_array(content_diction):
            arrays = {plat: np.zeros((len(content_diction.keys()))) for plat in platforms}
            index = 0
            for _, plats in content_diction.items():
                for p, value in plats.items():
                    arrays[p][index] = value
                index += 1
            return arrays

        platforms = sorted(data[self.platform_col].unique())
        if time_granularity == "D":
            data[self.timestamp_col] = data[self.timestamp_col].apply(
                lambda x: '{year}-{month:02}-{day}'.format(year=x.year, month=x.month, day=x.day))
        elif time_granularity == "H":
            data[self.timestamp_col] = data[self.timestamp_col].apply(
                lambda x: '{year}-{month:02}-{day}:{hour}'.format(year=x.year, month=x.month, day=x.day, hour=x.hour))
        elif time_granularity == "M":
            data[self.timestamp_col] = data[self.timestamp_col].apply(
                lambda x: '{year}-{month:02}-{day}:{hour}:{min}'.format(year=x.year, month=x.month,
                                                                        day=x.day, hour=x.hour, min=x.minute))
        time_interval = data[self.timestamp_col].unique()
        if len(nodes) > 0:
            group_col = self.content_col
        if len(communities) > 0:
            group_col = self.community_col
        content_over_time = {}
        if len(nodes) == 0 and len(communities) == 0:  # Population level
            if measure == "share":
                content_over_time = data.groupby(self.timestamp_col).apply(
                    lambda x: x[self.platform_col].value_counts().to_dict()).to_dict()
            else:
                content_over_time = data.groupby(self.timestamp_col).apply(
                    lambda x: x.groupby(self.platform_col).apply(lambda y: len(y[self.user_col].unique())).to_dict()).to_dict()

            all_platforms = get_array(content_over_time)
            pl_1, pl_2, val = [], [], []
            for i, (p1, t1) in enumerate(all_platforms.items()):
                for j, (p2, t2) in enumerate(all_platforms.items()):
                    pearson_corr = pearsonr(t1, t2)
                    pl_1.append(p1)
                    pl_2.append(p2)
                    val.append(pearson_corr[0])
            matrix = pd.DataFrame({"platform_1": pl_1, "platform_2": pl_2, "value": val})
            return matrix

        else:
            for idx, group in data.groupby(group_col):
                if measure == "share":
                    platform_over_time = data.groupby(self.timestamp_col).apply(
                        lambda x: x[self.platform_col].value_counts().to_dict()).to_dict()
                else:
                    platform_over_time = data.groupby(self.timestamp_col).apply(
                        lambda x: x.groupby(self.platform_col).apply(lambda y: len(y[self.user_col].unique())).to_dict()).to_dict()
                content_over_time[idx] = platform_over_time
                for t in time_interval:
                    try:
                        _ = content_over_time[idx][t]
                    except KeyError:
                        content_over_time[idx][t] = {plat: 0 for plat in platforms}

            all_platforms = {}
            for content, diction in content_over_time.items():
                all_platforms[content] = get_array(diction)
            content_to_correlation = {}
            for c, times in all_platforms.items():
                pl_1, pl_2, val = [], [], []
                for i, (p1, t1) in enumerate(times.items()):
                    for j, (p2, t2) in enumerate(times.items()):
                        pearson_corr = pearsonr(t1, t2)
                        pl_1.append(p1)
                        pl_2.append(p2)
                        val.append(pearson_corr[0])
                content_to_correlation[c] = pd.DataFrame({"platform_1": pl_1, "platform_2": pl_2, "value": val})
            return content_to_correlation

    def lifetime_of_spread(self, nodes=[], communities=None):
        """
        Ranks the different platforms based on the lifespan of content/community/population
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a DataFrame (columns = platform, value)
                If community, a  dictionary mapping each community to a DataFrame (columns= platform, value)
                If nodes, returns a dictionary mapping each piece of information to a ranked list of platforms
                    (by longest lifespan).
                        Ex: {info_1: [github, twitter, reddit],
                            info_2: [reddit, twitter], ... }
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        def lifetime(grp, indx, col):
            aud = grp.groupby(col).apply(lambda x: (x[self.timestamp_col].max() - x[self.timestamp_col].min()).seconds).to_dict()
            return [item[indx] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]

        def lifetime_distributions(grp):
            return grp.groupby(self.content_col).apply(lambda x: (x[self.timestamp_col].max() - x[self.timestamp_col].min()).seconds).tolist()

        if len(nodes) == 0 and len(communities) == 0:
            lifetimes = []
            plats = []
            for inx, group in data.groupby(self.platform_col):
                life = lifetime_distributions(group)
                lifetimes.extend(life)
                plats.extend([inx]*len(life))
            return pd.DataFrame({"platform": plats, "value": lifetimes})
        elif len(nodes) > 0:
            return data.groupby(self.content_col).apply(lifetime, indx=0, col=self.platform_col).to_dict()
        elif len(communities) > 0:
            community_diction = {}
            for com_idx, community in data.groupby(self.community_col):
                lifetimes = []
                plats = []
                for inx, group in community.groupby(self.platform_col):
                    life = lifetime_distributions(group)
                    lifetimes.extend(life)
                    plats.extend([inx] * len(life))
                community_diction[com_idx] = pd.DataFrame({"platform": plats, "value": lifetimes})
            return community_diction

    def correlation_of_information(self, measure="share", communities=None):
        """
        Compute Pearson correlation
        1. Correlation between shares of information across platforms
        2. Correlation between audience sizes
        3. Correlation between lifetimes of information pieces across platforms
        4. Correlation between speeds of information across platforms
        :param measure: What to measure: number of share, audience, lifetime, or speed?
        :param communities: List of communities
        :return: If population, a matrix of correlations between all platforms based on the measure provided
                If community, a dictionary mapping each community to a matrix of correlations between all platforms
                    based on the measure provided.
        """

        if communities is None:
            communities = self.community_list
        data = self.select_data(communities=communities)

        platforms = sorted(data[self.platform_col].unique())
        if len(communities) > 0:
            group_col = self.community_col
        else:
            group_col = self.platform_col

        def get_measurement(grp, ix, plats):
            for j, (idx, sub_group) in enumerate(grp.groupby(self.content_col)):
                if measure == "share":
                    count = len(sub_group)
                elif measure == "audience":
                    count = len(sub_group[self.user_col].unique())
                elif measure == "lifetime":
                    count = pd.Timedelta(
                        sub_group[self.timestamp_col].max() - sub_group[self.timestamp_col].min()).seconds
                elif measure == "speed":
                    denominator = pd.Timedelta(
                        sub_group[self.timestamp_col].max() - sub_group[self.timestamp_col].min()).seconds
                    if denominator == 0:  # Happens when a piece of content only appears once on a platform
                        count = 0
                    else:
                        count = len(sub_group) / (pd.Timedelta(
                            sub_group[self.timestamp_col].max() - sub_group[self.timestamp_col].min()).seconds)
                else:
                    print("ERROR: Not a valid correlation option. Choices are: share, audience, lifetime, speed.")
                    count = 0
                plats[ix][j] = count
            return plats

        plat_counts = {plat: np.zeros((len(data))) for plat in platforms}
        community_to_plat = {}
        for index, group in data.groupby(group_col):
            if len(communities) > 0:
                community_to_plat[index] = plat_counts
                for plat_index, plat_group in group.groupby(self.platform_col):
                    community_to_plat[index] = get_measurement(plat_group, plat_index, community_to_plat[index])
            else:
                plat_counts = get_measurement(group, index, plat_counts)

        if len(communities) > 0:
            community_correlations = {}
            for comm, plat_diction in community_to_plat.items():
                pl_1, pl_2, val = [], [], []
                for i, (p1, t1) in enumerate(plat_diction.items()):
                    for j, (p2, t2) in enumerate(plat_diction.items()):
                        pearson_corr = pearsonr(t1, t2)
                        pl_1.append(p1)
                        pl_2.append(p2)
                        val.append(pearson_corr[0])
                community_correlations[comm] = pd.DataFrame({"platform_1": pl_1, "platform_2": pl_2, "value": val})
            return community_correlations
        else:
            pl_1, pl_2, val = [], [], []
            for i, (p1, t1) in enumerate(plat_counts.items()):
                for j, (p2, t2) in enumerate(plat_counts.items()):
                    pearson_corr = pearsonr(t1, t2)
                    pl_1.append(p1)
                    pl_2.append(p2)
                    val.append(pearson_corr[0])

            result = pd.DataFrame({"platform_1": pl_1, "platform_2": pl_2, "value": val})

            return result
