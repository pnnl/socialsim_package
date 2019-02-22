import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

from .measurements import MeasurementsBaseClass

"""
Checklist: 
- order_of_spread: (1,10) 
- time_delta: (2, 11)
- overlapping_users: (3, 12)
- size_of_audience: (5)
- speed_of_spread: (6, 14)
- temporal_correlation: (8, 9)
- size_of_shares: (4, 13)
- lifetime_of_spread: (7, 15)
- correlation_of_information: (16,17,18,19)

Questions for Emily:
    1. Lifetime/speed of the community = first appearance of any content in community to last or average 
        lifetime of each content
            - If average, should they be normalized 
    2. Should community correlations be returning a dictionary of community to matrix of correlations? Or one matrix
        for all communities?
        
Changes to make:
    1. Create community copy if given a list of communities. 
        To be built from the metadata (a dictionary: community to a list of nodes in that community)
"""


class CrossPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, meatadata=None, communities=None, 
        platform_col="platform", timestamp_col="nodeTime", 
        user_col="nodeUserID", content_col="informationID", 
        community_col="community", audience_col="audience",
        log_file='cross_platform_measurements_log.txt', node_list=None, 
        community_list=None):
        """

        :param dataset: dataframe containing all pieces of content and associated data, sorted by time
        :param configuration:
        :param platform_col: name of the column containing the platforms
        :param timestamp_col: name of the column containg the time of the content
        :param user_col: name of the column containing the user ids of each piece of content
        :param content_col: name of the column where the content can be found
        :param community_col: name of the column containing the subset of content in a community
        :param audience_col: name of the column containing the audience amount each content piece has reached
        :param log_file:
        """
        super(CrossPlatformMeasurements, self).__init__(dataset, configuration, 
            log_file=log_file)

        self.dataset            = dataset
        self.community_set      = communities
        self.timestamp_col      = timestamp_col
        self.user_col           = user_col
        self.platform_col       = platform_col
        self.content_col        = content_col
        self.community_col      = community_col
        self.audience_col       = audience_col

        self.measurement_type = 'cross_platform'

        if node_list == "all":
            self.node_list = self.dataset[self.content_col].tolist()
        elif node_list is not None:
            self.node_list = node_list
        else:
            self.node_list = []

        if communities is not None:
            if community_list == "all":
                self.community_list = self.community_set[self.community_col]
                self.community_list = self.community_list.unique()
                print(self.community_list)
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

    def order_of_spread(self, nodes=[], communities=[]):
        """
        Determine the order of spread between platforms of a community/content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a dictionary mapping each platform to an array of percent freq. in each rank
                If community, a nested dictionary mapping each community to a dictionary mapping each platform
                    to an array of percent freq. in each rank
                Else, a dictionary mapping between the content to the ranked list of platforms
        """

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
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
            return plat_diction

        if len(communities) > 0:
            data.drop_duplicates(subset=[self.community_col, self.content_col, self.platform_col], inplace=True)
            data = data.groupby(self.community_col).apply(
                lambda x: x.groupby(self.content_col).apply(lambda y: y[self.platform_col].values).to_dict())
            community_platform_order = data.to_dict()
            keywords_to_order = {}
            for comm, content_diction in community_platform_order.items():
                keywords_to_order[comm] = platform_order(content_diction)
        else:
            data.drop_duplicates(subset=[self.content_col, self.platform_col], inplace=True)
            data = data.groupby(self.content_col).apply(lambda x: x[self.platform_col].values)
            keywords_to_order = data.to_dict()
            if len(nodes) == 0 and len(communities) == 0:
                keywords_to_order = platform_order(keywords_to_order)
        return keywords_to_order

    def time_delta(self, time_granularity="s", nodes=[], communities=[]):
        """
        Determine the amount of time it takes for a community/content to appear on another platform
        :param time_granularity: Unit of time to calculate {s=seconds, m=minutes, h=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix (i x j) of distributions of the time passed since platform i to platform j
                If community, a dictionary mapping a community to a matrix (i x j) of distributions of the time passed
                    since platform i to platform j
                Else, a dictionary mapping a content to a list (equal in length to the number of platforms)
                of the time passed since the first time that content was observed. This list is sorted alphabetically
                to preserve the same ordering of platforms in all cases. This can causes negative times.
        """

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        data = data.sort_values(self.timestamp_col)
        platforms = sorted(data[self.platform_col].unique())

        if len(communities) > 0:
            group_col = [
                self.community_col, 
                self.content_col, 
                self.platform_col
                ]

            data.drop_duplicates(subset=group_col, inplace=True)
            data.sort_values(by=[self.platform_col], inplace=True)

            # Can some of these lambda functions be turned into full functions 
            # for readability?
            data = data.groupby(self.community_col).apply(lambda x: x.groupby(self.content_col).apply(lambda y: y[self.timestamp_col].values).to_dict())

            time_delta = data.to_dict()
            delta = {}
            for comm, content_diction in time_delta.items():
                temp = {}
                for k, v in content_diction.items():
                    temp[k] = [0]
                    for i in v[1:]:
                        temp[k].append((np.datetime64(i) - np.datetime64(v[0])) / np.timedelta64(1, time_granularity))
                matrix = [[[]] * len(platforms)] * len(platforms)
                for k, v in temp.items():
                    for i in range(len(v)):
                        for j in range(len(v)):
                            if i != j:
                                if v[i] < v[j]:
                                    matrix[i][j].append(abs(v[j] - v[i]))
                delta[comm] = matrix
            return delta
        else:
            group_col = [self.content_col, self.platform_col]
            data.drop_duplicates(subset=group_col, inplace=True)
            data.sort_values(by=[self.platform_col], inplace=True)
            data = data.groupby(self.content_col).apply(lambda x: x[self.timestamp_col].values)
            time_delta = data.to_dict()
            delta = {}
            for k, v in time_delta.items():
                delta[k] = [0]
                for i in v[1:]:
                    delta[k].append((np.datetime64(i) - np.datetime64(v[0])) / np.timedelta64(1, time_granularity))
            if len(nodes) == 0 and len(communities) == 0:
                matrix = [[[]] * len(platforms)] * len(platforms)
                for k, v in delta.items():
                    for i in range(len(v)):
                        for j in range(len(v)):
                            if i != j:
                                if v[i] < v[j]:
                                    matrix[i][j].append(abs(v[j] - v[i]))
                return matrix
            else:
                return delta

    def overlapping_users(self, nodes=[], communities=[]):
        """
        Calculate the percentage of users common to all platforms (that share in a community/content)
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix of percentages of common users to any pair of platforms.
                Else, a dictionary mapping a community/content to a matrix of the percentages of common users that
                share in that community/content across all pairs of platforms
        """

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
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

        def get_meas(grp):
            meas = np.zeros((len(platforms), len(platforms)))
            for i, p1 in enumerate(platforms):
                for j, p2 in enumerate(platforms):
                    if p1 == p2:
                        x = 1.0
                    else:
                        x = (grp[p1] & grp[p2]).sum()
                        total = float(grp[p1].sum())
                        if total > 0:
                            x = x / total
                        else:
                            x = 0
                    meas[i][j] = x
            return meas

        if len(nodes) != 0 or len(communities) != 0:
            user_platform = user_platform.groupby(group_col)
            meas = user_platform.apply(get_meas).to_dict()
        else:
            meas = get_meas(user_platform)
        return meas

    def size_of_audience(self, nodes=[], communities=[]):
        """
        Determine the ranking of audience sizes on each platform
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a ranked list of the platforms with the largest audience sizes
                Else, a dictionary mapping the community/content to a ranked list of the platforms with the largest
                    audience sizes.
        """

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        if len(communities) > 0:
            group_col = self.community_col
        elif len(nodes) > 0:
            group_col = self.content_col
        else:
            group_col = self.platform_col

        def audience(grp):
            aud = grp.groupby(self.platform_col).apply(lambda x: x[self.audience_col].sum()).to_dict()
            return [(item[0],item[1]) for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]

        if len(nodes) == 0 and len(communities) == 0:
            return audience(data)
        else:
            return data.groupby(group_col).apply(audience).to_dict()

    def speed_of_spread(self, nodes=[], communities=[]):
        """
        Determine the speed at which the information is spreading
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a list of distributions of information speeds on each platform. Ordered alphabetically
                    for consistency.
                If community, a dictionary mapping each community to a list of distributions of information speeds on
                    each platform. Ordered alphabetically for consistency.
        a ranked list of the platforms that typically spread information the fastest. Ranks
                    are determined by the average speed of each piece of content found on that platform.
                If community, a dictionary mapping each community to a ranked list of the platfroms that spread the
                    information found in the community the fastest.
                Else, a dictionary mapping each content to the ranked list of platforms on which it spreads the fastest
        """
        #print(data.groupby("community").apply(lambda x: x.groupby("platform").apply(lambda y: y.groupby("content").apply(check_zero_speed).tolist()).tolist()).to_dict())

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        # Looks like all of these are being defined every time the function is 
        # called. Can we make them a little more general and define them once?
        def check_zero_speed(row):
            time = (row[self.timestamp_col].max() - row[self.timestamp_col].min()).seconds
            if time == 0:
                speed = -1
            else:
                speed = row[self.audience_col].sum() / time
            return speed

        def audience_over_time(grp, indx, col):
            aud = grp.groupby(col).apply(check_zero_speed).to_dict()
            return [item[indx] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]

        # Can't find where this function is called. Do we need it?
        def speed_distribution(grp):
            aud = grp.groupby(self.platform_col).apply(lambda x: x.groupby(self.content_col).apply(check_zero_speed).tolist()).tolist()
            return aud

        def average_audience_speed(grp):
            aud = grp.groupby(self.platform_col).apply(audience_over_time, indx=1, col=self.content_col).to_dict()
            return [item[0] for item in sorted(aud.items(), reverse=True, key=lambda kv: np.mean(kv[1]))]

        if len(communities) > 0:
            return data.groupby(self.community_col).apply(average_audience_speed).to_dict()
        elif len(nodes) > 0:
            return data.groupby(self.content_col).apply(audience_over_time, indx=0, col=self.platform_col).to_dict()
        else:
            data = data.groupby(self.platform_col).apply(audience_over_time, indx=1, col=self.content_col).to_dict()
            return [item[0] for item in sorted(data.items(), reverse=True, key=lambda kv: np.mean(kv[1]))]

    def size_of_shares(self, nodes=[], communities=[]):
        """
        Determine the number of shares per platform
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a ranked list of platforms based on total activity
                Else, a dictionary mapping the community/content to a ranked list of platforms based on activity
        """

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        if len(nodes) == 0 and len(communities) == 0:
            plat_counts = data[self.platform_col].value_counts().to_dict()
            return [item[0] for item in sorted(plat_counts.items(), reverse=True, key=lambda kv: kv[1])]
        elif len(nodes) > 0:
            group_col = self.content_col
        elif len(communities) > 0:
            group_col = self.community_col
        plat_counts = {}
        for index, group in data.groupby(group_col):
            diction = group[self.platform_col].value_counts().to_dict()
            plat_counts[index] = [(item[0], item[1]) for item in
                                  sorted(diction.items(), reverse=True, key=lambda kv: kv[1])]
        return plat_counts

    def temporal_correlation(self, measure="share", time_granularity="D", nodes=[], communities=[]):
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

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
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
                    lambda x: x.groupby(self.platform_col)[self.audience_col].sum().to_dict()).to_dict()

            all_platforms = get_array(content_over_time)
            matrix = np.zeros((len(platforms), len(platforms)))
            for i, (_, t1) in enumerate(all_platforms.items()):
                for j, (_, t2) in enumerate(all_platforms.items()):
                    pearson_corr = pearsonr(t1, t2)
                    matrix[i][j] = pearson_corr[0]
            return matrix

        else:
            for idx, group in data.groupby(group_col):
                if measure == "share":
                    platform_over_time = data.groupby(self.timestamp_col).apply(
                        lambda x: x[self.platform_col].value_counts().to_dict()).to_dict()
                else:
                    platform_over_time = data.groupby(self.timestamp_col).apply(
                        lambda x: x.groupby(self.platform_col)[self.audience_col].sum().to_dict()).to_dict()
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
                matrix = np.zeros((len(platforms), len(platforms)))
                for i, (_, t1) in enumerate(times.items()):
                    for j, (_, t2) in enumerate(times.items()):
                        pearson_corr = pearsonr(t1, t2)
                        matrix[i][j] = pearson_corr[0]
                content_to_correlation[c] = matrix
            return content_to_correlation

    def lifetime_of_spread(self, nodes=[], communities=[]):
        """
        Ranks the different platforms based on the lifespan of content/community/population
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, returns a list of ranked platforms based on the average lifetime of all content on a
                    platform
                If community, a dictionary mapping the community to a list of ranked platforms based on the average
                    lifetime of all content in the community on a platform
                Else, returns a dictionary of content to a ranked list of platforms
        """

        if len(nodes) == 0:
            nodes = self.node_list
        if len(communities) == 0:
            communities = self.community_list
        data = self.select_data(nodes, communities)

        # Looks like platforms is never used. Should it be? Can we remove this line?
        platforms = sorted(data[self.platform_col].unique())

        def lifetime(grp, indx, col):
            aud = grp.groupby(col).apply(lambda x: (x[self.timestamp_col].max() - x[self.timestamp_col].min()).seconds).to_dict()
            return [item[indx] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]

        def average_lifetime(grp):
            aud = grp.groupby(self.platform_col).apply(lifetime, indx=1, col=self.content_col).to_dict()
            return [item[0] for item in sorted(aud.items(), reverse=True, key=lambda kv: np.mean(kv[1]))]

        if len(nodes) == 0 and len(communities) == 0:
            data = data.groupby(self.platform_col).apply(lifetime, indx=1, col=self.content_col).to_dict()
            return [item[0] for item in sorted(data.items(), reverse=True, key=lambda kv: np.mean(kv[1]))]
        elif len(nodes) > 0:
            return data.groupby(self.content_col).apply(lifetime, indx=0, col=self.platform_col).to_dict()
            # content_to_rank = {}
            # data = data.sort_values([self.timestamp_col], ascending=True)
            # for index, group in data.groupby(self.content_col):
            #     content_to_rank[index] = {plat: 0 for plat in platforms}
            #     for index2, subgroup in group.groupby(self.platform_col):
            #         content_to_rank[index][index2] = pd.Timedelta(
            #             subgroup[self.timestamp_col].iloc[-1] - subgroup[self.timestamp_col].iloc[0]).seconds
            #     content_to_rank[index] = [item[0] for item in sorted(
            #         content_to_rank[index].items(), reverse=True, key=lambda kv: kv[1])]
            # return content_to_rank

        elif len(communities) > 0:
            return data.groupby(self.community_col).apply(average_lifetime).to_dict()

    def correlation_of_information(self, measure="share", communities=[]):
        """
        Compute Pearson correlation
        1. Correlation between shares of information across platforms
        2. Correlation between audience sizes
        3. Correlation between lifetimes of information pieces across platforms
        4. Correlation between speeds of information across platforms
        :param measure: What to measure: number of share, audience, lifetime, or speed?
        :param communities: List of communities
        :return: A matrix of correlations between all platforms based on the measure provided
        """

        if len(communities) == 0:
            communities = self.community_list
        data = self.select_data(communities=communities)

        platforms = sorted(data[self.platform_col].unique())
        if len(communities) > 0:
            group_col = [self.community_col, self.platform_col]
        else:
            group_col = [self.content_col, self.platform_col]

        plat_counts = {plat: np.zeros((len(data))) for plat in platforms}
        for i, (_, group) in enumerate(data.groupby(group_col)):
            count = 0
            if measure == "share":
                count = len(group)
            elif measure == "audience":
                count = group[self.audience_col].sum()
            elif measure == "lifetime":
                count = pd.Timedelta(group[self.timestamp_col].iloc[-1] - group[self.timestamp_col].iloc[0]).seconds
            elif measure == "speed":
                count = group[self.audience_col].sum() / (pd.Timedelta(
                    group[self.timestamp_col].max() - group[self.timestamp_col].min()).seconds)
                if count == float("inf"):  # Happens when a piece of content only appears once on a platform
                    count = 0
            else:
                print("ERROR: Not a valid correlation option. Choices are: share, audience, lifetime, speed.")
            plat = group[self.platform_col].values[0]
            plat_counts[plat][i] = count
        matrix = np.zeros((len(platforms), len(platforms)))
        for i, (_, t1) in enumerate(plat_counts.items()):
            for j, (_, t2) in enumerate(plat_counts.items()):
                pearson_corr = pearsonr(t1, t2)
                matrix[i][j] = pearson_corr[0]
        return matrix

