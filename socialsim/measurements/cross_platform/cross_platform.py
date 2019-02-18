import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from collections import Counter, defaultdict

from ..measurements import MeasurementsBaseClass


# from measurements import MeasurementsBaseClass
# TODO:
#   5. (TEST) Which platforms have the largest audience for the information?
#   6. (TEST) On which platforms does the information spread fastest?
#   9. (TEST) Do different platforms show similar temporal patterns of audience growth?
#   10. (TEST) In what order does the content typically reach each platform?
#   11. (TEST) What is the distribution of time elapsed between platforms?
#   16. (TEST) Are the same pieces of information receiving the most shares on different platforms?
#   17. (TEST) Are the same pieces of information reaching the largest audiences on different platforms?
#   18. (TEST) Are the same pieces of information living longest on different platforms?
#   19. (TEST) Are the same pieces of information spreading fastest on different platforms?
#   Run a single measurement without a config


class CrossPlatformMeasurements(MeasurementsBaseClass):
    # class CrossPlatformMeasurements():
    def __init__(self, dataset, configuration=None, platform="platform", parent_node_col="parentID", node_col="nodeID",
                 root_node_col="rootID", timestamp_col="nodeTime", user_col="nodeUserID", content_col="content",
                 community=None, audience="audience", log_file='cross_platform_measurements_log.txt'):
        super(CrossPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)
        # super(CrossPlatformMeasurements, self).__init__()
        self.dataset = dataset
        self.root_node_col = root_node_col
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.platform_col = platform
        self.content_col = content_col
        self.community_col = community
        self.audience_col = audience

    def select_data(self, nodes=[], communities=[]):
        """
        Subset the data based on the given communities or pieces of content
        :param data: DataFrame of all given data
        :param nodes: List of specific content
        :param communities: List of communities
        :return: New DataFrame with the select communities/content only
        """
        if len(nodes) > 0:
            data = self.dataset.loc[self.dataset[self.content_col].isin(nodes)]
        elif len(communities) > 0:
            data = self.dataset[self.dataset[self.community_col].isin(communities)]
        else:
            data = self.dataset.copy()
        return data

    def order_of_spread(self, nodes=[], communities=[]):
        """
        Determine the order of spread between platforms of a community/content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a dictionary mapping each platform to an array of percent freq. in each rank
                Else, a dictionary mapping between the community/content to the ranked list of platforms
        """
        data = self.select_data(nodes, communities)
        platforms = sorted(data[self.platform_col].unique())
        if len(communities) > 0:
            group_col = self.community_col
        else:
            group_col = self.content_col
        data.drop_duplicates(subset=[group_col, self.platform_col], inplace=True)
        data = data.groupby(group_col).apply(lambda x: x[self.platform_col].values)
        keywords_to_order = data.to_dict()
        if len(nodes) == 0 and len(communities) == 0:
            platform_diction = {p: np.zeros((len(platforms))) for p in platforms}
            for k, v in keywords_to_order.items():
                for p in platforms:
                    position,  = np.where(v == p)
                    # position = v.index(p)
                    platform_diction[p][position] += 1
            for k, v in platform_diction.items():
                platform_diction[k] = v / v.sum(axis=0)
            keywords_to_order = platform_diction
        return keywords_to_order

    def time_delta(self, time_granularity="s", nodes=[], communities=[]):
        """
        Determine the amount of time it takes for a community/content to appear on another platform
        :param time_granularity: Unit of time to calculate {s=seconds, m=minutes, h=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix (i x j) of distributions of the time passed since platform i to platform j
                    Else, a dictionary mapping a community/content to a list of the time passed since the first time
                    that community/content was observed
        """

        data = self.select_data(nodes, communities)
        data = data.sort_values(self.timestamp_col)
        platforms = sorted(data[self.platform_col].unique())
        if len(communities) > 0:
            group_col = [self.community_col, self.platform_col]
        else:
            group_col = [self.content_col, self.platform_col]
        data.drop_duplicates(subset=group_col, inplace=True)
        data.sort_values(by=[self.platform_col], inplace=True)
        # time_delta = {}
        data = data.groupby(group_col[0]).apply(lambda x: x[self.timestamp_col].values)
        time_delta = data.to_dict()
        # for index, group in data.groupby(group_col[0]):
        #     time_delta[index] = group[self.timestamp_col].values
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

        data = self.select_data(nodes, communities)
        platforms = sorted(data[self.platform_col].unique())
        data['values'] = 1
        data = data.drop_duplicates(subset=[self.user_col, self.platform_col])
        cols = [self.user_col, self.platform_col, 'values']
        index_cols = [self.user_col]
        if len(communities) > 0:
            cols = [self.user_col, self.platform_col, self.community, 'values']
            index_cols = [self.user_col, self.community]
            group_col = self.community
        if len(nodes) > 0:
            cols = [self.user_col, self.platform_col, self.content_col, 'values']
            index_cols = [self.user_col, self.content_col]
            group_col = self.content_col

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

        :param nodes:
        :param communities:
        :return:
        """
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
            aud = data.groupby(group_col).apply(lambda x: x[self.audience_col].sum()).to_dict()
            return [item[0] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]
        else:
            return data.groupby(group_col).apply(audience).to_dict()

    def speed_of_spread(self, time_granularity="s", nodes=[], communities=[]):
        """

        :param nodes:
        :param communities:
        :return:
        """
        data = self.select_data(nodes, communities)
        if len(communities) > 0:
            group_col = self.community_col
        else:
            group_col = self.content_col

        # data.groupby([group_col, self.platform_col])[self.timestamp_col].agg(np.ptp)

        def audience_over_time(grp):
            aud = grp.groupby(self.platform_col).apply(lambda x: x[self.audience_col].sum() / (
                    x[self.timestamp_col].max() - x[self.timestamp_col].min()).seconds).to_dict()
            return [item[0] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]

        if len(nodes) == 0 and len(communities) == 0:
            aud = data.groupby(self.platform_col).apply(lambda x: x[self.audience_col].sum() / (
                    x[self.timestamp_col].max() - x[self.timestamp_col].min()).seconds).to_dict()
            return [item[0] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]
        else:
            return data.groupby(group_col).apply(audience_over_time).to_dict()

    def size_of_shares(self, nodes=[], communities=[]):
        """
        Determine the number of shares per platform
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a ranked list of platforms based on total activity
                Else, a dictionary mapping the community/content to a ranked list of platforms based on activity
        """
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
            plat_counts[group[group_col].values[0]] = [(item[0], item[1]) for item in
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
        data = self.select_data(nodes, communities)
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
            # for index, group in data.groupby(self.timestamp_col):
            #     content_over_time[group[self.timestamp_col].values[0]] = {}
            #     content_over_time[group[self.timestamp_col].values[0]] = group[
            #         self.platform_col].value_counts().to_dict()
        else:
            for idx, group in data.groupby(group_col):
                if measure == "share":
                    platform_over_time = data.groupby(self.timestamp_col).apply(
                        lambda x: x[self.platform_col].value_counts().to_dict()).to_dict()
                else:
                    platform_over_time = data.groupby(self.timestamp_col).apply(
                        lambda x: x.groupby(self.platform_col)[self.audience_col].sum().to_dict()).to_dict()
                content_over_time[idx] = platform_over_time
                # content_over_time[group[group_col].values[0]] = {}
                # for index, subgroup in group.groupby(self.timestamp_col):
                #     content_over_time[idx][subgroup[index]] = subgroup[self.platform_col].value_counts().to_dict()
                for t in time_interval:
                    try:
                        _ = content_over_time[idx][t]
                    except KeyError:
                        content_over_time[idx][t] = {plat: 0 for plat in platforms}

        def get_array(content_diction):
            arrays = {plat: np.zeros((len(content_diction.keys()))) for plat in platforms}
            index = 0
            for time, plats in content_diction.items():
                for p, value in plats.items():
                    arrays[p][index] = value
                index += 1
            return arrays

        if len(nodes) == 0 and len(communities) == 0:
            all_platforms = get_array(content_over_time)
            matrix = np.zeros((len(platforms), len(platforms)))
            for i, (p1, t1) in enumerate(all_platforms.items()):
                for j, (p2, t2) in enumerate(all_platforms.items()):
                    pearson_corr = pearsonr(t1, t2)
                    matrix[i][j] = pearson_corr[0]
            return matrix
        else:
            all_platforms = {}
            for content, diction in content_over_time.items():
                all_platforms[content] = get_array(diction)
            content_to_correlation = {}
            for c, times in all_platforms.items():
                matrix = np.zeros((len(platforms), len(platforms)))
                for i, (p1, t1) in enumerate(times.items()):
                    for j, (p2, t2) in enumerate(times.items()):
                        pearson_corr = pearsonr(t1, t2)
                        matrix[i][j] = pearson_corr[0]
                content_to_correlation[c] = matrix
            return content_to_correlation

    def lifetime_of_spread(self, nodes=[], communities=[]):
        """
        Ranks the different platforms based on the lifespan of content/community/population
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, returns a list of ranked platforms.
                Else, returns a dictionary of content/community to a ranked list of platforms
        """

        data = self.select_data(nodes, communities)
        platforms = sorted(data[self.platform_col].unique())
        print(platforms)
        if len(nodes) == 0 and len(communities) == 0:
            ranks = {plat: 0 for plat in platforms}
            for index, group in data.sort_values([self.timestamp_col], ascending=True).groupby(self.platform_col):
                ranks[index] = pd.Timedelta(
                    group[self.timestamp_col].iloc[-1] - group[self.timestamp_col].iloc[0]).seconds
            return [item[0] for item in sorted(ranks.items(), reverse=True, key=lambda kv: kv[1])]
        else:
            if len(nodes) > 0:
                group_col = self.content_col
            elif len(communities) > 0:
                group_col = self.community_col
            content_to_rank = {}
            for index, group in data.sort_values([self.timestamp_col], ascending=True).groupby(group_col):
                content_to_rank[index] = {plat: 0 for plat in platforms}
                for index2, subgroup in group.groupby(self.platform_col):
                    content_to_rank[index][subgroup[self.platform_col].values[0]] = pd.Timedelta(
                        subgroup[self.timestamp_col].iloc[-1] - subgroup[self.timestamp_col].iloc[0]).seconds
                content_to_rank[index] = [item[0] for item in sorted(
                    content_to_rank[index].items(), reverse=True, key=lambda kv: kv[1])]
            return content_to_rank

    def correlation_of_information(self, measure="share", communities=[]):
        """
        Compute Pearson correlation
        1. Correlation between shares of information across platforms
        2. Correlation between audience sizes
        3. Correlation between lifetimes of information pieces across platforms
        4. Correlation between speeds of information across platforms
        :param measure: What to measure: number of share, audience, lifetime, or speed?
        :param communities: List of communities
        :return:
        """
        data = self.select_data(communities=communities)
        platforms = sorted(data[self.platform_col].unique())
        if len(communities) > 0:
            group_col = [self.community_col, self.platform_col]
        else:
            group_col = [self.content_col, self.platform_col]
        plat_counts = {plat: np.zeros((len(data))) for plat in platforms}
        for i, (index, group) in enumerate(data.groupby(group_col)):
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
                    print(group[self.timestamp_col].max())
                    print(group[self.timestamp_col].min())
            else:
                print("ERROR: Not a valid correlation option. Choices are: share, audience, lifetime, speed.")
            plat = group[self.platform_col].values[0]
            plat_counts[plat][i] = count
        matrix = np.zeros((len(platforms), len(platforms)))
        for i, (p1, t1) in enumerate(plat_counts.items()):
            for j, (p2, t2) in enumerate(plat_counts.items()):
                pearson_corr = pearsonr(t1, t2)
                matrix[i][j] = pearson_corr[0]
        return matrix

