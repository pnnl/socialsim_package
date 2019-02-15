import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from collections import Counter, defaultdict


# from ..measurements import MeasurementsBaseClass
# from measurements import MeasurementsBaseClass
# TODO:
#   5. (Audience) Which platforms have the largest audience for the information?
#   6. (Speed) On which platforms does the information spread fastest?
#   9. (Audience) Do different platforms show similar temporal patterns of audience growth?
#   16. (Size) Are the same pieces of information receiving the most shares on different platforms?
#   17. (Audience) Are the same pieces of information reaching the largest audiences on different platforms?
#   18. (Lifetime) Are the same pieces of information living longest on different platforms?
#   19. (Speed) Are the same pieces of information spreading fastest on different platforms?


# class CrossPlatformMeasurements(MeasurementsBaseClass):
class CrossPlatformMeasurements():
    def __init__(self, dataset, configuration=None, platform="platform", parent_node_col="parentID", node_col="nodeID",
                 root_node_col="rootID", timestamp_col="nodeTime", user_col="nodeUserID", content_col="content",
                 community=None, log_file='cross_platform_measurements_log.txt'):
        # super(CrossPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)
        super(CrossPlatformMeasurements, self).__init__()
        self.dataset = dataset
        self.root_node_col = root_node_col
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.platform_col = platform
        self.content_col = content_col
        self.community_col = community
        # self.content_df = pd.DataFrame()

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
            data = self.dataset
        return data

    def order_of_spread(self, nodes=[], communities=[]):
        """
        #TODO: How does this change for population?
        Determine the order of spread between platforms of a community/content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: A dictionary mapping between the community/content to the ranked list of platforms
        """
        data = self.select_data(self.dataset, nodes, communities)
        keywords_to_order = {}
        if len(nodes) > 0:
            group_col = self.content_col
        if len(communities) > 0:
            group_col = self.community_col
        data.drop_duplicates(subset=[group_col, self.platform_col], inplace=True)
        for index, group in data.groupby(group_col):
            keywords_to_order[group[group_col].values[0]] = group[self.platform_col].values
        return keywords_to_order

    def time_delta(self, time_granularity="s", nodes=[], communities=[]):
        """
        Determine the amount of time it takes for a community/content to appear on another platform
        :param time_granularity: Unit of time to calculate {s=seconds, m=minutes, h=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a list of the time passed since the first obsevered time.
                    Else, a dictionary mapping a community/content to a list of the time passed since the first time
                    that community/content was observed
        TODO: Should the node level arrays be ordered to match with population/community? (i.e. [t, g, r] always?)
        """

        data = self.select_data(self.dataset, nodes, communities)
        data = data.sort_values(self.timestamp_col)
        if len(nodes) > 0:
            group_col = [self.content_col, self.platform_col]
        elif len(communities) > 0:
            group_col = [self.community_col, self.platform_col]
        else:
            group_col = [self.platform_col]
        data.drop_duplicates(subset=group_col, inplace=True)
        if len(nodes) == 0 and len(communities) == 0:
            time_delta = []
            for index, group in data.groupby(group_col[0]):
                time_delta.append(group[self.timestamp_col])
            delta = [0]
            for t in time_delta[1:]:
                delta.append((np.datetime64(t) - np.datetime64(time_delta[0]))/np.timedelta64(1,time_granularity))
            return delta
        else:
            time_delta = {}
            for index, group in data.groupby(group_col[0]):
                time_delta[group[group_col[0]].values[0]] = group[self.timestamp_col].values
            delta = {}
            for k, v in time_delta.items():
                delta[k] = [0]
                for i in v[1:]:
                    delta[k].append((np.datetime64(i) - np.datetime64(v[0]))/np.timedelta64(1,time_granularity))
                    # delta[k].append(pd.Timedelta(i - v[0]).seconds / divide_val)
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

        data = self.select_data(self.dataset, nodes, communities)
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

    def size_of_shares(self, nodes=[], communities=[]):
        """
        Determine the number of shares per platform
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a ranked list of platforms based on total activity
                Else, a dictionary mapping the community/content to a ranked list of platforms based on activity
        """
        data = self.select_data(self.dataset, nodes, communities)
        if len(nodes) == 0 and len(communities) == 0:
            plat_counts = data[self.platform_col].value_counts().to_dict()
            return [item[0] for item in sorted(plat_counts.items(), key=lambda kv: kv[1])]
        elif len(nodes) > 0:
            group_col = self.content_col
        elif len(communities) > 0:
            group_col = self.community_col
        plat_counts = {}
        for index, group in data.groupby(group_col):
            diction = group[self.platform_col].value_counts().to_dict()
            plat_counts[group[group_col].values[0]] = [(item[0], item[1]) for item in sorted(diction.items(), reverse=True, key=lambda kv: kv[1])]
        return plat_counts

    def temporal_share_correlation(self, time_granularity="D", nodes=[], communities=[]):
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
        :param time_granularity: The scale on which to aggregate activity {S=seconds, M=minutes, H=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix of pearson correlations between platforms.
                    Else, a dictionary mapping a community/content to the matrix of correlations
        """
        data = self.select_data(self.dataset, nodes, communities)
        platforms = sorted(data[self.platform_col].unique())
        if time_granularity == "D":
            print(data[self.timestamp_col])
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
            for index, group in data.groupby(self.timestamp_col):
                content_over_time[group[self.timestamp_col].values[0]] = {}
                content_over_time[group[self.timestamp_col].values[0]] = group[self.platform_col].value_counts().to_dict()
        if len(nodes) > 0 or len(communities) > 0:
            for idx, group in data.groupby(group_col):
                content_over_time[group[group_col].values[0]] = {}
                for index, subgroup in group.groupby(self.timestamp_col):
                    content_over_time[group[group_col].values[0]][subgroup[self.timestamp_col].values[0]] = subgroup[
                        self.platform_col].value_counts().to_dict()
                for t in time_interval:
                    try:
                        _ = content_over_time[group[group_col].values[0]][t]
                    except KeyError:
                        content_over_time[group[group_col].values[0]][t] = {plat: 0 for plat in platforms}

        def get_array(diction):
            arrays = {plat: np.zeros((len(diction.keys()))) for plat in platforms}
            index = 0
            for time, plats in diction.items():
                for p, value in plats.items():
                    arrays[p][index] = value
                index += 1
            return arrays

        if len(nodes) == 0 and len(communities) == 0:
            all_platforms = get_array(content_over_time)
            matrix = np.zeros((len(platforms), len(platforms)))
            for i, (p1, t1) in enumerate(all_platforms.items()):
                for j, (p2, t2) in enumerate(all_platforms.items()):
                    pearson_corr = pearsonr(t1,t2)
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
        :return: If population, returns a list of ranked platforms. Else, returns a dictionary of content/community to
                a ranked list of platforms
        """

        data = self.select_data(self.dataset, nodes, communities)
        platforms = sorted(data[self.platform_col].unique())
        if len(nodes) == 0 and len(communities) == 0:
            ranks = {plat: 0 for plat in platforms}
            for index, group in data.sort_values([self.timestamp_col], ascending=True).groupby(self.platform_col):
                ranks[self.platform_col] = pd.Timedelta(group[self.timestamp_col].iloc[-1] - group[self.timestamp_col].iloc[0]).seconds
            return [item[0] for item in sorted(ranks.items(), reverse=True, key=lambda kv: kv[1])]
        else:
            if len(nodes) > 0:
                group_col = self.content_col
            elif len(communities) > 0:
                group_col = self.community_col
            content_to_rank = {}
            for index, group in data.sort_values([self.timestamp_col], ascending=True).groupby(group_col):
                content_to_rank[group[group_col].values[0]] = {plat: 0 for plat in platforms}
                for index2, subgroup in group.groupby(self.platform_col):
                    content_to_rank[group[group_col].values[0]][subgroup[self.platform_col].values[0]] = pd.Timedelta(
                        subgroup[self.timestamp_col].iloc[-1] - subgroup[self.timestamp_col].iloc[0]).seconds
                print(content_to_rank[group[group_col].values[0]])
                content_to_rank[group[group_col].values[0]] = [item[0] for item in sorted(
                    content_to_rank[group[group_col].values[0]].items(), reverse=True, key=lambda kv: kv[1])]
            return content_to_rank

    def correlation_of_shares(self, communities=[]):
        """

        :param node:
        :param communities:
        :return:
        """
        data = self.select_data(self.dataset, communities=communities)
        platforms = sorted(data[self.platform_col].unique())
        if len(communities) == 0:
            group_col = [self.content_col, self.platform_col]
        else:
            group_col = [self.community_col, self.platform_col]
        plat_counts = {plat: np.zeros((len(data))) for plat in platforms}
        for i, (index, group) in enumerate(data.groupby(group_col)):
            count = len(group)
            plat = group[self.platform_col].values[0]
            plat_counts[plat][i] = count
        matrix = np.zeros((len(platforms), len(platforms)))
        for i, (p1, t1) in enumerate(plat_counts.items()):
            for j, (p2, t2) in enumerate(plat_counts.items()):
                pearson_corr = pearsonr(t1,t2)
                matrix[i][j] = pearson_corr[0]



