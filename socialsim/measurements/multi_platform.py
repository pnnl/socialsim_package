import pandas as pd

from .measurements import MeasurementsBaseClass
from ..utils import add_communities_to_dataset


class MultiPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, metadata=None, platform_col="platform",
                 timestamp_col="nodeTime", user_col="nodeUserID", content_col="informationID",
                 community_col="community", node_list=None, community_list=None,
                 log_file='multi_platform_measurements_log.txt'):
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
        super(MultiPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)

        self.dataset = dataset
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.platform_col = platform_col
        self.content_col = content_col
        self.community_col = community_col

        self.measurement_type = 'multi_platform'

        if metadata is None:
            self.community_set = self.dataset
            self.community_set[self.community_col] = "Default Community"

            if node_list == "all":
                self.node_list = self.dataset[self.content_col].tolist()
            elif node_list is not None:
                self.node_list = node_list
            else:
                self.node_list = []
        else:
            community_directory = metadata.community_directory
            self.community_set = add_communities_to_dataset(dataset,
                                                            community_directory)
            self.node_list = metadata.node_list

        if self.community_set is not None:
            if community_list is not None and len(community_list) > 0:
                self.community_list = community_list
            else:
                self.community_list = self.community_set[self.community_col].unique()
        else:
            self.community_list = []

    def select_data(self, nodes=None, communities=None, platform="all"):
        """
        Subset the data based on the given communities or pieces of content
        :param nodes: List of specific content
        :param communities: List of communities
        :param platform:
        :return: New DataFrame with the select communities/content only
        """

        if nodes is None:
            nodes = []
        if communities is None:
            communities = []
        if platform == "all":
            platform_list = self.dataset[self.platform_col].unique()
        else:
            platform_list = [platform]

        if len(nodes) > 0:
            data = self.dataset.loc[(self.dataset[self.content_col].isin(nodes)) &
                                    (self.dataset[self.platform_col].isin(platform_list))]
        elif len(communities) > 0:
            data = self.community_set[(self.community_set[self.community_col].isin(communities)) &
                                      (self.community_set[self.platform_col].isin(platform_list))]
        else:
            # If platform_list is all, will this be modifying the dataframe or make a copy?
            data = self.dataset.loc[self.dataset[self.platform_col].isin(platform_list)]
            # data = self.dataset.copy()
        return data

    @staticmethod
    def get_time_diff_granularity(time_diff, granularity):

        if granularity.lower() == 's':
            return time_diff.seconds
        elif granularity.lower() == 'm':
            return time_diff.minutes
        elif granularity.lower() == 'h':
            return time_diff.hours
        elif granularity.lower() == 'd':
            return time_diff.days

    @staticmethod
    def get_time_granularity(time, time_unit):

        if time_unit.lower() == 'm':
            time /= 60.0
        if time_unit.lower() == 'h':
            time /= 60.0 * 60.0
        if time_unit.lower() == 'd':
            time /= 60.0 * 60.0 * 24.0

        return time

    def preprocess(self, node_level, nodes, community_level, communities, platform):

        if node_level and len(nodes) == 0:
            nodes = self.node_list
        elif node_level and nodes == "all":
            nodes = self.dataset[self.content_col].unique()
        elif not node_level:
            nodes = []

        if community_level and len(communities) == 0:
            communities = self.community_list
        elif community_level and communities == "all":
            communities = self.community_set[self.community_col].unique()
        elif not community_level:
            communities = []

        data = self.select_data(nodes, communities, platform=platform)

        return data

    def number_of_shares(self, node_level=False, community_level=False, nodes=None, communities=None, platform="all"):
        """
        Description: Determine the number of times a piece of information is shared

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe of each node and the number of times that info is shared
                If community level : a dictionary mapping between each community and a dataframe
                If node level: a dataframe of each node and the number of times that info is shared
        """

        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def counts(group):
            grp_count = group[self.content_col].value_counts()
            return pd.DataFrame({"node": grp_count.index, "count": grp_count.values})

        if not community_level:
            num_shares = data[self.content_col].value_counts().rename_axis("node").reset_index(name="count")
        else:
            num_shares = {}
            for com, grp in data.groupby("community"):
                num_shares[com] = counts(grp)

        return num_shares

    def number_of_shares_over_time(self, node_level=False, community_level=False, nodes=None,
                                   communities=None, platform="all"):
        """
        Description: Determine the number of times a piece of information is shared over time

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a series with the average number of shares at each time
                If community level : a dictionary mapping each community to a series with the
                                     average number of shares at each time
                If node level: a dataframe of every node and their counts at each time
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)
        all_times = sorted(data[self.timestamp_col].unique())

        def get_count_at_time(grp, content):
            counts = []
            for time in all_times:
                counts.append((grp[self.timestamp_col] == time).sum())
            value_counts = pd.DataFrame([content] + counts, index=["node"] + all_times).transpose()
            value_counts = value_counts.set_index("node")
            return value_counts

        if community_level:
            community_dict = {}
            for comm, group in data.groupby(self.community_col):
                single_comm_df = []
                for i, subgroup in group.groupby(self.content_col):
                    single_comm_df.append(get_count_at_time(subgroup, i))
                time_series = pd.concat(single_comm_df)
                community_dict[comm] = time_series.mean()
        else:
            content_df = []
            for i, group in data.groupby(self.content_col):
                content_df.append(get_count_at_time(group, i))
            time_series = pd.concat(content_df)
            if node_level:
                return time_series
            else:
                return time_series.mean()

    def top_info_shared(self, k, node_level=False, community_level=False, nodes=None, communities=None, platform="all"):
        """
        Description:

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a sorted dataframe with the top k nodes and their counts
                If community level : a dictionary maaping each community to a sorted dataframe
                                     with the top k nodes and their counts
                If node level: a sorted dataframe with the top k nodes and their counts
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_top_k(grp):
            counts_of_shared = grp.groupby(self.content_col).apply(lambda x: len(x))
            counts_of_shared.reset_index(inplace=True)
            counts_of_shared.rename(index=str, columns={0: "value"}, inplace=True)
            return counts_of_shared.nlargest(k, "value").reset_index(drop=True)

        if not community_level:
            return get_top_k(data)
        else:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_top_k(group)
            return community_dict

    def unique_users(self, node_level=False, community_level=False, nodes=None, communities=None, platform="all"):
        """
        Description:

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level:
                If community level :
                If node level: a dictionary mapping each piece of infor to the number of unique
                               users reached by that info
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)
        if node_level:
            return data.groupby(self.content_col).apply(lambda x: len(x[self.user_col].unique())).to_dict()

    def unique_users_over_time(self, node_level=False, community_level=False, nodes=None,
                               communities=None, platform="all"):
        """
        Description:

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level:
                If community level :
                If node level: a dataframe where each node is a row and the columns contain the number of unique users
                               at each time step
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)
        all_times = sorted(data[self.timestamp_col].unique())

        def get_count_at_time(grp, content):
            counts = []
            for time in all_times:
                counts.append(len((grp.loc[grp[self.timestamp_col] == time, self.user_col]).unique()))
            value_counts = pd.DataFrame([content] + counts, index=["node"] + all_times).transpose()
            value_counts = value_counts.set_index("node")
            return value_counts

        if community_level:
            community_dict = {}
            for comm, group in data.groupby(self.community_col):
                single_comm_df = []
                for i, subgroup in group.groupby(self.content_col):
                    single_comm_df.append(get_count_at_time(subgroup, i))
                time_series = pd.concat(single_comm_df)
                community_dict[comm] = time_series.mean()
        else:
            users_over_time = []
            for i, group in data.groupby(self.content_col):
                users_over_time.append(get_count_at_time(group, i))
            time_series = pd.concat(users_over_time)
            if node_level:
                return time_series
            else:
                return time_series.mean()

    def top_audience_reach(self, k, node_level=False, community_level=False, nodes=None,
                           communities=None, platform="all"):
        """
        Description:

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a sorted dataframe with the top k nodes and the size of audience reached
                If community level : a dictionary maaping each community to a sorted dataframe
                                     with the top k nodes and the size of audience reached
                If node level: a sorted dataframe with the top k nodes and the size of audience reached
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_top_k(grp):
            counts_of_shared = grp.groupby(self.content_col).apply(lambda x: len(x[self.user_col].unique()))
            counts_of_shared.reset_index(inplace=True)
            counts_of_shared.rename(index=str, columns={0: "value"}, inplace=True)
            return counts_of_shared.nlargest(k, "value").reset_index(drop=True)

        if not community_level:
            return get_top_k(data)
        else:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_top_k(group)
            return community_dict

    def lifetime_of_info(self, node_level=False, community_level=False, nodes=None, communities=None, platform="all"):
        """
        Description:

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: the average lifetime between the first and last event
                If community level : a dictionary mapping each community to the average lifetime
                                     between the first and last event
                If node level: a dataframe of nodes and their lifetimes
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_lifetime(grp):
            life = grp.groupby(self.content_col).apply(
                lambda x: x[self.timestamp_col].max() - x[self.timestamp_col].min())
            life = life.reset_index()
            life.rename(index=str, columns={0: "value"}, inplace=True)
            return life

        if not community_level:
            lifetimes = get_lifetime(data)
            if not node_level:
                return lifetimes.mean()
            else:
                return lifetimes
        else:
            community_dict = {}
            for i, comm in data.groupby(self.community_col):
                community_dict[i] = get_lifetime(comm).mean()
            return community_dict

    def top_lifetimes(self, k, node_level=False, community_level=False, nodes=None, communities=None, platform="all"):
        """
        Description:

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a sorted dataframe with the top k nodes and lifetimes
                If community level : a dictionary mapping each community to a sorted dataframe
                                     with the top k nodes and lifetimes
                If node level: a sorted dataframe with the top k nodes and lifetimes
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_top_k(grp):
            counts_of_shared = grp.groupby(self.content_col).apply(
                lambda x: x[self.timestamp_col].max() - x[self.timestamp_col].min())
            counts_of_shared.reset_index(inplace=True)
            counts_of_shared.rename(index=str, columns={0: "value"}, inplace=True)
            return counts_of_shared.nlargest(k, "value").reset_index(drop=True)

        if not community_level:
            return get_top_k(data)
        else:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_top_k(group)
            return community_dict

    def speed_of_info(self, time_unit="h", node_level=False, community_level=False, nodes=None,
                      communities=None, platform="all"):
        """
        Description:

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level:
                If community level :
                If node level:
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            time = self.get_time_granularity(time, time_unit)
            if time == 0:
                speed = -1
            else:
                speed = len(grp) / time
            return speed

        if node_level:
            speeds = data.groupby(self.content_col).apply(get_speed).reset_index()
            speeds.columns = [self.content_col, 'value']
            speeds = speeds[speeds['value'] != -1]
            return speeds

    def speed_of_info_over_time(self, time_unit="h", node_level=False, community_level=False, nodes=None,
                                communities=None, platform="all"):
        """
        Description:

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level:
                If community level :
                If node level:
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_speed(grp):

            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            time = self.get_time_granularity(time, time_unit)
            if time == 0:
                speed = -1
            else:
                speed = len(grp) / time
            return speed

        if node_level:
            data.groupby(self.content_col).apply()
