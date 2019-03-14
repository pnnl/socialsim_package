import pandas as pd

from collections import Counter, OrderedDict
from .measurements import MeasurementsBaseClass
from ..utils import add_communities_to_dataset


class MultiPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, metadata=None, platform_col="platform",
                 timestamp_col="nodeTime", user_col="nodeUserID", content_col="informationID",
                 community_col="community", node_list=None, community_list=None, id_col='nodeID',
                 root_col="rootID", log_file='multi_platform_measurements_log.txt'):
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
        self.id_col = id_col
        self.root_col = root_col

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

    def select_data(self, nodes=[], communities=[], platform="all"):
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
            if str(nodes) == 'all':
                data = self.dataset.loc[self.dataset[self.platform_col].isin(platform_list)]
            else:
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

        print(nodes)

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

    def number_of_shares(self, node_level=False, community_level=False, nodes=[], communities=[], platform="all"):
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
            for com, grp in data.groupby(self.community_col):
                num_shares[com] = counts(grp)

        return num_shares

    def number_of_shares_over_time(self, time_bin='D',delta_t = False,
                                   node_level=False, community_level=False,
                                   nodes=[], communities=[], platform="all"):
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

        group_cols = [self.content_col]
        if community_level:
            group_cols += [self.community_col]

        if not delta_t:
            #get time series in absolute time
            data = data.set_index(self.timestamp_col)
            data = data.groupby([pd.Grouper(freq=time_bin)] + group_cols)[self.id_col].count().reset_index()
        else:
            #get time series as function of time elapsed since first event
            grouped = data.groupby(group_cols)[self.timestamp_col].min().reset_index()
            grouped.columns = group_cols + ['first_event_time']
            data = data.merge(grouped,on=group_cols)
            data[self.timestamp_col] = data[self.timestamp_col] - data['first_event_time']

            if time_bin == 'D':
                data[self.timestamp_col] = data[self.timestamp_col].dt.days
            elif time_bin.lower() == 'h':
                data[self.timestamp_col] = data[self.timestamp_col].dt.hours
            elif time_bin.lower() == 'm':
                data[self.timestamp_col] = data[self.timestamp_col].dt.minutes
                
            data = data.groupby([self.timestamp_col] + group_cols)[self.id_col].count().reset_index()

        data = pd.pivot_table(data,index=self.timestamp_col,columns=self.content_col,values=self.id_col).fillna(0)

        if node_level:
            meas = {col:pd.DataFrame(data[col]).rename(columns={col:'value'}) for col in data.columns}
        else:
            data = data.mean(axis=1)
            data.name = 'value'
            meas = data.reset_index()
        
        print(meas)

        return(meas)



    def distribution_of_shares(self, node_level=False, community_level=False, nodes=[], communities=[], platform="all"):
        """
        Description: Determine the distribution of the amount of shares of information

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe with the amount of shares and their counts
                If community level : a dictionary mapping each community to a dataframe with
                                     the amount of shares and their counts
                If node level: a dataframe with the amount of shares and their counts.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_distr(grp):
            count = grp[self.content_col].value_counts().tolist()
            counter = Counter(count)
            return pd.DataFrame({"share_amount": list(counter.keys()), "info_count": list(counter.values())})

        if community_level:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_distr(group)
            return community_dict
        else:
            return get_distr(data)

    def top_info_shared(self, k, node_level=False, community_level=False, nodes=[], communities=[], platform="all"):
        """
        Description: Determine the top K pieces of information shared the most.

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
                If node level: a sorted dataframe with the top k nodes and their counts.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_top_k(grp):
            counts_of_shared = grp.groupby(self.content_col).apply(lambda x: len(x))
            counts_of_shared = counts_of_shared.reset_index()
            counts_of_shared.rename(index=str, columns={0: "value"}, inplace=True)
            return counts_of_shared.nlargest(k, "value").reset_index(drop=True)

        if not community_level:
            return get_top_k(data)
        else:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_top_k(group)
            return community_dict

    def unique_users(self, node_level=False, community_level=False, nodes=[], communities=[], platform="all"):
        """
        Description: Calculate the number of unique users reached by each piece of information

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe with each piece of info to the number of unique users
                                     reached by that info.
                If community level: a dictionary mapping each community to a dataframe with each piece of info to the
                                    number of unique users reached by that info.
                If node level: a dataframe with each piece of info to the number of unique users reached by that info.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_users(grp):
            user_count = grp.groupby(self.content_col).apply(lambda x: len(x[self.user_col].unique())).reset_index()
            user_count.columns = [self.content_col, "value"]
            return user_count

        if community_level:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_users(group)
            return community_dict
        else:
            return get_users(data)

    def unique_users_over_time(self, node_level=False, community_level=False,
                               nodes=[], communities=[], platform="all"):
        """
        Description: Calculate the number of users at each time step

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a time series with the mean number of users at each time step
                If community level: a dictionary mapping each community to a time series with the mean number of users
                                    at each time step
                If node level: a dataframe where each node is a row and the columns contain the number of unique users
                               at each time step
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_count_at_time(grp, content):
            value_counts = grp.groupby(self.timestamp_col).apply(lambda x: len(x[self.user_col].unique()))
            value_counts = value_counts.rename("value")
            value_counts.index.name = self.timestamp_col
            value_counts = value_counts.reset_index().transpose()
            value_counts.columns = value_counts.iloc[0]
            value_counts = value_counts.reindex(value_counts.index.drop(self.timestamp_col))
            value_counts[self.content_col] = content
            value_counts = value_counts.reset_index(drop=True)
            return value_counts

        if community_level:
            community_dict = {}
            for comm, group in data.groupby(self.community_col):
                single_comm_df = []
                for i, subgroup in group.groupby(self.content_col):
                    single_comm_df.append(get_count_at_time(subgroup, i))
                time_series = pd.concat(single_comm_df, sort=False).fillna(0).reset_index(drop=True)
                time_series = time_series.set_index(self.content_col)
                time_series = time_series.reindex(sorted(time_series.columns), axis=1)
                community_dict[comm] = time_series.mean().tolist()
            return community_dict
        else:
            users_over_time = []
            for i, group in data.groupby(self.content_col):
                users_over_time.append(get_count_at_time(group, i))
            time_series = pd.concat(users_over_time, sort=False).fillna(0).reset_index(drop=True)
            time_series = time_series.set_index(self.content_col)
            time_series = time_series.reindex(sorted(time_series.columns), axis=1)
            if node_level:
                return time_series
            else:
                return time_series.mean().tolist()

    def distribution_of_users(self, node_level=False, community_level=False,
                              nodes=[], communities=[], platform="all"):
        """
        Description: Determine the distribution of the number of users reached

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe of the number of unique users and their counts.
                If community level : a dictionary mapping each community to a dataframe of the number of unique users
                                     and their counts
                If node level: a dataframe of the number of unique users and their counts.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_user_counts(grp):
            user_counts = Counter(
                grp.groupby(self.content_col).apply(lambda x: len(x[self.user_col].unique())).tolist())
            return pd.DataFrame({"user_amount": list(user_counts.keys()), "info_count": list(user_counts.values())})

        if community_level:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_user_counts(group)
            return community_dict
        else:
            return get_user_counts(data)

    def top_audience_reach(self, k, node_level=False, community_level=False,
                           nodes=[], communities=[], platform="all"):
        """
        Description: Determine the top K pieces of information with the largest audience reach.

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
                If node level: a sorted dataframe with the top k nodes and the size of audience reached.
                                Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_top_k(grp):
            counts_of_shared = grp.groupby(self.content_col).apply(lambda x: len(x[self.user_col].unique()))
            counts_of_shared = counts_of_shared.reset_index()
            counts_of_shared.rename(index=str, columns={0: "value"}, inplace=True)
            return counts_of_shared.nlargest(k, "value").reset_index(drop=True)

        if not community_level:
            return get_top_k(data)
        else:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_top_k(group)
            return community_dict

    def lifetime_of_info(self, node_level=False, community_level=False, nodes=[], communities=[], platform="all"):
        """
        Description: Calculate the lifetime of the pieces of information

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a distribution of the average lifetimes for each piece of info
                If community level: a dictionary mapping each community to a distribution of
                                    the average lifetimes for each piece of info
                If node level: a dataframe of nodes and their lifetimes.
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
                return lifetimes.mean().tolist()
            else:
                return lifetimes
        else:
            community_dict = {}
            for i, comm in data.groupby(self.community_col):
                community_dict[i] = get_lifetime(comm).mean().tolist()
            return community_dict

    def lifetime_of_threads(self, node_level=False, community_level=False, nodes=[], communities=[], platform="all"):
        """
        Description: Calculate the lifetime of the initial shares

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a distribution of the average lifetimes for each piece of info
                If community level: a dictionary mapping each community to a distribution of
                                    the average lifetimes for each piece of info
                If node level: a dataframe of nodes and their lifetimes.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_lifetime(grp):
            life = grp.groupby(self.root_col).apply(
                lambda x: x[self.timestamp_col].max() - x[self.timestamp_col].min())
            life = life.reset_index()
            life.rename(index=str, columns={0: "value"}, inplace=True)
            return life

        if not community_level:
            lifetimes = get_lifetime(data)
            if not node_level:
                return lifetimes.mean().tolist()
            else:
                return lifetimes
        else:
            community_dict = {}
            for i, comm in data.groupby(self.community_col):
                community_dict[i] = get_lifetime(comm).mean().tolist()
            return community_dict

    def top_lifetimes(self, k, node_level=False, community_level=False, nodes=[], communities=[], platform="all"):
        """
        Description: Determine the top K pieces of information with the longest lifetimes

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
                If node level: a sorted dataframe with the top k nodes and lifetimes.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_top_k(grp):
            counts_of_shared = grp.groupby(self.content_col).apply(
                lambda x: x[self.timestamp_col].max() - x[self.timestamp_col].min())
            counts_of_shared = counts_of_shared.reset_index()
            counts_of_shared.rename(index=str, columns={0: "value"}, inplace=True)
            return counts_of_shared.nlargest(k, "value").reset_index(drop=True)

        if not community_level:
            return get_top_k(data)
        else:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = get_top_k(group)
            return community_dict

    def speed_of_info(self, time_unit="h", node_level=False, community_level=False,
                      nodes=[], communities=[], platform="all"):
        """
        Description: Determine the speed at which each piece of information is spreading

        Input:
            time_unit: Granularity of time
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe with each node and speed
                If community level : a dictionary mapping each community to a dataframe with each node and speed
                If node level: a dataframe with each node and speed.
                               Only differs from Population if subset of nodes is passed.
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

        def speed_from_nodes(grp):
            speeds = grp.groupby(self.content_col).apply(get_speed).reset_index()
            speeds.columns = [self.content_col, 'value']
            speeds = speeds[speeds['value'] != -1]
            return speeds

        if community_level:
            community_dict = {}
            for i, group in data.groupby(self.community_col):
                community_dict[i] = speed_from_nodes(group)
            return community_dict
        else:
            return speed_from_nodes(data)

    def speed_of_info_over_time(self, time_unit="h", node_level=False, community_level=False,
                                nodes=[], communities=[], platform="all"):
        """
        Description: Determine the speed of information over time

        Input:
            time_unit: Granularity of time
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe of the average speed at each time step of all nodes
                If community level : a dictionary mapping each community to a dataframe of the average speed at each
                                    time step of every node in the community
                If node level: a dataframe of nodes and their speeds at each time step
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform)

        def get_all_times():
            all_times = []
            times = data[[self.timestamp_col]].drop_duplicates()
            for i, row in times.iterrows():
                current_time = row[self.timestamp_col]
                t = '{year}-{month:02}-{day}'.format(year=current_time.year, month=current_time.month,
                                                     day=current_time.day)
                if time_unit == "h":
                    t += ":{hour}".format(hour=current_time.hour)
                all_times.append(t)
            return all_times

        times_over_lifetime = get_all_times()

        def get_speed_over_time(grp):
            count_at_time = {t: 0 for t in times_over_lifetime}
            for i, subgroup in grp.groupby(self.timestamp_col):
                t = '{year}-{month:02}-{day}'.format(year=i.year, month=i.month,
                                                     day=i.day)
                if time_unit == "h":
                    t += ":{hour}".format(hour=i.hour)
                count_at_time[t] += len(subgroup)
            if node_level:
                sorted_time = OrderedDict(sorted(count_at_time.items()))
                return pd.DataFrame({"time": list(sorted_time.keys()), "value": list(sorted_time.values())})
            else:
                return OrderedDict(sorted(count_at_time.items()))

        if node_level:
            node_times = data.groupby(self.content_col).apply(get_speed_over_time).reset_index()
            return node_times[[self.content_col, "time", "value"]]
        elif community_level:
            community_dict = data.groupby(self.community_col).apply(get_speed_over_time).to_dict()
            community_sizes = data.groupby(self.community_col).apply(lambda x: len(x)).to_dict()
            final_dict = {}
            for comm, time_series in community_dict.items():
                size = community_sizes[comm]
                for time, count in time_series.items():
                    time_series[time] /= size
                final_dict[comm] = pd.DataFrame({"time": list(time_series.keys()), "value": list(time_series.values())})
            return final_dict
        else:
            time_series = get_speed_over_time(data)
            size = len(data[self.id_col].unique())
            for time, count in time_series.items():
                time_series[time] /= size
            return pd.DataFrame({"time": list(time_series.keys()), "value": list(time_series.values())})

    def distribution_of_speed(self, time_unit="h", node_level=False, community_level=False,
                              nodes=[], communities=[], platform="all"):
        """
        Description: Calculate the distributions of the varying speeds of the information

        Input:
            time_unit: Granularity of time
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe of the speeds of nodes and their counts
                If community level: a dictionary mapping each community to a dataframe of the
                                    speeds of nodes and their counts
                If node level: a dataframe of the speeds of nodes and their counts.
                                Only differs from Population if subset of nodes is passed.
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

        def distro(grp):
            speeds = grp.groupby(self.content_col).apply(get_speed).reset_index()
            speeds.columns = ["speed", 'value']
            speeds = speeds[speeds['value'] != -1]
            counts = speeds["value"].value_counts()
            counts = counts.rename_axis("content")
            counts = counts.rename("value")
            return pd.DataFrame({"speed": counts.index, "value": counts.values})

        if community_level:
            community_distributions = {}
            for i, group in data.groupby(self.community_col):
                community_distributions[i] = distro(group)
            return community_distributions
        else:
            return distro(data)

    def top_speeds(self, k, time_unit="h", node_level=False, community_level=False,
                   nodes=[], communities=[], platform="all"):
        """
        Description: Determine the top K pieces of information that spread the fastest.

        Input:
            k: Int. Specifies the number of pieces of information ranked
            time_unit: Granularity of time
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: dataframe with the top k nodes ranked descending order by speed.
                                     Position in dataframe is rank.
                If community level : Dictionary mapping communities to ranked dataframes of top k nodes
                                     in that community.
                If node level: dataframe with the top k nodes ranked descending order by speed. Position
                                in dataframe is rank. Only differs from Population if subset of nodes is passed.
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

        def get_top_k(grp):
            counts_of_shared = grp.groupby(self.content_col).apply(lambda x: get_speed(x))
            counts_of_shared = counts_of_shared.reset_index()
            counts_of_shared.rename(index=str, columns={0: "value"}, inplace=True)
            return counts_of_shared.nlargest(k, "value").reset_index(drop=True)

        if community_level:
            community_speeds = {}
            for i, group in data.groupby(self.community_col):
                community_speeds[i] = get_top_k(group)
            return community_speeds
        else:
            return get_top_k(data)
