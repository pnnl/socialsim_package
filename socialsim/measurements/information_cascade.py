from collections import Counter
from igraph      import Graph

import numpy  as np
import pandas as pd

import pysal

from .validators   import check_empty
from .validators   import check_root_only
from .measurements import MeasurementsBaseClass

class InformationCascadeMeasurements(MeasurementsBaseClass):
    def __init__(self, main_df, configuration, metadata, platform,
        parent_node_col="parentID", node_col="nodeID", root_node_col="rootID",
        timestamp_col="nodeTime", user_col="nodeUserID", filter_on_col=None,
        filter_in_list=[], log_file='cascade_measurements_log.txt'):
        """
        Description:

        Inputs:
            :main_df: df containing all original tweets/posts and the
                tweets/comments in their cascades
            :parent_node: name of the column containg the uid of the node who
                was retweeted from
            :node: name of the column containg the uid of the node who
                retweeted from node in parent_node
            :timestamp: time of the original tweet/retweet

        Outputs:
            None
        """
        super(InformationCascadeMeasurements, self).__init__(main_df, 
            configuration, log_file=log_file)

        self.measurement_type = 'cascade'

        self.main_df         = main_df.copy()
        self.root_node_col   = root_node_col
        self.parent_node_col = parent_node_col
        self.node_col        = node_col
        self.timestamp_col   = timestamp_col
        self.user_col        = user_col
        self.filter_on_col   = filter_on_col
        self.filter_in_list  = filter_in_list

        columns = ['informationID', 'urlDomains', 'partialParentID']

        self.main_df = self.main_df.drop(columns=columns,errors='ignore')
        self.main_df = self.main_df.drop_duplicates()

        self.main_df.drop

        if len(self.main_df) > 0:
            # for reddit community measurements
            if self.filter_on_col is not None and len(filter_in_list) > 0:
                self.main_df = self.main_df[self.main_df[self.filter_on_col].isin(self.filter_in_list)]
            self.cascade_distribution_measurement_df = None
            self.community_users_count_timeseries_df = {}

            try:
                self.main_df['communityID'] = self.main_df['nodeAttributes'].apply(lambda x: eval(x)['communityID'])
            except:
                pass

            self.preprocess_and_create_nx_dict()

        self.functions = [
        'get_node_level_measurements',
        'get_cascades_distribution_measurements'
        ]


    def preprocess_and_create_nx_dict(self):
        self.scms = {}
        for cascade_identifier, cascade_df in self.main_df.groupby(self.root_node_col):
            if len(cascade_df[cascade_df[self.node_col] == cascade_df[self.root_node_col]].index) > 0:
                self.scms[cascade_identifier] = SingleCascadeMeasurements(main_df=cascade_df,
                    parent_node_col=self.parent_node_col, root_node_col=self.root_node_col,
                    node_col=self.node_col, timestamp_col=self.timestamp_col, user_col=self.user_col)


    @check_empty(default=None)
    def get_node_level_measurements(self, single_cascade_measurement, **kwargs):
        """
        Description:

        Input:
            :single_cascade_measurement: function to obtain the single cascade
                level timeseries/distribution measurement

        Output:
            :result: (dict) A dictionary with the following properties:
                keys: rootID,
                values: dataframe for depth based and timeseries measurements
                and single value for gini/palma measurements returned by the
                cascade_measurement function
        """
        result = {}

        for cascade_identifier, scm in self.scms.items():
            attribute   = getattr(scm, single_cascade_measurement)(**kwargs)
            update_dict = {cascade_identifier: attribute}
            result.update(update_dict)

        return result


    def split_communities(self, data, community_grouper):
        """
        Description:

        Input:

        Output:
        """
        communities = {}
        
        for community in data[community_grouper].unique():
            columns = [c for c in data.columns if c!=community_grouper]

            community_data = data[data[community_grouper]==community][columns]

            communities.update({community:community_data})

        return communities


    @check_empty(default=None)
    def get_cascades_distribution_measurements(self):
        """
        Description:

        Inputs:
            None
        
        Output:
            :return: pandas dataframe with cascade identiifer and "depth", 
            "breadth", "size", "structural_virality" and lifetime for each 
            cascade in the population
        """
        cascades_distribution_measurements = []
        for cascade_identifier, scm in self.scms.items():
            cascade_information = [
                cascade_identifier,
                scm.community,
                scm.cascade.get_cascade_depth(),
                scm.cascade.get_cascade_size(),
                scm.cascade.get_cascade_breadth(),
                scm.cascade.get_cascade_structural_virality(),
                scm.cascade.get_cascade_lifetime()
                ]

            cascades_distribution_measurements.append(cascade_information)

        cols = [
            "rootID", 
            "communityID", 
            "depth", 
            "size", 
            "breadth", 
            "structural_virality", 
            "lifetime"
            ]

        data = pd.DataFrame(cascades_distribution_measurements, columns=cols)

        self.cascade_distribution_measurement_df = data

    """
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    The methods below this line are all measurement functions.
    """
    def cascade_collection_distribution_of(self, attribute, 
        community_grouper=None):
        """
        :param attribute: "depth", "size", "breadth", "structural_virality", 
            "lifetime"
        """
        if self.cascade_distribution_measurement_df is None:
            self.get_cascades_distribution_measurements()

        if not community_grouper:
            meas = self.cascade_distribution_measurement_df[["rootID", attribute]]
            meas.columns = ['content', 'value']
        else:
            meas = {}
            for community in self.cascade_distribution_measurement_df[community_grouper].unique():
                if community != '':
                    df = self.cascade_distribution_measuremnt_df

                    df = df[df[community_grouper] == community][["rootID", attribute]]
                    df.columns = ['content', 'value']
                    meas[community] = df
        return meas


    @check_empty(default=None)
    def get_cascade_collection_timeline_timeseries(self, time_granularity="M",
        community_grouper=None):
        """
         :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
         :param community_grouper: column that indicates a community, eg. communityID, keyword
         :return: pandas dataframe with mean "lifetime" of cascades that start in that interval
         """
        temporal_measurements = []
        result_df_columns = ["timestamp", "value"]
        grouper = [pd.Grouper(freq=time_granularity)]
        if community_grouper and community_grouper in self.main_df.columns:
            grouper.append(community_grouper)
            result_df_columns = ["timestamp", community_grouper, "value"]
        for ts, df in self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]]. \
                set_index(self.timestamp_col).groupby(grouper, sort=True):
            mean_lifetime = df.groupby(self.root_node_col).size().mean()

            temporal_measurements.append(list(ts) + [mean_lifetime] if community_grouper and community_grouper in self.main_df.columns else [ts, mean_lifetime])

        temporal_measurements = pd.DataFrame(temporal_measurements, columns=result_df_columns).fillna(0)

        if community_grouper:
            if community_grouper in temporal_measurements and '' not in temporal_measurements[community_grouper]:
                temporal_measurements = self.split_communities(temporal_measurements, community_grouper)
            else:
                temporal_measurements = {}

        return temporal_measurements


    @check_empty(default=None)
    def get_cascade_collection_size_timeseries(self, time_granularity="M",
        community_grouper=None):
        """
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
         :param community_grouper: column that indicates a community, eg. communityID, keyword
        :return: pandas dataframe with mean "size" of cascades that start in that interval
        """
        temporal_measurements = []
        result_df_columns = ["timestamp", "value"]
        grouper = [pd.Grouper(freq=time_granularity)]
        if community_grouper and community_grouper in self.main_df.columns:
            grouper.append(community_grouper)
            result_df_columns = ["timestamp", community_grouper, "value"]
        for ts, df in self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]]. \
                set_index(self.timestamp_col).groupby(grouper, sort=True):
            if len(df.index) > 0:
                mean_size = sum([self.scms[cascade_identifier].cascade.get_cascade_size() for cascade_identifier in
                                 df[self.root_node_col].values]) / len(df)
                temporal_measurements.append(
                    list(ts) + [mean_size] if community_grouper and community_grouper in self.main_df.columns else [ts,
                                                                                                                    mean_size])
            # temporal_measurements.append([*ts, mean_size] if community_grouper else [ts, mean_size])
        meas = pd.DataFrame(temporal_measurements, columns=result_df_columns)

        if len(meas.index) == 0:
            return None
        else:
            if community_grouper:
                if community_grouper in meas.columns and '' not in meas[community_grouper]:
                    meas = self.split_communities(meas, community_grouper)
                else:
                    meas = {}
            return meas


    @check_empty(default=None)
    def get_community_users_count_timeseries(self, time_granularity="M",
        community_grouper=None):
        """
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
         :param community_grouper: column that indicates a community, eg. communityID, keyword
        :return: pandas dataframe with number of unique users who participate in start in that interval
        """
        temporal_measurements = []

        if community_grouper in self.main_df.columns:
            for community_identifier, community_df in self.main_df.groupby(community_grouper):
                cumul_df = None
                for ts, df in community_df.set_index(self.timestamp_col).groupby(pd.Grouper(freq=time_granularity),
                                                                                 sort=True):
                    if cumul_df is None:
                        cumul_df = df.copy()
                        old_unique_users_count = 0
                    else:
                        old_unique_users_count = cumul_df[self.user_col].nunique()
                        cumul_df = cumul_df.append(df, ignore_index=True)
                    unique_users_count = cumul_df[self.user_col].nunique()
                    new_users_ratio = (unique_users_count - old_unique_users_count) / float(unique_users_count)
                    temporal_measurements.append([ts, community_identifier, unique_users_count, new_users_ratio])
                self.community_users_count_timeseries_df[time_granularity] = pd.DataFrame(temporal_measurements,
                                                                                          columns=["timestamp",
                                                                                                   community_grouper,
                                                                                                   "unique_users",
                                                                                                   "new_user_ratio"])


    def community_users_count(self, attribute, time_granularity,
        community_grouper):
        """
        :param attribute: "unique_users", "new_user_ratio"
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
        :param community_grouper: column that indicates a community, eg. communityID, keyword
        """
        if community_grouper in self.main_df.columns:
            if time_granularity not in self.community_users_count_timeseries_df:
                self.get_community_users_count_timeseries(time_granularity, community_grouper=community_grouper)
            df = self.community_users_count_timeseries_df[time_granularity][["timestamp", community_grouper, attribute]]
            df.columns = ["timestamp", community_grouper, "value"]

            meas = self.split_communities(df, community_grouper)

            return meas
        else:
            return None


    @check_empty(default=None)
    def cascade_collection_initialization_gini(self, community_grouper=None):
        if not community_grouper:
            root_node_users = self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]][
                self.user_col].values
            return pysal.explore.inequality.gini.Gini(list(Counter(root_node_users).values())).g
        elif community_grouper in self.main_df.columns:
            meas = {}
            for community in self.main_df[community_grouper].unique():
                root_node_users = self.main_df[(self.main_df[self.node_col] == self.main_df[self.root_node_col]) & (
                            self.main_df[community_grouper] == community)][self.user_col].values
                meas[community] = pysal.explore.inequality.gini.Gini(list(Counter(root_node_users).values())).g
            return meas
        else:
            return None


    @check_empty(default=None)
    @check_root_only(default=None)
    def cascade_collection_initialization_palma(self, community_grouper=None):

        if not community_grouper:
            root_node_users = self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]][
                self.user_col].values
            return palma_ratio(list(Counter(root_node_users).values()))
        elif community_grouper in self.main_df.columns:
            meas = {}
            for community in self.main_df[community_grouper].unique():
                root_node_users = self.main_df[(self.main_df[self.node_col] == self.main_df[self.root_node_col]) & (
                            self.main_df[community_grouper] == community)][self.user_col].values
                meas[community] = palma_ratio(list(Counter(root_node_users).values()))
            return meas
        else:
            return None


    @check_empty(default=None)
    def cascade_collection_participation_gini(self, community_grouper=None):
        if not community_grouper:
            all_node_users = self.main_df[self.user_col].values
            return pysal.explore.inequality.gini.Gini(list(Counter(all_node_users).values())).g
        elif community_grouper in self.main_df.columns:
            meas = {}
            for community in self.main_df[community_grouper].unique():
                all_node_users = self.main_df[self.main_df[community_grouper] == community][self.user_col].values
                meas[community] = pysal.explore.inequality.gini.Gini(list(Counter(all_node_users).values())).g
            return meas
        else:
            return None


    @check_empty(default=None)
    @check_root_only(default=None)
    def cascade_collection_participation_palma(self, community_grouper=None):
        if not community_grouper:
            all_node_users = self.main_df[self.user_col].values
            return palma_ratio(list(Counter(all_node_users).values()))
        elif community_grouper in self.main_df.columns:
            meas = {}
            for community in self.main_df[community_grouper].unique():
                all_node_users = self.main_df[self.main_df[community_grouper] == community][self.user_col].values
                meas[community] = palma_ratio(list(Counter(all_node_users).values()))
            return meas
        else:
            return None


    @check_empty(default=None)
    @check_root_only(default=1.0)
    def fraction_of_nodes_in_lcc(self, community_grouper=None):
        if not community_grouper:
            return max([scm.cascade.get_cascade_size() for scm in self.scms.values()]) / len(self.main_df)
        elif community_grouper in self.main_df.columns:
            meas = {}
            for community in self.main_df[community_grouper].unique():
                meas[community] = max(
                    [scm.cascade.get_cascade_size() for scm in self.scms.values() if scm.community == community]) / len(
                    self.main_df[self.main_df[community_grouper] == community])
            return meas
        else:
            return None


    def fraction_of_isolated_nodes(self):
        """
        not applicable since we do not consider isolated nodes as cascades
        """
        pass


    def fraction_of_nodes_with_outside_links(self):
        """
        We might not have url information in simulations
        """
        pass


class Cascade:
    """
    depth, max breadth, size and structural virality measurements from
    Soroush Vosoughi, Deb Roy, and Sinan Aral.
    The spread of true and false news online. Science. 2018
    """

    def __init__(self, main_df=None, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID",
                 timestamp_col="nodeTime", user_col="nodeUserID", community_col="communityID"):
        """
        Description:

        Input:
        :main_df: (pd.DataFrame) Dataset containing all tweets in the RT
            cascade of the original tweet.
        parent_node: name of the column containg the uid of the node who was retweeted from (None for cascade root)
        node: name of the column containg the uid of the node who retweeted from node in parent_node
        timestamp: time of the original tweet/retweet
        """
        self.parent_node_col = parent_node_col
        self.node_col        = node_col
        self.root_node_col   = root_node_col
        self.timestamp_col   = timestamp_col
        self.user_col        = user_col
        self.community_col   = community_col

        if main_df is not None:

            if self.community_col in main_df.columns:
                self.community = main_df[self.community_col].values[0]
            else:
                self.community = ''

            if len(main_df) > 0:
                self.preprocess_and_create_nx(main_df, set_cascade=True)
            else:
                self.main_df = main_df
        else:
            self.community = ''


    def set_root_node(self, main_df):
        root_df = main_df[main_df[self.node_col] == main_df[self.root_node_col]]
        self.root_node = root_df[self.node_col].values[0]


    def preprocess_and_create_nx(self, main_df, set_cascade=True):
        if set_cascade:
            self.main_df = main_df
            root_df = self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]]
            self.root_node = root_df[self.node_col].values[0]
            self.cascade_nx = igraph_from_pandas_edgelist(
                main_df[main_df[self.node_col] != main_df[self.root_node_col]],
                source=self.node_col,
                target=self.parent_node_col,
                directed=False)
        return main_df


    def update_cascade(self, df):
        if not hasattr(self, 'main_df') or len(self.main_df) == 0:
            self.main_df = df

            if self.community_col in df.columns:
                self.community = self.main_df[self.community_col].values[0]

            #root_df = self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]]
            #self.root_node = root_df[self.node_col].values[0]
            self.root_node = self.main_df[self.root_node_col].values[0]
            self.cascade_nx = igraph_from_pandas_edgelist(
                self.main_df[self.main_df[self.node_col] != self.main_df[self.root_node_col]],
                source=self.node_col,
                target=self.parent_node_col,
                directed=False)
        else:
            igraph_add_edges_to_existing_graph(self.cascade_nx, df, source=self.node_col, target=self.parent_node_col)
            self.main_df = pd.concat([self.main_df, df], sort=True)

    """
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    The methods below this line are all measurement functions.
    """
    def get_depth_of_each_node(self):
        self.main_df.loc[:, "depth"] = -1

        self.main_df.loc[self.main_df[self.node_col] == self.root_node, 'depth'] = 0
        seed_nodes = [self.root_node]
        depth = 1

        while len(seed_nodes) > 0:
            self.main_df.loc[(self.main_df[self.parent_node_col].isin(seed_nodes)) & (self.main_df[self.node_col] != self.main_df[self.parent_node_col]), 'depth'] = depth
            seed_nodes = self.main_df[(self.main_df[self.parent_node_col].isin(seed_nodes)) & (self.main_df[self.node_col] != self.main_df[self.parent_node_col])][self.node_col].values

            if not len(set(seed_nodes))==len(seed_nodes):
                print(seed_nodes)
                print(len(set(seed_nodes)))
                print(len(seed_nodes))

            assert len(set(seed_nodes)) == len(seed_nodes)
            depth += 1


    @check_empty(default=None)
    @check_root_only(default=0)
    def get_cascade_depth(self):
        self.get_depth_of_each_node()
        return max(self.main_df['depth'])


    @check_empty(default=0)
    @check_root_only(default=1)
    def get_cascade_size(self):
        return Graph.vcount(self.cascade_nx)


    @check_empty(default=None)
    @check_root_only(default=0)
    def get_cascade_breadth(self):
        self.get_depth_of_each_node()
        return max(self.main_df.groupby('depth').size().reset_index(name='breadth_at_depth')['breadth_at_depth'])


    @check_empty(default=None)
    @check_root_only(default=None)
    def get_cascade_structural_virality(self):
        """
        :return: structural virality of a single cascade.
                 For definition:
                 Soroush Vosoughi, Deb Roy, Sinan Aral. The spread of true and false news online. Science. 2018
        """
        n = Graph.vcount(self.cascade_nx)
        # try:
        #     return igraph_wiener_index(self.cascade_nx) * 2 / (n * (n - 1))
        # except:
        #     return None
        return igraph_wiener_index(self.cascade_nx) * 2 / (n * (n - 1))


    @check_empty(default=set())
    def get_cascade_nodes(self, unique=True):
        if unique:
            return set(self.main_df[self.user_col])
        else:
            return set(self.main_df[self.node_col])


    @check_empty(default=None)
    @check_root_only(default=0)
    def get_cascade_lifetime(self, granularity="D"):
        """
        :param granularity: "s", "m", "H", "D"  [seconds/minutes/days/hours]
        """
        try:
            lifetime = (max(self.main_df[self.timestamp_col]) - min(self.main_df[self.timestamp_col])).total_seconds()
        except:
            lifetime = (max(self.main_df[self.timestamp_col]) - min(self.main_df[self.timestamp_col]))
        if granularity in ["m", "H", "D"]:
            lifetime /= 60
        if granularity in ["H", "D"]:
            lifetime /= 60
        if granularity == "D":
            lifetime /= 24
        return lifetime


class SingleCascadeMeasurements:
    """
    measurements for a cascade i.e. node level measurements
    """

    def __init__(self, main_df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID",
                 timestamp_col="nodeTime", user_col="nodeUserID",
                 community_col="communityID"):
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.root_node_col = root_node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.community_col = community_col

        if self.community_col in main_df.columns:
            self.community = main_df[self.community_col].values[0]
        else:
            self.community = ''

        self.cascade = Cascade(parent_node_col=self.parent_node_col, node_col=self.node_col,
                               root_node_col=self.root_node_col, timestamp_col=self.timestamp_col,
                               user_col=self.user_col)
        if main_df is not None:
            if len(main_df) > 0:
                self.main_df = self.cascade.preprocess_and_create_nx(main_df, set_cascade=True)
            else:
                self.main_df = main_df
        self.reset_incremental_measurements()

    def reset_incremental_measurements(self):
        self.temporal_measurements = {}
        self.depth_based_measurement_df = None

    def set_cascade(self, cascade=None):
        if cascade:
            self.cascade = cascade
            self.main_df = cascade.main_df
        else:
            # create cascade according to main_df, useful for changing the cascade object back to full cascade after incremental measurements
            self.cascade = Cascade(self.main_df, parent_node_col=self.parent_node_col, node_col=self.node_col,
                                   root_node_col=self.root_node_col, timestamp_col=self.timestamp_col,
                                   user_col=self.user_col)

    # for incremental measurements, to start with no nodes in the cascade
    def reset_cascade(self):
        self.cascade = Cascade(parent_node_col=self.parent_node_col, node_col=self.node_col,
                               root_node_col=self.root_node_col, timestamp_col=self.timestamp_col,
                               user_col=self.user_col)


    """
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    The methods below this line are all measurement functions.
    """
    @check_empty(default=None)
    def get_temporal_measurements(self, time_granularity="M"):
        """
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
        :return: pandas dataframe with "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio" at each timestamp
        """
        self.reset_cascade()
        temporal_measurements = []
        old_unique_nodes_count = 1  # root node, since we start iterating from depth 1



        for ts, df in self.main_df.set_index(self.timestamp_col).groupby(pd.Grouper(freq=time_granularity), sort=True):
            if len(df) > 0:

                self.cascade.update_cascade(df)
                old_unique_nodes_count, temporal_measurement = self.get_incremental_cascade_measurements(ts, old_unique_nodes_count)
                temporal_measurements.append(temporal_measurement)



        columns = ["timestamp", "depth", "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio"]



        self.temporal_measurements[time_granularity] = pd.DataFrame(temporal_measurements, columns=columns)
        self.temporal_measurements[time_granularity].fillna(value=np.nan, inplace=True)
        self.set_cascade()

    def cascade_timeseries_of(self, attribute, time_granularity):
        """
        Description:

        Input:
            :attribute: (str) Could be any of the following: "depth",
                "breadth", "size", "structural_virality", "unique_nodes",
                "new_node_ratio"

            :time_granularity: (str) Again, could be any of the following:
                "Y", "M", "D", "H".
        """

        if time_granularity not in self.temporal_measurements:
            self.get_temporal_measurements(time_granularity)


        meas = self.temporal_measurements[time_granularity][['timestamp', attribute]]
        meas.columns = ['timestamp','value']

        return meas

    @check_empty(default=None)
    def get_depth_based_measurements(self):
        """
        Description:

        Input:

        Output:

        :return: pandas dataframe with "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio" at each depth
        """
        self.main_df["depth"] = -1
        self.reset_cascade()
        self.cascade.set_root_node(self.main_df)

        self.main_df.loc[self.main_df[self.node_col] == self.cascade.root_node, 'depth'] = 0

        seed_nodes = [self.cascade.root_node]
        depth = 1

        while len(seed_nodes) > 0:
            self.main_df.loc[(self.main_df[self.parent_node_col].isin(seed_nodes)) & (
                        self.main_df[self.node_col] != self.main_df[self.parent_node_col]), 'depth'] = depth
            seed_nodes = self.main_df[(self.main_df[self.parent_node_col].isin(seed_nodes)) & (
                        self.main_df[self.node_col] != self.main_df[self.parent_node_col])][self.node_col].values

            assert len(set(seed_nodes)) == len(seed_nodes)
            depth += 1
        depth_based_measurements = [[0,0,0,0,0,0]]
        old_unique_nodes_count = 1  # root node, since we start iterating from depth 1
        self.cascade.update_cascade(self.main_df[self.main_df["depth"] == 0])  # initialize with root
        for depth in range(1, max(self.main_df['depth']) + 1):
            self.cascade.update_cascade(self.main_df[self.main_df["depth"] == depth])
            old_unique_nodes_count, depth_based_measurement = self.get_incremental_cascade_measurements(depth,
                                                                                                        old_unique_nodes_count,
                                                                                                        by_depth=True)
            depth_based_measurements.append(depth_based_measurement)
        self.depth_based_measurement_df = pd.DataFrame(depth_based_measurements,
                                                       columns=["depth", "breadth", "size", "structural_virality",
                                                                "unique_nodes", "new_node_ratio"])
        self.depth_based_measurement_df.fillna(value=np.nan, inplace=True)
        self.set_cascade()

    def cascade_depth_by(self, attribute):
        """
        :param attribute: "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio"
        """

        if self.depth_based_measurement_df is None:
            self.get_depth_based_measurements()

        meas = self.depth_based_measurement_df[['depth', attribute]]
        meas.columns = ['depth','value']
        return meas

    def get_incremental_cascade_measurements(self, grouper_value, old_unique_nodes_count, by_depth=False):
        unique_nodes_count = len(self.cascade.get_cascade_nodes(unique=True))

        all_measurements = [grouper_value,
                            self.cascade.get_cascade_depth(),
                            self.cascade.get_cascade_breadth(),
                            self.cascade.get_cascade_size(),
                            self.cascade.get_cascade_structural_virality(),
                            unique_nodes_count,
                            (unique_nodes_count - old_unique_nodes_count) / float(unique_nodes_count)]



        if by_depth:
            all_measurements = all_measurements[1:]

        return unique_nodes_count, all_measurements

    @check_empty(default=None)
    @check_root_only(default=0)
    def cascade_participation_gini(self):
        return pysal.explore.inequality.gini.Gini(self.node_participation()).g

    @check_empty(default=None)
    @check_root_only(default=None)
    def cascade_participation_palma(self):
        return palma_ratio(self.node_participation())

    def node_participation(self):
        return self.main_df.groupby(self.main_df[self.user_col]).size().reset_index(name='counts')['counts'].values


def palma_ratio(values):
    values = np.sort(np.array(values))
    percent_nodes = np.arange(1, len(values) + 1) / float(len(values))
    # percent of events taken by top 10% of nodes
    p10 = np.sum(values[percent_nodes >= 0.9])
    # percent of events taken by bottom 40% of nodes
    p40 = np.sum(values[percent_nodes <= 0.4])
    try:
        p = float(p10) / float(p40)
    except ZeroDivisionError:
        return None
    return p


def igraph_from_pandas_edgelist(df, source, target, directed):

    edgelist = df[[source, target]].apply(tuple, axis=1).tolist()

    return Graph.TupleList(edgelist, directed=directed)


def igraph_add_edges_to_existing_graph(graph, df, source, target):
    if len(df) > 0:
        edgelist = df[[source, target]].apply(tuple, axis=1).tolist()
        new_vertices = set(df[[source, target]].values.flatten().tolist())
        old_vertices = set(graph.vs['name'])
        graph.add_vertices(list(new_vertices - old_vertices))
        graph.add_edges(edgelist)
    return graph


def igraph_wiener_index(graph):
    # igraph only works if node names are string, otherwise it assumes that the names are indices
    all_nodes = graph.vs['name']
    wiener_index = 0
    for i, node in enumerate(all_nodes):
        wiener_index += sum([len(path) - 1 for path in graph.get_all_shortest_paths(node, to=all_nodes[i + 1:])])
    return wiener_index
