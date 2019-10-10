import pandas as pd
import igraph as ig
import numpy  as np

import warnings

SNAP_LOADED = False
try:
    import snap as sn

    SNAP_LOADED = True
except:
    SNAP_LOADED = False
    warnings.warn('SNAP import failed. Using igraph version of code instead.')


import math
import time as tMeasures
from collections import OrderedDict

import tqdm
import os

import re

from .measurements import MeasurementsBaseClass

from ..load import convert_datetime
from ..utils import add_communities_to_dataset


from copy import deepcopy
from .validators import check_empty
from .validators import check_root_only
from collections import Counter
import pysal



def palma_ratio(values):
    if len(values) == 0:
        warnings.warn('Cannot compute palma ratio, no values passed (empty list)')
        return None
    sorted_values = np.sort(np.array(values))
    percent_nodes = np.arange(1, len(sorted_values) + 1) / float(len(sorted_values))
    xvals = np.linspace(0, 1, 10)
    percent_nodes_interp = np.interp(xvals, percent_nodes, sorted_values)
    top_10_pct = float(percent_nodes_interp[-1])
    bottom_40_pct = float(np.sum(percent_nodes_interp[0:4]))
    try:
        palma_ratio = top_10_pct / bottom_40_pct
    except ZeroDivisionError:
        return None
    return palma_ratio


def get_edge_string(x, y):
    x, y = sorted(list([x, y]))
    return '{}_{}'.format(x, y)


def time_since(since):
    now = tMeasures.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def get_undirected_edgelist(df, weight_filter):
    edgelist_df = df[['nodeUserID','parentUserID']].copy()
    edgelist_df['edge'] = [sorted([n,p]) for n,p in zip(df['nodeUserID'],df['parentUserID'])]
    edgelist_df['userA'] = [x[0] for x in edgelist_df['edge']]
    edgelist_df['userB'] = [x[1] for x in edgelist_df['edge']]

    edgelist_df = edgelist_df.groupby(['userA','userB']).size().reset_index().rename(columns={0:'count'})
    edgelist_df = edgelist_df[edgelist_df['count'] >= weight_filter]
    edgelist = edgelist_df[['userA','userB','count']].apply(tuple,axis=1).tolist()

    return edgelist

VALUE_COLUMN = 'value'
TIMESTEP_COLUMN = 'time'


class EvolutionMeasurements(MeasurementsBaseClass):
    """
    This class implements Time Series Network specific measurements. It uses iGraph and
    SNAP libraries with Python interfaces. For installation information please
    visit the websites for the two packages.

    iGraph-Python at http://igraph.org/python/
    SNAP Python at https://snap.stanford.edu/snappy/
    """

    def __init__(self, dataset, platform, configuration={}, metadata=None,
                 test=False, plot_graph=False,
                 parent_node_col="parentID", node_col="nodeID", root_node_col="rootID",
                 user_col="nodeUserID", weight_filter=1,
                 log_file="evolution_measurements_log.txt",
                 content_col="informationID", community_col="community",
                 node_list=None, community_list=None, time_granularity='M', timestamp_col='nodeTime'):


        super(EvolutionMeasurements, self).__init__(dataset,
            configuration=configuration, log_file=log_file)


        self.measurement_type = 'evolution'
        self.weight_filter = weight_filter
        self.root_node_col = root_node_col
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.community_col = community_col
        self.content_col = content_col

        self.configuration = configuration
        self.platform = platform

        self.data = dataset[dataset['platform']==self.platform].copy()

        if metadata is None or (metadata.community_directory is None and metadata.communities is None):
            self.community_set = self.data.copy()
            self.community_set[self.community_col] = "Default Community"
        else:
            community_directory = metadata.community_directory
            communities = metadata.communities
            self.community_set = add_communities_to_dataset(dataset,
                                                            community_directory,
                                                            communities)

        if metadata is None or metadata.node_list is None:
            if node_list == "all":
                self.node_list = self.data[self.content_col].tolist()
            elif node_list is not None:
                self.node_list = node_list
            else:
                self.node_list = []
        else:
            self.node_list = metadata.node_list

        if self.community_set is not None:
            if community_list is not None and len(community_list) > 0:
                self.communities = community_list
            else:
                self.communities = self.community_set[self.community_col].dropna().unique()
        else:
            self.communities = []


        self.time_granularity = time_granularity


        self.process_and_build_graphs(timestamp_col, configuration, metadata, test, plot_graph,
                                      parent_node_col, node_col, root_node_col, user_col, weight_filter)

    def list_measurements(self):
        count = 0
        for f in dir(self):
            if not f.startswith('_'):
                func = getattr(self, f)
                if callable(func):
                    doc_string = func.__doc__
                    if not doc_string is None and 'Measurement:' in doc_string:
                        desc = re.search('Description\:([\s\S]+?)Input', doc_string).groups()[0].strip()
                        desc = desc.replace('\n', ' ').replace('\t', ' ')
                        while '  ' in desc:
                            desc = desc.replace('  ', ' ')
                        print('{}) {}: {}'.format(count + 1, f, desc))
                        print()
                        count += 1

    def process_and_build_graphs(self, timestamp_col, configuration, metadata, test, plot_graph,
                                 parent_node_col, node_col, root_node_col, user_col, weight_filter):

        self.cascade_em = {}

        if self.node_list == [] or self.node_list == 'all':
            nodes_for_cascade_em = self.data[self.content_col].unique().tolist()
        else:
            nodes_for_cascade_em = self.node_list

        for node in nodes_for_cascade_em:
            self.cascade_em[node] = CascadeEvolutionMeasurements(self.data,
                                                                 time_granularity=self.time_granularity,
                                                                 timestamp_col=timestamp_col,
                                                                 configuration=configuration,
                                                                 metadata=metadata,
                                                                 test=test,
                                                                 log_file=f'evolution_measurements_log_{node}.txt',
                                                                 plot_graph=plot_graph, node=node,
                                                                 parent_node_col=parent_node_col, node_col=node_col,
                                                                 root_node_col=root_node_col, user_col=user_col,
                                                                 weight_filter=weight_filter)
            self.cascade_em[node].load_time_series_graphs()

    ####################################################################################################################
    # Information Related
    ####################################################################################################################

    def tendency_to_include_URL(self, node_level=False, community_level=False,
                                nodes=[], communities=[]):
        """
                Measurement: tendency_to_include_URL

                Description:  Average tendency for URLs to be included in discussion posts/comments over time
                [information evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)
        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with CascadeEvolutionMeasurements objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].tendency_to_include_URL(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set['community'] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].tendency_to_include_URL(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].tendency_to_include_URL(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res


        return None

    def tendency_to_link_external(self, node_level=False, community_level=False,
                                  nodes=[], communities=[]):
        """
                Measurement: tendency_to_link_external

                Description: Average tendency for URLs that link outside the platform of interaction to be included
                in discussion posts/comments over time [information evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)


        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].tendency_to_link_external(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].tendency_to_link_external(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].tendency_to_link_external(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None

    def number_of_domains_linked_over_time(self, node_level=False, community_level=False,
                        nodes=[], communities=[]):
        """
                Measurement: number_of_domains_linked_over_time

                Description: Average tendency for URLs that link outside the platform of interaction to be included
                in discussion posts/comments over time [information evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].number_of_domains_linked_over_time(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].number_of_domains_linked_over_time(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].number_of_domains_linked_over_time(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None


    def gini_coefficient_over_time(self, node_level=False, community_level=False,
                                                nodes=[], communities=[]):
        """
                Measurement: gini_coefficient_over_time

                Description: Gini Coefficient for Cumulative Snapshots over time for a piece of information or average
                 gini coefficient over time for a set of pieces of information [information evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].cascade_collection_participation_gini(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].cascade_collection_participation_gini(platform=self.platform))

                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].cascade_collection_participation_gini(platform=self.platform))

            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None

    def palma_ratio_over_time(self, node_level=False, community_level=False,
                                           nodes=[], communities=[]):
        """
                Measurement: palma_ratio_over_time

                Description: Palma Ratio for Cumulative Snapshots over time for a piece of information or average
                palma ratio over time for a set of pieces of information [information evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].cascade_collection_participation_palma(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].cascade_collection_participation_palma(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].cascade_collection_participation_palma(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None

    ####################################################################################################################
    # User Connections (User-Network) Related
    ####################################################################################################################

    def uniqueness_of_user_connections(self, node_level=False, community_level=False,
                                       nodes=[], communities=[]):
        """
                Measurement: uniqueness_of_user_connections

                Description: Ratio of total number of unique connections between pairs of users to the sum of
                unique connections between pairs of users in each timestep  [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: scalar per informationID (node_level), distribution per communityID (community_level),
                        distribution over all informationIDs in population (population_level)

        """
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].fluctuability(platform=self.platform)
            return res

        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].fluctuability(platform=self.platform))
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].fluctuability(platform=self.platform))
            return pop_res

    def mean_uniqueness_of_user_connections(self, node_level=False, community_level=False, nodes=[], communities=[]):
        """
                Measurement: mean_uniqueness_of_user_connections

                Description: Typical (mean-average) ratio of total number of unique connections between pairs of users
                to the sum of unique connections between pairs of users in each timestep [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: scalar per communityID (community_level) or scalar for population (population_level)

        """
        mean_res = None
        if node_level:
            # node_level mean_uniqueness_of_user_connections is simply uniqueness_of_user_connections
            return self.uniqueness_of_user_connections(node_level=True,
                                                                 community_level=False,
                                                                 nodes=nodes, communities=communities)

        if community_level:

            community_dist = self.uniqueness_of_user_connections(node_level=False,
                                                                 community_level=community_level,
                                                                 nodes=nodes, communities=communities)
            mean_res = {}
            for community in community_dist.keys():
                community_dist_vals = [x for x in community_dist[community] if x is not None]
                mean_res[community] = np.mean(community_dist_vals)

        if not community_level:
            # run at a poulation level
            population_dist = self.uniqueness_of_user_connections(node_level=False,
                                                                  community_level=community_level,
                                                                  nodes=nodes, communities=communities)
            population_dist = [x for x in population_dist if x is not None]
            mean_res = np.mean(population_dist)

        return mean_res


    def persistence_of_connectivity(self, node_level=False, community_level=False,
                                       nodes=[], communities=[]):
        """
                Measurement: uniqueness_of_user_connections

                Description: Ratio of total number of unique connections between pairs of users to the sum of
                unique connections between pairs of users in each timestep  [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].persistence_of_connectivity(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].persistence_of_connectivity(platform=self.platform))

                if len(comm_res) > 0:
                    comm_res = pd.concat(comm_res)
                    comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                else:
                    comm_res = None

                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].persistence_of_connectivity(platform=self.platform))
            if len(pop_res) > 0:
                pop_res = pd.concat(pop_res)
                pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            else:
                pop_res = None

            return pop_res

        return None


    def mean_persistence_of_connectivity(self, node_level=False, community_level=False, nodes=[], communities=[]):
        """
                Measurement: mean_persistence_of_connectivity

                Description: Proportion of edges at time t that reoccur at time t + 1 averaged across all time steps
                for each piece of information  [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: scalar per informationID (node_level), or distribution for informationIDs per communityID
                (community_level) or all informationIDs in the population (population_level)

        """

        ## mean across all timebins per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                persistence_of_connectivity_node = self.cascade_em[node].persistence_of_connectivity(
                    platform=self.platform)
                if persistence_of_connectivity_node is not None:
                    mean_persistence_of_connectivity_node = persistence_of_connectivity_node[VALUE_COLUMN].mean()
                else:
                    mean_persistence_of_connectivity_node = None
                res[node] = mean_persistence_of_connectivity_node
            return res

        ## distribution of mean across all timebins per node for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    persistence_of_connectivity_node = self.cascade_em[node].persistence_of_connectivity(
                        platform=self.platform)
                    if persistence_of_connectivity_node is not None:
                        comm_res.append(persistence_of_connectivity_node[VALUE_COLUMN].mean())
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                persistence_of_connectivity_node = self.cascade_em[node].persistence_of_connectivity(
                    platform=self.platform)
                if persistence_of_connectivity_node is not None:
                    pop_res.append(persistence_of_connectivity_node[VALUE_COLUMN].mean())

            return pop_res

        return None



    def audience_size_over_time(self, node_level=False, community_level=False,
                                nodes=[], communities=[]):
        """
                Measurement: audience_size_over_time

                Description: Cumulative number of users (nodes) in the graph over time (node level)
                or mean number of nodes in graphs (community or population level)  [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].number_of_nodes(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].number_of_nodes(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].number_of_nodes(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None


    def volume_of_user_connections_over_time(self, node_level=False, community_level=False,
                                             nodes=[], communities=[]):
        """
                Measurement: volume_of_user_connections_over_time

                Description: Number of unique user - user connections (edges) that occurred over time [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].number_of_edges(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].number_of_edges(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].number_of_edges(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None


    def density_over_time(self, node_level=False, community_level=False,
                          nodes=[], communities=[]):
        """
                Measurement: density_over_time

                Description: Density of cumulative user-user network graph at each timestep [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].density(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].density(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].density(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None


    def assortativity_coefficient_over_time(self, node_level=False, community_level=False,
                                            nodes=[], communities=[]):
        """
                Measurement: assortativity_coefficient_over_time

                Description: Assortativity Coefficient of cumulative user-user network graph at each timestep
                [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].assortativity_coefficient(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].assortativity_coefficient(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].assortativity_coefficient(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None


    def number_of_connected_components_over_time(self, node_level=False, community_level=False,
                                                 nodes=[], communities=[]):
        """
                Measurement: number_of_connected_components_over_time

                Description: Number of connected components within cumulative user-user network graph at each timestep
                [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].number_of_connected_components(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].number_of_connected_components(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].number_of_connected_components(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None

    def average_clustering_coefficient_over_time(self, node_level=False, community_level=False,
                                                 nodes=[], communities=[]):
        """
                Measurement: average_clustering_coefficient_over_time

                Description: Average clustering coefficient of cumulative user-user network graph at each timestep
                [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].average_clustering_coefficient(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].average_clustering_coefficient(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].average_clustering_coefficient(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None

    def max_node_degree_over_time(self, node_level=False, community_level=False,
                                  nodes=[], communities=[]):
        """
                Measurement: max_node_degree_over_time

                Description: Max degree of node degree distributions for cumulative user-user network graph at each
                timestep [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].max_node_degree(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].max_node_degree(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].max_node_degree(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None

    def mean_node_degree_over_time(self, node_level=False, community_level=False,
                                   nodes=[], communities=[]):
        """
                Measurement: mean_node_degree_over_time

                Description: Mean degree of node degree distributions for cumulative user-user network graph at each
                timestep [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].mean_node_degree(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].mean_node_degree(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].mean_node_degree(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None

    def community_modularity_over_time(self, node_level=False, community_level=False,
                                       nodes=[], communities=[]):
        """
                Measurement: community_modularity_over_time

                Description: Clauset-Newman-Moore modularity at each timestep that measures how strongly the network
                resolves into communities or modules as network evolves [network evolution]

                Input: platform (which platform to consider e.g. specific platform or "all" platforms as a combined
                        user-user graph

                Output: timeseries dataframe per informationID (node_level),
                timeseries averaged at each timebin across all informationIDs per communityID (community_level),
                timeseries averaged at each timebin across all informationIDs in population (population_level)

        """
        ## timeseries per node
        if node_level:
            if nodes == []:
                # if not passed, set nodes to all nodes with cascade_evolutionmeasurement objects
                nodes = self.cascade_em.keys()
            res = {}
            for node in nodes:
                res[node] = self.cascade_em[node].community_modularity(platform=self.platform)
            return res

        ## mean by timebin across all nodes for community, population
        if community_level:
            if communities == []:
                # if not passed, set communities to all communities specified at instantiation
                communities = self.communities
            res = {}
            for community in communities:
                comm_res = []
                community_nodes = list(set(self.community_set[self.community_set[self.community_col] == community][self.content_col]))
                for node in community_nodes:
                    comm_res.append(self.cascade_em[node].community_modularity(platform=self.platform))
                comm_res = pd.concat(comm_res)
                comm_res = comm_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
                res[community] = comm_res
            return res

        if not node_level and not community_level:
            # run at a poulation level
            pop_res = []
            for node in self.cascade_em.keys():
                pop_res.append(self.cascade_em[node].community_modularity(platform=self.platform))
            pop_res = pd.concat(pop_res)
            pop_res = pop_res.groupby(TIMESTEP_COLUMN, as_index=False)[[VALUE_COLUMN]].mean()
            return pop_res

        return None















class CascadeEvolutionMeasurements(MeasurementsBaseClass):
    """
    This class implements Time Series Network specific measurements. It uses iGraph and
    SNAP libraries with Python interfaces. For installation information please
    visit the websites for the two packages.

    iGraph-Python at http://igraph.org/python/
    SNAP Python at https://snap.stanford.edu/snappy/
    """

    def __init__(self, dataset, time_granularity='M', timestamp_col='nodeTime', configuration={},
                 metadata=None, test=False,
                 log_file='evolution_measurements_log.txt', plot_graph=False,
                 node="", parent_node_col="parentID", node_col="nodeID", root_node_col="rootID",
                 user_col="nodeUserID", content_col='informationID', weight_filter=1, verbose=False):

        super(CascadeEvolutionMeasurements, self).__init__(dataset,
                                                           configuration, log_file=log_file)

        self.verbose = verbose

        self.measurement_type = 'evolution'
        self.weight_filter = weight_filter
        self.root_node_col = root_node_col
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.content_col = content_col

        self.main_df = dataset.copy()
        if node != "":
            self.main_df = self.main_df.loc[self.main_df[self.content_col] == node].copy()

        self.informationIDs_string = ' InformationID(s): {}'.format(set(self.main_df[self.content_col].unique()))

        self.platforms = set(self.main_df['platform'].unique())
        self.build_undirected_graph = {'reddit': self.reddit_build_undirected_graph,
                                       'twitter': self.twitter_build_undirected_graph,
                                       'telegram': self.telegram_build_undirected_graph,
                                       'github': self.github_build_undirected_graph,
                                       'youtube': self.youtube_build_undirected_graph,
                                       'all': self.combined_build_undirected_graph}
        self.time_series_gUNig = {}
        if SNAP_LOADED:
            self.time_series_gUNsn = {}
        self.time_series_dfs = {}
        self.timestamp_orders = {}
        self.min_temporal_dists = {}
        self.full_vertices = {}
        self.connected_components_full_gUNig = {}

        if self.verbose: print("Creating time series DataFrames...")
        total_start_time = tMeasures.time()
        for platform in self.platforms:
            if platform not in self.build_undirected_graph:
                if self.verbose: print('Skipping the platform \'{}\'. No graph creation method for it.'.format(platform))
                continue
            start_time = tMeasures.time()
            self.time_series_dfs[platform] = OrderedDict()  # each dataframe per period
            platform_df = self.main_df[self.main_df['platform'] == platform].copy()
            if len(platform_df) == 0:
                continue
            start_period = platform_df[timestamp_col].min().to_period(freq=time_granularity).to_timestamp(how='S')

            end_period = platform_df[timestamp_col].max().to_period(freq=time_granularity).to_timestamp(how='E')
            self.main_df['{}_period'.format(platform)] = np.nan
            self.timestamp_orders[platform] = {}
            prev_ts = start_period - pd.Timedelta(seconds=1)

            for i, ts in enumerate(pd.date_range(start=start_period, end=end_period, freq=time_granularity)):
                ts_end = ts.to_period(freq=time_granularity).to_timestamp(how='E')
                ts_str = self.timestamp_to_str(ts, time_granularity)
                ts_platform_df = platform_df[platform_df[timestamp_col] <= ts_end].copy()
                self.main_df.loc[(self.main_df[timestamp_col] > prev_ts) & (
                            self.main_df[timestamp_col] <= ts_end), '{}_period'.format(platform)] = ts_str
                prev_ts = ts_end
                self.time_series_dfs[platform][ts_str] = ts_platform_df
                self.timestamp_orders[platform][ts_str] = i
            if self.verbose: print("\'{}\' DataFrames are created. (Time: {})".format(platform, time_since(start_time)))

        start_time = tMeasures.time()

        start_period = self.main_df[timestamp_col].min().to_period(freq=time_granularity).to_timestamp(how='S')

        end_period = self.main_df[timestamp_col].max().to_period(freq=time_granularity).to_timestamp(how='E')
        self.time_series_dfs['all'] = OrderedDict()
        self.main_df['all_period'] = np.nan
        self.timestamp_orders['all'] = {}
        prev_ts = start_period - pd.Timedelta(seconds=1)
        for i, ts in enumerate(pd.date_range(start=start_period, end=end_period, freq=time_granularity)):
            ts_end = ts.to_period(freq=time_granularity).to_timestamp(how='E')
            ts_str = self.timestamp_to_str(ts, time_granularity)
            ts_main_df = self.main_df[self.main_df[timestamp_col] <= ts_end].copy()
            self.main_df.loc[(self.main_df[timestamp_col] > prev_ts) & (
                        self.main_df[timestamp_col] <= ts_end), 'all_period'] = ts_str
            prev_ts = ts_end
            self.time_series_dfs['all'][ts_str] = ts_main_df
            self.timestamp_orders['all'][ts_str] = i
        if self.verbose: print("Full DataFrames across all platforms are created. (Time: {})".format(time_since(start_time)))

        if self.verbose: print("Done. Total loading time: {}".format(time_since(total_start_time)))

    def timestamp_to_str(self, ts, time_granularity):
        if time_granularity == 'Y':
            return ts.strftime('%Y')
        elif time_granularity == 'M':
            return ts.strftime('%Y-%m')
        elif time_granularity == 'Q':
            return ts.strftime('%Y-%m (Q)')
        elif time_granularity == 'D':
            return ts.strftime('%Y-%m-%d')
        elif time_granularity == 'W':
            return ts.strftime('%Y-%m-%d (W)')
        elif time_granularity == 'h':
            return ts.strftime('%Y-%m-%dT%H')
        elif time_granularity == 'm':
            return ts.strftime('%Y-%m-%dT%H:%M')
        elif time_granularity == 's':
            return ts.strftime('%Y-%m-%dT%H:%SZ')
        else:
            return ts.strftime('%Y-%m-%dT%H:%M:%SZ')

    def load_time_series_graphs(self):
        total_start_time = tMeasures.time()
        if self.verbose: print("Creating time series graphs...")

        for platform in self.time_series_dfs:
            start_time = tMeasures.time()
            if SNAP_LOADED:
                self.time_series_gUNsn[platform] = OrderedDict()
            self.time_series_gUNig[platform] = OrderedDict()
            if platform == 'all':
                implemented_platforms = {'twitter', 'youtube', 'reddit'}
                not_implemented = False
                for existing_platform in self.platforms:
                    if existing_platform not in implemented_platforms:
                        not_implemented = True
                        print(
                            'platform {} is not implmented for full graph analysis across all platforms. Skipping building time series full graphs'.format(
                                existing_platform))
                    break
                if not_implemented:
                    continue
            for ts in self.time_series_dfs[platform]:
                gUNig, gUNsn = self.build_undirected_graph[platform](self.time_series_dfs[platform][ts],
                                                                     weight_filter=self.weight_filter)
                self.time_series_gUNig[platform][ts] = gUNig
                if SNAP_LOADED:
                    self.time_series_gUNsn[platform][ts] = gUNsn
            if self.verbose: print("'{}' graphs are created. (Time: {})".format(platform, time_since(start_time)))
        if self.verbose: print("Done. Total loading time: {}".format(time_since(total_start_time)))


    def tendency_to_include_URL(self, platform='all'):
        """

        Description: tendency_to_include_URL, i.e. the proportion of nodes (tweets/posts/comments/messages/etc)
        that included a URL over time

        """
        res = []
        for ts in self.time_series_dfs[platform]:
            tendency_to_include_url = np.mean(self.time_series_dfs[platform][ts]['has_URL'].values)
            res.append(
                {TIMESTEP_COLUMN: ts, VALUE_COLUMN: tendency_to_include_url})
        return pd.DataFrame(res)

    def tendency_to_link_external(self, platform='all'):
        """

        Description: tendency_to_include_URL, i.e. the proportion of nodes (tweets/posts/comments/messages/etc)
        that included a link (URL) that had a domain outside the platform of interaction over time

        """
        res = []
        for ts in self.time_series_dfs[platform]:
            tendency_to_link_external = np.mean(self.time_series_dfs[platform][ts]['links_to_external'].values)
            res.append(
                {TIMESTEP_COLUMN: ts, VALUE_COLUMN: tendency_to_link_external})
        return pd.DataFrame(res)

    def number_of_domains_linked_over_time(self, platform='all'):
        """

        Description: tendency_to_include_URL, i.e. the proportion of nodes (tweets/posts/comments/messages/etc)
        that included a URL over time

        """
        res = []
        for ts in self.time_series_dfs[platform]:
            domains_linked = list(self.time_series_dfs[platform][ts]['domain_linked'])
            #print(ts,'\n', self.time_series_dfs[platform][ts][['nodeID', 'nodeUserID','nodeTime','domain_linked']].head())
            domains_linked_not_list = list(set([y for y in domains_linked if type(y) is str]))
            domains_linked = list(set([x for y in domains_linked for x in y if x!= '' and type(y) is list]))
            domains_linked = set(domains_linked + domains_linked_not_list)
            #print(ts, domains_linked)
            res.append(
                {TIMESTEP_COLUMN: ts, VALUE_COLUMN: len(domains_linked)})
        return pd.DataFrame(res)



    def mean_shortest_path_length(self, platform='all'):
        """
        mean shortest path length of cumulative snapshots of the graph over time

        """
        total_start_time = tMeasures.time()
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []

        for ts in sorted(self.time_series_gUNig[platform].keys()):
            if SNAP_LOADED:
                if self.time_series_gUNsn[platform][ts].Empty():
                    warnings.warn('Empty graph at time step \'{}\' on the platform \'{}\''.format(ts, platform))
                    value = 0
                else:
                    value = sn.GetBfsEffDiam(self.time_series_gUNsn[platform][ts],
                                             self.time_series_gUNsn[platform][ts].GetNodes(), False)
            else:
                if ig.Graph.vcount(self.time_series_gUNig[platform][ts]) == 0:
                    warnings.warn('Empty graph at time step \'{}\' on the platform \'{}\''.format(ts, platform))
                    value = 0
                else:
                    shortest_paths = self.time_series_gUNig[platform][ts].shortest_paths_dijkstra(mode='ALL')
                    shortest_paths_cleaned = np.array(shortest_paths).flatten()
                    shortest_paths_cleaned = shortest_paths_cleaned[np.isfinite(shortest_paths_cleaned)]
                    value = np.percentile([float(x) for x in shortest_paths_cleaned], 90)

            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: value})
        return pd.DataFrame(res)

    def number_of_nodes(self, platform='all'):
        """
        Measurement: number_of_nodes

        Description: Calculate the number of nodes in the cumulative graph snapshots over time

        Input:

        """

        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return
        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: ig.Graph.vcount(self.time_series_gUNig[platform][ts])})
        return pd.DataFrame(res)


    def number_of_edges(self, platform='all'):
        """
        Measurement: number_of_edges

        Description: Calculate the number of edges in the cumulative graph snapshots over time

        Input:

        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: ig.Graph.ecount(self.time_series_gUNig[platform][ts])})
        return pd.DataFrame(res)

    def density(self, platform='all'):
        """
        Measurement: density

        Description: Calculate density of the cumulative graph snapshots over time

        Input:

        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: ig.Graph.density(self.time_series_gUNig[platform][ts])})
        return pd.DataFrame(res)

    def assortativity_coefficient(self, platform='all'):
        """
        Measurement: assortativity_coefficient

        Description: Calculate the assortativity degree coefficient of the cumulative graph snapshots over time

        Input:


        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: ig.Graph.assortativity_degree(
                self.time_series_gUNig[platform][ts])})
        return pd.DataFrame(res)

    def number_of_connected_components(self, platform='all'):
        """
        Measurement: number_of_connected_components

        Description: Calculate the number of connected components in the cumulative graph snapshots over time

        Input:


        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: len(
                ig.Graph.components(self.time_series_gUNig[platform][ts], mode="WEAK"))})
        return pd.DataFrame(res)

    def largest_connected_component(self, platform='all'):
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: ig.Graph.vcount(
                ig.Graph.components(self.time_series_gUNig[platform][ts], mode="WEAK").giant())})
        return pd.DataFrame(res)

    def average_clustering_coefficient(self, platform='all'):
        """
        Measurement: average_clustering_coefficient

        Description: Calculate the average clustering coefficient of the cumulative graph snapshots over time

        Input:


        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            if SNAP_LOADED:
                value = sn.GetClustCf(self.time_series_gUNsn[platform][ts])
            else:
                value = self.time_series_gUNig[platform][ts].transitivity_avglocal_undirected(mode='zero')

            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: value})
        return pd.DataFrame(res)

    def max_node_degree(self, platform='all'):
        """
        Measurement: max_node_degree

        Description: Determine the max degree of any node in the cumulative graph snapshots over time

        Input:


        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            if ig.Graph.vcount(self.time_series_gUNig[platform][ts]) == 0:
                warnings.warn('Empty graph at time step \'{}\' on the platform \'{}\''.format(ts, platform))
                value = 0
            else:
                value = max(ig.Graph.degree(self.time_series_gUNig[platform][ts]))

            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: value})
        return pd.DataFrame(res)

    def mean_node_degree(self, platform='all'):
        """
        Measurement: mean_node_degree

        Description: Calculate the mean node degree of the cumulative graph snapshots over time

        Input:


        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            if len(list(self.time_series_gUNig[platform][ts].vs)) == 0:
                warnings.warn('Empty graph at time step \'{}\' on the platform \'{}\''.format(ts, platform))
                value = 0
            else:
                value = 2.0 * ig.Graph.ecount(self.time_series_gUNig[platform][ts]) / ig.Graph.vcount(
                    self.time_series_gUNig[platform][ts])

            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: value})
        return pd.DataFrame(res)

    def degree_distribution(self, platform='all'):
        """
        Measurement: degree_distribution

        Description: Get the distribution of all node degrees in the cumulative graph snapshots over time

        Input:


        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = {}
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            vertices = [v.attributes()['name'] for v in self.time_series_gUNig[platform][ts].vs]
            degVals = self.time_series_gUNig[platform][ts].degree(vertices)
            res[ts] = pd.DataFrame([{'node': vertices[idx], 'value': degVals[idx]} for idx in range(len(vertices))])

        return res

    def community_modularity(self, platform='all'):
        """
        Measurement: community_modularity

        Description: Calculate the community modularity of the cumulative graph snapshots over time

        Input:

        """
        if not self.time_series_gUNig:
            self.load_time_series_graphs()
        if platform not in self.time_series_gUNig:
            print("Time series graphs are not built for the platform '{}'".format(platform))
            return

        res = []
        for ts in sorted(self.time_series_gUNig[platform].keys()):
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: ig.Graph.modularity(self.time_series_gUNig[platform][ts],
                                                                                ig.Graph.community_multilevel(
                                                                                    self.time_series_gUNig[platform][
                                                                                        ts]))})
        return pd.DataFrame(res)

    def get_parent_uids(self, df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID",
                        user_col="nodeUserID"):
        """
        :return: adds parentUserID column with user id of the parent if it exits in df
        if it doesn't exist, uses the user id of the root instead
        if both doesn't exist: NaN
        """
        tweet_uids = pd.Series(df[user_col].values, index=df[node_col]).to_dict()
        df.loc[:,'parentUserID'] = df[parent_node_col].map(tweet_uids)
        df.loc[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull()), 'parentUserID'] = \
            df[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull())][root_node_col].map(tweet_uids)

        df = df[df['nodeUserID'] != df['parentUserID']].copy()

        return df

    def github_build_undirected_graph(self, df, project_on='nodeID', weight_filter=1):
        main_df = df[['nodeUserID', 'nodeID']].copy()

        right_nodes = np.array(main_df['nodeID'].unique().tolist())
        el = main_df.apply(tuple, axis=1).tolist()
        edgelist = list(set(el))
        gUNsn = None

        # iGraph graph object construction
        B = ig.Graph.TupleList(edgelist, directed=False)
        names = np.array(B.vs["name"])
        types = np.isin(names, right_nodes)
        B.vs["type"] = types
        p1, p2 = B.bipartite_projection(multiplicity=False)

        gUNig = None
        if project_on == "user":
            gUNig = p1
        else:
            gUNig = p2

        if SNAP_LOADED:
            # SNAP graph object construction
            gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                gUNsn.AddEdge(e.source, e.target)

        return gUNig, gUNsn

    def twitter_build_undirected_graph(self, df, weight_filter=1):
        """
        build twitter undirected graph
        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])

        edgelist = get_undirected_edgelist(df, weight_filter)

        gUNsn = None

        # iGraph graph object construction
        gUNig = ig.Graph.TupleList(edgelist, directed=False, weights=True)
        gUNig.simplify(combine_edges='sum')

        if SNAP_LOADED:
            # SNAP graph object construction
            gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                gUNsn.AddEdge(e.source, e.target)

        return gUNig, gUNsn

    def telegram_build_undirected_graph(self, df, weight_filter=1):
        """
        build telegram undirected graph
        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = get_undirected_edgelist(df, weight_filter)
        gUNsn = None

        # iGraph graph object construction
        gUNig = ig.Graph.TupleList(edgelist, directed=False)
        gUNig.simplify(combine_edges='sum')

        if SNAP_LOADED:
            # SNAP graph object construction
            gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                gUNsn.AddEdge(e.source, e.target)

        return gUNig, gUNsn

    def reddit_build_undirected_graph(self, df, weight_filter=1):
        """
        build reddit undirected graph
        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = get_undirected_edgelist(df, weight_filter)
        gUNsn = None
        # iGraph Graph object construction
        gUNig = ig.Graph.TupleList(edgelist, directed=False, weights=True)
        gUNig.simplify(combine_edges='sum')

        if SNAP_LOADED:
            # SNAP graph object construction
            gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                gUNsn.AddEdge(e.source, e.target)

        return gUNig, gUNsn

    def youtube_build_undirected_graph(self, df, weight_filter=1):
        """
        build youtube undirected graph
        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = get_undirected_edgelist(df, weight_filter)
        gUNsn = None
        # iGraph Graph object construction
        gUNig = ig.Graph.TupleList(edgelist, directed=False, weights=True)
        gUNig.simplify(combine_edges='sum')

        if SNAP_LOADED:
            # SNAP graph object construction
            gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                gUNsn.AddEdge(e.source, e.target)

        return gUNig, gUNsn

    def combined_build_undirected_graph(self, df, weight_filter=1):
        """
        build combined undirected graph
        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = get_undirected_edgelist(df, weight_filter)
        gUNsn = None
        # iGraph Graph object construction
        gUNig = ig.Graph.TupleList(edgelist, directed=False, weights=True)
        gUNig.simplify(combine_edges='sum')

        if SNAP_LOADED:
            # SNAP graph object construction
            gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                gUNsn.AddEdge(e.source, e.target)

        return gUNig, gUNsn

    @check_empty(default=None)
    def cascade_collection_participation_gini(self, platform='all', community_grouper=None):
        """

        Description: gini coefficient summarizing to what extent are posts sets of cascades disproportionately
        authored by a subset of the users who participate in the cumulative snapshots over time?

        """
        res = []
        for ts in self.time_series_dfs[platform]:
            all_node_users = self.time_series_dfs[platform][ts][self.user_col].values
            res.append(
                {TIMESTEP_COLUMN: ts, VALUE_COLUMN: pysal.explore.inequality.gini.Gini(list(Counter(all_node_users).values())).g})
        return pd.DataFrame(res)


    @check_empty(default=None)
    @check_root_only(default=None)
    def cascade_collection_participation_palma(self, platform='all', community_grouper=None):
        """

        Description: palma ratio summarizing to what extent are posts within a collection of information cascades disproportionately
        authored by a subset of the users who participate in the cumulative snapshots over time ?

        """
        res = []
        for ts in self.time_series_dfs[platform]:
            all_node_users = self.time_series_dfs[platform][ts][self.user_col].values
            res.append({TIMESTEP_COLUMN: ts, VALUE_COLUMN: palma_ratio(list(Counter(all_node_users).values()))})
        return pd.DataFrame(res)


    def persistence_of_connectivity(self, platform='all', fill_missing_periods=False):
        if platform == 'all':
            platform_df = self.main_df
        else:
            platform_df = self.main_df[self.main_df['platform'] == platform]
        df = self.get_parent_uids(platform_df).dropna(subset=['parentUserID'])

        if len(platform_df) == 0:
            warning_message = 'No data for platform \'{}\''.format(platform)
            warnings.warn(warning_message)
            return
        elif len(df) == 0:
            warning_message = 'No parentUserID data for platform \'{}\' -- missing data of parent of nodes (e.g. userIDs for parent content or all details of parent content)\n'.format(platform) + self.informationIDs_string
            warnings.warn(warning_message)
            return


        df = df.rename(columns={'{}_period'.format(platform): 'time'})
        df['edge'] = [get_edge_string(n, p) for n, p in zip(df['nodeUserID'], df['parentUserID'])]
        first_bin_prev_time = 'FIRST BIN'
        timeset = set(df['time'].unique().tolist())
        timeset = [x for x in timeset if str(x) != 'nan']
        timebins = sorted(list(timeset))
        prev = {timebins[i]: timebins[i - 1] for i in range(1, len(timebins))}
        prev[timebins[0]] = first_bin_prev_time

        df['edge'] = [get_edge_string(n, p) for n, p in zip(df['nodeUserID'], df['parentUserID'])]

        edgesets = df.groupby('time', as_index=False).agg({'edge': lambda x: list(x)})
        previousbin = edgesets.copy().rename(columns={'time': 'prev_time', 'edge': 'prev_edges'})
        currentbin = edgesets.copy().rename(columns={'edge': 'edges'})
        currentbin['prev_time'] = [prev[time] for time in currentbin['time']]
        currentbin = currentbin[currentbin['prev_time'] != first_bin_prev_time]

        joined = pd.merge(currentbin, previousbin, on=['prev_time'], how='left')

        def persistence_of_connections(previous_edges, current_edges):
            n_consistent_edges = len(set(previous_edges).intersection(set(current_edges)))
            n_old_edges = len(set(previous_edges))
            return n_consistent_edges / float(n_old_edges)

        joined['persistence_of_connectivity'] = [persistence_of_connections(previous_edges, current_edges) for
                                                 previous_edges, current_edges in
                                                 zip(joined['prev_edges'], joined['edges'])]

        if fill_missing_periods:
            res_arr = []
            for ts in self.time_series_dfs[platform]:
                if ts in timeset:
                    res_arr.append([ts, joined[joined['time'] == ts]['persistence_of_connectivity']])
                else:
                    res_arr.append([ts, np.nan])
            res = pd.DataFrame(data=res_arr, columns=['Time', 'Persistence of Connectivity'])
            return res


        else:
            return joined[['time', 'persistence_of_connectivity']].rename(
                columns={'time': TIMESTEP_COLUMN, 'persistence_of_connectivity': VALUE_COLUMN})

    def fluctuability(self, platform='all'):
        if platform == 'all':
            platform_df = self.main_df
        else:
            platform_df = self.main_df[self.main_df['platform'] == platform]
        df = self.get_parent_uids(platform_df).dropna(subset=['parentUserID'])

        if len(platform_df) == 0:
            warning_message = '\nNo data for platform \'{}\''.format(platform)
            warnings.warn(warning_message)
            return
        elif len(df) == 0:
            warning_message = '\nNo parentUserID data for platform \'{}\' -- missing data of parent of nodes (e.g. userIDs for parent content or all details of parent content)\n'.format(platform) + self.informationIDs_string
            warnings.warn(warning_message)
            return

        df = df.rename(columns={'{}_period'.format(platform): 'time'})
        df['edge'] = [get_edge_string(n, p) for n, p in zip(df['nodeUserID'], df['parentUserID'])]
        edges = df[['edge', 'time']]
        uniq_edges_per_period = edges.drop_duplicates(['edge', 'time'])
        uniq_edges_whole_span = edges['edge'].unique()

        return float(len(uniq_edges_whole_span)) / float(len(uniq_edges_per_period))



