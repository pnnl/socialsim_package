import pandas as pd
import igraph as ig
import snap   as sn
import numpy  as np

from time import time

import community
import tqdm
import os

from .measurements import MeasurementsBaseClass

class SocialStructureMeasurements(MeasurementsBaseClass):
    """
    This class implements Network specific measurements. It uses iGraph and 
    SNAP libraries with Python interfaces. For installation information please 
    visit the websites for the two packages.

    iGraph-Python at http://igraph.org/python/
    SNAP Python at https://snap.stanford.edu/snappy/
    """
    def __init__(self, dataset, configuration, metadata, platform, test=False, 
        log_file='network_measurements_log.txt'):
        super(SocialStructureMeasurements, self).__init__(dataset, 
            configuration, log_file=log_file)

        self.measurement_type = 'social_structure'
        self.main_df = dataset

        

        if platform=='reddit':
            build_undirected_graph = self.reddit_build_undirected_graph
        elif platform=='twitter':
            build_undirected_graph = self.twitter_build_undirected_graph
        elif platform=='github':
            build_undirected_graph = self.github_build_undirected_graph

        build_undirected_graph(self.main_df)

    def mean_shortest_path_length(self):
        return sn.GetBfsEffDiam(self.gUNsn, 500, False)

    def number_of_nodes(self):
        return ig.Graph.vcount(self.gUNig)

    def number_of_edges(self):
        return ig.Graph.ecount(self.gUNig)

    def density(self):
        return ig.Graph.density(self.gUNig)

    def assortativity_coefficient(self):
        return ig.Graph.assortativity_degree(self.gUNig)

    def number_of_connected_components(self):
        return len(ig.Graph.components(self.gUNig, mode="WEAK"))

    def average_clustering_coefficient(self):
        return sn.GetClustCf(self.gUNsn)

    def max_node_degree(self):
        return max(ig.Graph.degree(self.gUNig))

    def mean_node_degree(self):
        return 2.0*ig.Graph.ecount(self.gUNig)/ig.Graph.vcount(self.gUNig)

    def degree_distribution(self):
        degVals = ig.Graph.degree(self.gUNig)
        return pd.DataFrame([{'node': idx, 'value': degVals[idx]} for idx in range(self.gUNig.vcount())])

    def community_modularity(self):
        return ig.Graph.modularity(self.gUNig,ig.Graph.community_multilevel(self.gUNig))

    def get_parent_uids(self,df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", user_col="nodeUserID"):
        """
        :return: adds parentUserID column with user id of the parent if it exits in df
        if it doesn't exist, uses the user id of the root instead
        if both doesn't exist: NaN
        """
        tweet_uids = pd.Series(df[user_col].values, index=df[node_col]).to_dict()
        df['parentUserID'] = df[parent_node_col].map(tweet_uids)
        df.loc[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull()), 'parentUserID'] = \
            df[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull())][root_node_col].map(tweet_uids)
        return df


    def github_build_undirected_graph(self, df, project_on='nodeID'):
        self.main_df = self.main_df[['nodeUserID','nodeID']]

        # below line will be deleted if commenting it out produces no errors
        # left_nodes = np.array(self.main_df['nodeUserID'].unique().tolist())
        right_nodes = np.array(self.main_df['nodeID'].unique().tolist())
        el = self.main_df.apply(tuple, axis=1).tolist()
        edgelist = list(set(el))

        #iGraph graph object construction
        B = ig.Graph.TupleList(edgelist, directed=False)
        names = np.array(B.vs["name"])
        types = np.isin(names, right_nodes)
        B.vs["type"] = types
        p1, p2 = B.bipartite_projection(multiplicity=False)

        self.gUNig = None
        if project_on=="user":
            self.gUNig = p1
        else:
            self.gUNig = p2

        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source,e.target)


    def twitter_build_undirected_graph(self, df):
        """
        Description:

        Input:

        Output:

        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = df[['nodeUserID','parentUserID']].apply(tuple,axis=1).tolist()

        #iGraph graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False)

        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source, e.target)


    def reddit_build_undirected_graph(self, df):
        """
        Description:

        Input:

        Output:

        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = df[['nodeUserID', 'parentUserID']].apply(tuple,axis=1)
        edgelist = edgelist.tolist()

        #iGraph Graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False)

        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source, e.target)
