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

from time import time

import tqdm
import os

import re

import random


from .measurements import MeasurementsBaseClass

class SocialStructureMeasurements(MeasurementsBaseClass):
    """
    This class implements Network specific measurements. It uses iGraph and 
    SNAP libraries with Python interfaces. For installation information please 
    visit the websites for the two packages.

    iGraph-Python at http://igraph.org/python/
    SNAP Python at https://snap.stanford.edu/snappy/
    """
    def __init__(self, dataset, platform='', configuration = {}, 
                 metadata = None, test=False, 
                 log_file='network_measurements_log.txt', plot_graph=False, 
                 node="", weight_filter=1):

        super(SocialStructureMeasurements, self).__init__(dataset, 
            configuration, log_file=log_file)

        self.measurement_type = 'social_structure'
        self.main_df = dataset

        # Subset data for a specific informationID
        if node != "":
            self.main_df = self.main_df.loc[self.main_df["informationID"]==node].copy()

        random.seed(37)

        if platform=='reddit':
            build_undirected_graph = self.reddit_build_undirected_graph
        elif platform=='twitter':
            build_undirected_graph = self.twitter_build_undirected_graph
        elif platform=='telegram':
            build_undirected_graph = self.telegram_build_undirected_graph
        elif platform=='github':
            build_undirected_graph = self.github_build_undirected_graph
        elif platform=='youtube':
            build_undirected_graph = self.youtube_build_undirected_graph
        else:
            # unknown platform, skip graph creation
            return

        build_undirected_graph(self.main_df,weight_filter=weight_filter)

        if plot_graph:
            node = node.replace("/", "__")
            graph_file_name = str(node)+"_"+str(platform)+"_igraph_network.png"
 

            visual_style = {}
            visual_style["vertex_size"] = 3
            try:
                visual_style["edge_width"] = self.gUNig.es['weight'] * (10.0 / np.max(self.gUNig.es['weight']))
            except:
                ''
            visual_style["bbox"] = (300, 300)
            
            try:
                self.graph = ig.plot(self.gUNig,layout=self.gUNig.layout('fr',weights=self.gUNig.es['weight']),inline=True,**visual_style)
            except:
                self.graph = ig.plot(self.gUNig,layout=self.gUNig.layout('fr'),inline=True,**visual_style)

        random.seed(37)


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

    def mean_shortest_path_length(self):
        """
        Measurement: mean_shortest_path_length

        Description: Calculate the mean shortest path

        Input: Graph

        Output:

        """
        if SNAP_LOADED:
            if self.gUNsn.Empty():
                warnings.warn('Empty graph')
                return 0
            return sn.GetBfsEffDiam(self.gUNsn, self.gUNsn.GetNodes(), False)
        else:
            if ig.Graph.vcount(self.gUNig) == 0:
                warnings.warn('Empty graph')
                return 0
            shortest_paths = self.gUNig.shortest_paths_dijkstra(mode='ALL')
            shortest_paths_cleaned =np.array(shortest_paths).flatten()
            shortest_paths_cleaned =shortest_paths_cleaned[np.isfinite(shortest_paths_cleaned)]
            return np.percentile([float(x) for x in shortest_paths_cleaned],90)

    def number_of_nodes(self):
        """
        Measurement: number_of_nodes

        Description: Calculate the number of nodes in the graph

        Input: Graph

        Output: Int.

        """
        return ig.Graph.vcount(self.gUNig)

    def number_of_edges(self):
        """
        Measurement: number_of_edges

        Description: Calculate the number of edges in the graph

        Input: Graph

        Output: Int.

        """
        return ig.Graph.ecount(self.gUNig)

    def density(self):
        """
        Measurement: density

        Description: Calculate density of graph

        Input: Graph

        Output: Int.

        """
        return ig.Graph.density(self.gUNig)

    def assortativity_coefficient(self):
        """
        Measurement: assortativity_coefficient

        Description: Calculate the assortativity degree coefficient of the graph

        Input: Graph

        Output: Float

        """
        return ig.Graph.assortativity_degree(self.gUNig)

    def number_of_connected_components(self):
        """
        Measurement: number_of_connected_components

        Description: Calculate the number of connected components in the graph

        Input: Graph

        Output: Int.

        """
        return len(ig.Graph.components(self.gUNig, mode="WEAK"))

    def largest_connected_component(self):
        return(ig.Graph.vcount(ig.Graph.components(self.gUNig, mode="WEAK").giant()))


    def average_clustering_coefficient(self):
        """
        Measurement: average_clustering_coefficient

        Description: Calculate the average clustering coefficient of the graph

        Input: Graph

        Output: Float

        """
        if SNAP_LOADED:
            return sn.GetClustCf(self.gUNsn)
        else:
            return self.gUNig.transitivity_avglocal_undirected(mode='zero')

    def max_node_degree(self):
        """
        Measurement: max_node_degree

        Description: Determine the max degree of any node in the graph

        Input: Graph

        Output: Int.

        """
        if ig.Graph.vcount(self.gUNig) == 0:
            warnings.warn('Empty graph')
            return 0

        return max(ig.Graph.degree(self.gUNig))

    def mean_node_degree(self):
        """
        Measurement: mean_node_degree

        Description: Calculate the mean node degree of the graph

        Input: Graph

        Output: Float

        """
        if len(list(self.gUNig.vs)) == 0:
            warnings.warn('Empty graph')
            return 0

        return 2.0*ig.Graph.ecount(self.gUNig)/ig.Graph.vcount(self.gUNig)

    def degree_distribution(self):
        """
        Measurement: degree_distribution

        Description: Get the distribution of all node degrees in the graph

        Input: Graph

        Output: DataFrame

        """
        vertices = [ v.attributes()['name'] for v in self.gUNig.vs]
        degVals = self.gUNig.degree(vertices) 
        return pd.DataFrame([{'node': vertices[idx], 'value': degVals[idx]} for idx in range(len(vertices))])

    def community_modularity(self):
        """
        Measurement: community_modularity

        Description: Calculate the community modularity of the graph

        Input: Graph

        Output: Float

        """
        random.seed(37)

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
        
        df = df[df['nodeUserID'] != df['parentUserID']]
        
        return df


    def github_build_undirected_graph(self, df, project_on='nodeID', weight_filter=1):
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

        if SNAP_LOADED:
            #SNAP graph object construction
            self.gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                self.gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                self.gUNsn.AddEdge(e.source,e.target)

    def get_undirected_edgelist(self, df, weight_filter):
        edgelist_df = df[['nodeUserID','parentUserID']].copy()
        edgelist_df['edge'] = [sorted([n,p]) for n,p in zip(df['nodeUserID'],df['parentUserID'])]
        edgelist_df['userA'] = [x[0] for x in edgelist_df['edge']]
        edgelist_df['userB'] = [x[1] for x in edgelist_df['edge']]

        edgelist_df = edgelist_df.groupby(['userA','userB']).size().reset_index().rename(columns={0:'count'})
        edgelist_df = edgelist_df[edgelist_df['count'] >= weight_filter]
        edgelist = edgelist_df[['userA','userB','count']].apply(tuple,axis=1).tolist()

        return edgelist

    def twitter_build_undirected_graph(self, df, weight_filter=1):
        """
        Description:

        Input:

        Output:

        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])


        edgelist = self.get_undirected_edgelist(df, weight_filter)


        #iGraph graph object construction 
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False, weights=True)
        self.gUNig.simplify(combine_edges='sum')
        

        if SNAP_LOADED:
            #SNAP graph object construction
            self.gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                self.gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                self.gUNsn.AddEdge(e.source, e.target)



    def telegram_build_undirected_graph(self, df, weight_filter=1):
        """
        Description:

        Input:

        Output:

        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])

        edgelist = self.get_undirected_edgelist(df, weight_filter)

        #iGraph graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False)
        self.gUNig.simplify(combine_edges='sum')

        if SNAP_LOADED:
            #SNAP graph object construction
            self.gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                self.gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                self.gUNsn.AddEdge(e.source, e.target)


    def reddit_build_undirected_graph(self, df, weight_filter=1):
        """
        Description:

        Input:

        Output:

        """
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])

        edgelist = self.get_undirected_edgelist(df, weight_filter)

        #iGraph Graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False, weights = True)
        self.gUNig.simplify(combine_edges='sum')


        if SNAP_LOADED:
            #SNAP graph object construction
            self.gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                self.gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                self.gUNsn.AddEdge(e.source, e.target)


    def youtube_build_undirected_graph(self, df, weight_filter=1):
        """
        build youtube undirected graph
        """

        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])

        edgelist = self.get_undirected_edgelist(df, weight_filter)

        #iGraph Graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False, weights = True)
        self.gUNig.simplify(combine_edges='sum')


        if SNAP_LOADED:
            #SNAP graph object construction
            self.gUNsn = sn.TUNGraph.New()
            for v in self.gUNig.vs:
                self.gUNsn.AddNode(v.index)
            for e in self.gUNig.es:
                self.gUNsn.AddEdge(e.source, e.target)

