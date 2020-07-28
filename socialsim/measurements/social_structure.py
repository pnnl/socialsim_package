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
                 node="", weight_filter=1, directed=True, node_list=None):

        super(SocialStructureMeasurements, self).__init__(dataset,
            configuration, log_file=log_file)

        self.measurement_type = 'social_structure'
        self.main_df = dataset

        # Subset data for a specific informationID
        if node != "":
            self.main_df = self.main_df.loc[self.main_df["informationID"]==node].copy()

        if metadata is None or metadata.node_list is None or metadata.node_list == 'all':
            if node_list == "all":
                self.node_list = self.main_df['informationID'].tolist()
            elif node_list is not None:
                self.node_list = node_list
            else:
                self.node_list = []
        else:
            self.node_list = metadata.node_list
            
        random.seed(37)

        if platform in ['twitter', 'reddit', 'telegram', 'youtube']:
            build_graph = self.social_media_build_graph
        elif platform=='github':
            build_graph = self.github_build_graph
        else:
            # unknown platform, skip graph creation
            return


        self.gUNigs = {}
        for info_id in self.node_list:
#        for info_id in self.main_df['informationID'].unique():
            graph = build_graph(self.main_df[self.main_df['informationID'] == info_id],
                                                          weight_filter=weight_filter,
                                           directed=directed)
            if not graph.vcount() == 0:
                self.gUNigs[info_id] = graph
        self.gUNig = build_graph(self.main_df,weight_filter=weight_filter,
                                 directed=directed)

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

    def mean_shortest_path_length(self, node_level=False):
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

            if not node_level:
                graphs = {'all':self.gUNig}
            else:
                graphs = self.gUNigs

            meas = {}
            for key,graph in graphs.item():
                if ig.Graph.vcount(graph) == 0:
                    warnings.warn('Empty graph')
                    return 0
                shortest_paths = graph.shortest_paths_dijkstra(mode='ALL')
                shortest_paths_cleaned =np.array(shortest_paths).flatten()
                shortest_paths_cleaned =shortest_paths_cleaned[np.isfinite(shortest_paths_cleaned)]
                meas[key] = np.percentile([float(x) for x in shortest_paths_cleaned],90)

            if not node_level:
                return meas['all']
            else:
                return meas


    def number_of_nodes(self, node_level=False):
        """
        Measurement: number_of_nodes

        Description: Calculate the number of nodes in the graph

        Input: Graph

        Output: Int.

        """

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():
            meas[key] = ig.Graph.vcount(graph)

        if not node_level:
            return meas['all']
        else:
            return meas

    def number_of_edges(self, node_level=False):
        """
        Measurement: number_of_edges

        Description: Calculate the number of edges in the graph

        Input: Graph

        Output: Int.

        """

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():
            meas[key] = ig.Graph.ecount(self.gUNig)

        if not node_level:
            return meas['all']
        else:
            return meas


    def density(self, node_level=False):
        """
        Measurement: density

        Description: Calculate density of graph

        Input: Graph

        Output: Int.

        """


        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():
            meas[key] = ig.Graph.density(self.gUNig)

        if not node_level:
            return meas['all']
        else:
            return meas


    def assortativity_coefficient(self, node_level=False):
        """
        Measurement: assortativity_coefficient

        Description: Calculate the assortativity degree coefficient of the graph

        Input: Graph

        Output: Float

        """

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():
            meas[key] = ig.Graph.assortativity_degree(self.gUNig)

        if not node_level:
            return meas['all']
        else:
            return meas

    def number_of_connected_components(self, node_level=False):
        """
        Measurement: number_of_connected_components

        Description: Calculate the number of connected components in the graph

        Input: Graph

        Output: Int.

        """

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():
            meas[key] = len(ig.Graph.components(graph, mode="WEAK"))

        if not node_level:
            return meas['all']
        else:
            return meas

    def largest_connected_component(self, node_level=False):

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():
            meas[key] = ig.Graph.vcount(ig.Graph.components(graph, mode="WEAK").giant())

        if not node_level:
            return meas['all']
        else:
            return meas


    def average_clustering_coefficient(self, node_level=False):
        """
        Measurement: average_clustering_coefficient

        Description: Calculate the average clustering coefficient of the graph

        Input: Graph

        Output: Float

        """
        if SNAP_LOADED:
            return sn.GetClustCf(self.gUNsn)
        else:

            if not node_level:
                graphs = {'all':self.gUNig}
            else:
                graphs = self.gUNigs

            meas = {}
            for key,graph in graphs.items():
                meas[key] = self.gUNig.transitivity_avglocal_undirected(mode='zero')

            if not node_level:
                return meas['all']
            else:
                return meas


    def max_node_degree(self, node_level=False):
        """
        Measurement: max_node_degree

        Description: Determine the max degree of any node in the graph

        Input: Graph

        Output: Int.

        """

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():

            if ig.Graph.vcount(graph) == 0:
                warnings.warn('Empty graph',key)
                continue

            meas[key] = max(ig.Graph.degree(graph))

        if not node_level:
            return meas['all']
        else:
            return meas


    def mean_node_degree(self, node_level=False):
        """
        Measurement: mean_node_degree

        Description: Calculate the mean node degree of the graph

        Input: Graph

        Output: Float

        """

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():

            if ig.Graph.vcount(graph) == 0:
                warnings.warn('Empty graph',key)
                continue

            meas[key] = 2.0*ig.Graph.ecount(graph)/ig.Graph.vcount(graph)

        if not node_level:
            return meas['all']
        else:
            return meas


    def degree_distribution(self,node_level=False, mode='ALL'):
        """
        Measurement: degree_distribution

        Description: Get the distribution of all node degrees in the graph

        Input: Graph

        Output: DataFrame

        """
        
        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():

            if ig.Graph.vcount(graph) == 0:
                warnings.warn('Empty graph',key)
                continue

            vertices = [ str(v.attributes()['name']) for v in graph.vs]

            if ig.Graph.vcount(graph) > 1:
                degVals = graph.degree(vertices,mode=mode)
            elif ig.Graph.vcount(graph) == 1:
                degVals = [1]
                
            meas[key] = pd.DataFrame([{'node': vertices[idx],
                'value': degVals[idx]} for idx in range(len(vertices))])

        if not node_level:
            return meas['all']
        else:
            return meas

    def pagerank_distribution(self,node_level=False):
        """
        Measurement: pagerank_distribution

        Description: Pagerank scores of all nodes in the graph

        Input: Graph, with "weight" attribute on edges

        Output: DataFrame

        """

        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():

            if ig.Graph.vcount(graph) == 0:
                warnings.warn('Empty graph',key)
                continue

            vertices = [ str(v.attributes()['name']) for v in graph.vs]
            prVals=graph.pagerank(vertices,weights='weight')
            meas[key] = pd.DataFrame([{'node': vertices[idx],
                'value': prVals[idx]} for idx in range(len(vertices))])

        if not node_level:
            return meas['all']
        else:
            return meas

    def community_modularity(self, node_level=False):
        """
        Measurement: community_modularity

        Description: Calculate the community modularity of the graph

        Input: Graph

        Output: Float

        """
        random.seed(37)


        if not node_level:
            graphs = {'all':self.gUNig}
        else:
            graphs = self.gUNigs

        meas = {}
        for key,graph in graphs.items():

            if ig.Graph.vcount(graph) == 0:
                warnings.warn('Empty graph',key)
                continue

            meas[key] = ig.Graph.modularity(graph,ig.Graph.community_multilevel(graph))

        if not node_level:
            return meas['all']
        else:
            return meas

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


    def github_build_graph(self, df, project_on='nodeID', directed=False):
        df = df[['nodeUserID','nodeID']]

        right_nodes = np.array(df['nodeID'].unique().tolist())
        el = df.apply(tuple, axis=1).tolist()
        edgelist = list(set(el))

        #iGraph graph object construction
        B = ig.Graph.TupleList(edgelist, directed=directed)
        names = np.array(B.vs["name"])
        types = np.isin(names, right_nodes)
        B.vs["type"] = types
        p1, p2 = B.bipartite_projection(multiplicity=False)

        if project_on=="user":
            gUNig = p1
        else:
            gUNig = p2

        #if graph empty, convert to singleton graph
        if gUNig.vcount()==0:
            gUNig.add_vertex('dummy')

        if SNAP_LOADED:
            #SNAP graph object construction
            self.gUNsn = sn.TUNGraph.New()
            for v in gUNig.vs:
                self.gUNsn.AddNode(v.index)
            for e in gUNig.es:
                self.gUNsn.AddEdge(e.source,e.target)

        return gUNig

    def get_edgelist(self, df, weight_filter, directed=True):
        edgelist_df = df[['nodeUserID','parentUserID']].copy()

        if not directed:
            edgelist_df['edge'] = [sorted([n,p]) for n,p in zip(df['nodeUserID'],df['parentUserID'])]
        else:
            edgelist_df['edge'] = [[n,p] for n,p in zip(df['nodeUserID'],df['parentUserID'])]

        edgelist_df['userA'] = ['u-' + str(x[0]) for x in edgelist_df['edge']]
        edgelist_df['userB'] = ['u-' + str(x[1]) for x in edgelist_df['edge']]

        edgelist_df = edgelist_df.groupby(['userA','userB']).size().reset_index().rename(columns={0:'count'})
        edgelist_df = edgelist_df[edgelist_df['count'] >= weight_filter]
        edgelist = edgelist_df[['userA','userB','count']].apply(tuple,axis=1).tolist()

        return edgelist

    def social_media_build_graph(self, df, weight_filter=1, directed=False):
        """
        Description:

        Input:

        Output:

        """
        print('Building directed=',directed,'graph')
        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        
        edgelist = self.get_edgelist(df, weight_filter, directed=directed)

        #iGraph graph object construction
        graph = ig.Graph.TupleList(edgelist, directed=directed, weights=True)
        graph.simplify(combine_edges='sum')

        #if graph empty, convert to singleton graph
        if graph.vcount()==0:
            graph.add_vertex('dummy')

        if SNAP_LOADED:
            #SNAP graph object construction
            self.gUNsn = sn.TUNGraph.New()
            for v in graph.vs:
                self.gUNsn.AddNode(v.index)
            for e in graph.es:
                self.gUNsn.AddEdge(e.source, e.target)


        return(graph)




