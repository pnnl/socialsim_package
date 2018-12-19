import igraph as ig
import snap   as sn

class NetworkMeasurements(MeasurementsBaseclass):
    def __init__(self, platform):

        if platform=='github':
            build_undirected_graph = self._github_build_undirected_graph

        elif platform=='twitter':
            build_undirected_graph = self._twitter_build_undirected_graph

        elif platform=='reddit':
            build_undirected_graph = self._reddit_build_undirected_graph

        self.undirected_graph = None

        self.measurements = {
        'population':
            {
            'mean shortest path length': MeanShortestPathLength(),
            'number of nodes': NumberOfNodes()
            }
        }

    def _github_build_undirected_graph(self, dataset):
        """
        Description:

        Inputs:

        Outputs:
            None
        """
        dataset = dataset[['nodeUserID','nodeID']]

        left_nodes = np.array(dataset['nodeUserID'].unique().tolist())
        right_nodes = np.array(dataset['nodeID'].unique().tolist())
        el = dataset.apply(tuple, axis=1).tolist()
        edgelist = list(set(el))

        #iGraph Graph object construction
        B = ig.Graph.TupleList(edgelist, directed=False)
        names = np.array(B.vs["name"])
        types = np.isin(names,right_nodes)
        B.vs["type"] = types
        p1,p2 = B.bipartite_projection(multiplicity=False)

        self.gUNig = None
        if (self.project_on == "user"):
            self.gUNig = p1
        else:
            self.gUNig = p2

        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source,e.target)


    def _twitter_build_undirected_graph(self, dataset):
        """
        Description:

        Inputs:

        Outputs:
            None
        """
        dataset = self.get_parent_uids(dataset)
        dataset = dataset.dropna(subset=['parentUserID'])

        edgelist = dataset[['nodeUserID','parentUserID']]
        edgelist = edgelist.apply(tuple, axis=1).tolist()

        #iGraph Graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False)

        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source,e.target)


    def _reddit_build_undirected_graph(self, dataset):
        """
        Description:

        Inputs:

        Outputs:
            None
        """

        dataset = self.get_parent_uids(dataset)
        dataset = dataset.dropna(subset=['parentUserID'])

        edgelist = dataset[['nodeUserID','parentUserID']]
        edgelist = edgelist.apply(tuple, axis=1).tolist()

        #iGraph Graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False)

        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source, e.target)

"""
These are all of the Network Measurements.
"""

class MeanShortestPathLength(Measurement):
    def __init__(self):
        """
        Description:

        Inputs:

        Outputs:
            :result:
        """
        self.scale = 'population'
        self.name  = 'mean shortest path length'

    def run(dataset):
        """
        Description:

        Inputs:

        Outputs:
            :result:
        """
        result = sn.GetBfsEffDiamAll(dataset, 500, False)[3]

        return result

class NumberOfNodes(Measurement):
    def __init__(self):
        """
        Description:

        Inputs:

        Outputs:
            None
        """
        self.scale = 'population'
        self.name  = 'number of nodes'

    def run(dataset):
        """
        Description:

        Inputs:

        Outputs:
            :result:
        """

        result = ig.Graph.vcount(dataset)

        return result
