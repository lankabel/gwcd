import igraph as ig
import pandas as pd
import os
import click
from array import array
from collections import defaultdict
from operator import itemgetter
import numpy as np
from optparse import OptionParser
from textwrap import dedent
import math

import logging
import os
import sys

# Loading the graph
graph_name = './graph_taxi_all_length_3.gmlz'

if os.path.isfile(graph_name):
    g = ig.Graph.Read_GraphMLz(graph_name)
    print("Graph " + graph_name + " exists. Loading graph.")
    number_of_nodes = len(g.vs)
    number_of_links = len(g.es)
else:
    print("File " + graph_name + " does not exist. Creating graph")
    df_start = pd.read_csv("taxi_yellow_2017_07_averaged.csv", delimiter=',')
    df_start = df_start[df_start['pulocationid'] < 264]
    df_start = df_start[df_start['dolocationid'] < 264]
    df_start=df_start[df_start['average_trip_distance']>0]



    f = lambda x: x['passenger_count'] * math.pow(x['average_trip_distance'], -x['average_trip_distance']/avg)

    tuples = [tuple(x) for x in df_start[df_start.columns[0:2]].values]

    g = ig.Graph.TupleList(tuples, directed=False, edge_attrs=['weight'])
    # g.es['weight'] = df[df.columns[3]].values
    # g.es['weight'] = df_start['weight'].values
    g.es['passenger_count'] = df_start[df_start.columns[2]].values
    g.es['trip_distance'] = df_start[df_start.columns[3]].values
    # g = ig.Graph.Read_Ncol('all_links1.csv',names=True, directed=True)
    number_of_nodes = len(g.vs)
    number_of_links = len(g.es)

    # load coordinates as dataframe

    print('Writing node coordinates.'
          )
    coordinates = pd.read_csv("taxi_zones_centroids.csv", delimiter=",")

    # TODO: make variables here changeable from load
    with click.progressbar(g.vs) as bar:
        for vertex in bar:
            # print(vertex['name'])
            vertex['x'], vertex['y'] = (float(coordinates.loc[coordinates['fid'] == vertex['name']]['x']),
                                        float(coordinates.loc[coordinates['fid'] == vertex['name']]['y']))

    print('Coordinate loading completed.')

    with click.progressbar(g.es) as links:
        for link in links:
            link['length'] = np.sqrt(pow((g.vs[link.source]['x'] - g.vs[link.target]['x']),2) + pow((g.vs[link.source]['y'] - g.vs[link.target]['y']),2))
    df_start['length'] = g.es['length']
    avg = df_start['length'].mean()

    a = lambda x: x['passenger_count'] * math.pow(x['length'], -x['length']/avg)


    df_start['weight'] = df_start.apply(a, axis=1)
    g.es['weight'] = df_start['weight'].values


    g.write_graphmlz(graph_name)


# Graph loading finished
# attr = "weight"
# degrees = g.degree()
# strengths = g.strength(weights=attr)
# weights = g.es[attr]

# adjedgelist = []
# get_eid = g.get_eid  # prelookup
# with click.progressbar(range(g.vcount())) as bar:
#     for i in bar:
#         # print (i)
#         weis = dict((j.target, j['weight']) for j in g.es.select(_source=0))
#         weis[i] = strengths[i] / degrees[i]
#         adjedgelist.append(weis)
#
#     sqsums = [sum(value * value for value in vec.values())
#               for vec in adjedgelist]


# with click.progressbar(g.es) as bar:
#     for link in bar:
#         k = g.vs[link.source]

class TanimotoSimilarityCalculator(object):
    """Calculates pairwise Tanimoto coefficients on a given weighted
    graph. When calculating the similarities, it is assumed that every
    vertex is linked to itself with an edge whose weight is equal to the
    average weight of edges adjacent to the vertex."""

    def __init__(self, graph, attr="weight", att2='trip_distance'):
        degrees = graph.degree()
        strengths = graph.strength(weights=attr)
        distances = graph.strength(weights=att2)

        # weights = graph.es[attr]

        self._adjedgelist = []
        self._ddjedgelist = []
        get_eid = graph.get_eid  # prelookup

        print('Calculating Tannimoto')
        with click.progressbar(range(g.vcount())) as bar:
            for i in bar:
                # print (i)
                weis = dict((j.target, j['weight']) for j in g.es.select(_source=i))
                weis[i] = strengths[i] / degrees[i]
                self._adjedgelist.append(weis)

        print('Calculating distances')
        with click.progressbar(range(g.vcount())) as bar:
            for i in bar:
                # print (i)
                deis = dict((j.target, math.pow(j['trip_distance'], 2)) for j in g.es.select(_source=i))
                deis[i] = distances[i] / degrees[i]
                self._ddjedgelist.append(deis)

        self._sqsums = [sum(value * value for value in vec.values())
                        for vec in self._adjedgelist]
        self._dqsums = [sum(value * value for value in vec.values())
                        for vec in self._ddjedgelist]

    def get_similarity_weight(self, v1, v2):
        """Returns the Tanimoto coefficient of the two given vertices,
        assuming that both of them are linked to themselves."""
        vec1, vec2 = self._adjedgelist[v1], self._adjedgelist[v2]

        if len(vec1) > len(vec2):
            # vec1 must always be the smaller
            vec1, vec2 = vec2, vec1

        numerator = sum(value * vec2.get(key, 0)
                        for key, value in vec1.items())
        return numerator / (self._sqsums[v1] + self._sqsums[v2] - numerator)

    def get_similarity_dist(self, v1, v2):
        """Returns the Cosine coefficient of the two given vertices,
        assuming that both of them are linked to themselves."""
        vec1, vec2 = self._ddjedgelist[v1], self._ddjedgelist[v2]

        if len(vec1) > len(vec2):
            # vec1 must always be the smaller
            vec1, vec2 = vec2, vec1

        numerator = sum(value * vec2.get(key, 0)
                        for key, value in vec1.items())
        return numerator / (np.sqrt(self._dqsums[v1]) * np.sqrt(self._dqsums[v2]))

    def get_similarity(self, v1, v2):
        # print (self.get_similarity_dist(v1,v2))
        return (self.get_similarity_weight(v1,v2))
        # print (self.get_similarity_dist(v1,v2) * self.get_similarity_weight(v1,v2))
        # return (self.get_similarity_dist(v1, v2) * self.get_similarity_weight(v1, v2))

    def get_similarity_many(self, pairs):
        """Returns the Jaccard similarity between many pairs of vertices,
        assuming that all vertices are linked to themselves."""
        sim = self.get_similarity
        return [sim(*pair) for pair in pairs]


class EdgeCluster(object):
    """Class representing a group of edges (i.e. a group of vertices
    in the line graph)

    This class also keeps track of the original vertices the edges
    refer to."""

    __slots__ = ("vertices", "edges")

    def __init__(self, vertices, edges):
        self.vertices = set(vertices)
        self.edges = set(edges)

    def is_smaller_than(self, other):
        """Compares this group of edges with another one based on
        size."""
        return len(self.edges) < len(other.edges)

    def partition_density(self):
        """Returns the number of edges times the relative density
        of this group. This value is used in the calculation of
        the overall partition density, used to select the best
        threshold."""
        m, n = len(self.edges), len(self.vertices)
        if n <= 2:
            return 0.
        return m * (m - n + 1) / (n - 2) / (n - 1)

    def merge_from(self, other):
        """Merges another group of edges into this one, updating
        self.vertices and self.edges"""
        self.vertices |= other.vertices
        self.edges |= other.edges

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__,
                               self.vertices, self.edges)


class EdgeClustering(object):
    """Class representing an edge clustering of a graph as a whole.

    This class is essentially a list of `EdgeCluster` instances
    plus some additional bookkeeping to facilitate the easy lookup
    of the cluster of a given edge.
    """

    def __init__(self, edgelist):
        """Constructs an initial edge clustering of the given graph
        where each edge belongs to its own cluster.

        The graph is given by its edge list in the `edgelist`
        parameter."""
        self.clusters = [EdgeCluster(edge, (i,))
                         for i, edge in enumerate(edgelist)]
        self.membership = list(range(len(edgelist)))
        self.d = 0.0

    def lookup(self, edge):
        """Returns the cluster of a given edge"""
        return self.clusters[self.membership[edge]]

    def merge_edges(self, edge1, edge2):
        """Merges the clusters corresponding to the given edges."""
        cid1, cid2 = self.membership[edge1], self.membership[edge2]

        # Are they the same cluster?
        if cid1 == cid2:
            return

        cl1, cl2 = self.clusters[cid1], self.clusters[cid2]

        # We will always merge the smaller into the larger cluster
        if cl1.is_smaller_than(cl2):
            cl1, cl2 = cl2, cl1
            cid1, cid2 = cid2, cid1

        # Save the partition densities
        dc1, dc2 = cl1.partition_density(), cl2.partition_density()

        # Merge the smaller cluster into the larger one
        for edge in cl2.edges:
            self.membership[edge] = cid1
        cl1.merge_from(cl2)
        self.clusters[cid2] = cl1

        # Update D
        self.d += cl1.partition_density() - dc1 - dc2

    def partition_density(self):
        """Returns the overall partition density of the clustering."""
        return self.d * 2.0 / len(self.membership)


class HLC(object):
    """Hierarchical link clustering algorithm on a given graph.

    This class implements the algorithm outlined in Ahn et al: Link communities
    reveal multiscale complexity in networks, Nature, 2010. 10.1038/nature09182

    The implementation supports undirected and unweighted networks only at the
    moment, and it is assumed that the graph does not contain multiple or loop
    edges. This is not ensured within the class for sake of efficiency.

    The class provides the following attributes:

    - `graph` contains the graph being analysed
    - `min_size` contains the minimum size of the clusters one is interested
      in. It is advised to set this to at least 3 (which is the default value)
      to ensure that pseudo-clusters containing only two nodes do not turn up
      in the results.

    The algorithm may be run with or without a similarity threshold. When no
    similarity threshold is passed to the `run()` method, the algorithm will
    scan over the possible range of similarities and return a partition that
    corresponds to the similarity with the highest partition density. In this
    case, the similarity threshold and the partition density is recorded in
    the `last_threshold` and `last_partition_density` attributes. The former
    is also set properly when a single similarity threshold is used.
    """

    def __init__(self, graph=None, min_size=3):
        """Constructs an instance of the algorithm. The algorithm
        will be run on the given `graph` with the given minimum
        community size `min_size`."""
        self._graph = None
        self._edgelist = None
        self.last_threshold = None
        self.last_partition_density = None
        self.graph = graph
        self.min_size = int(min_size)

    @property
    def graph(self):
        """Returns the graph being clustered."""
        return self._graph

    @graph.setter
    def graph(self, graph):
        """Sets the graph being clustered."""
        self._graph = graph
        self._edgelist = graph.get_edgelist()

    def run(self, threshold=None):
        """Runs the hierarchical link clustering algorithm on the
        graph associated to this `HLC` instance, cutting the dendrogram
        at the given `threshold`. If the threshold is `None`, the
        optimal threshold will be selected using the partition density
        method described in Ahn et al, 2010. Returns a generator that
        will generate the clusters one by one.
        """
        if threshold is None:
            return self._run_iterative()
        else:
            return self._run_single(threshold)

    def get_edge_similarity_graph(self):
        """Calculates the edge similarity graph of the graph assigned
        to this `HLC` instance."""

        # Construct the line graph
        linegraph = self.graph.linegraph()

        # Select the appropriate similarity function
        if "weight" in self.graph.edge_attributes():
            similarity = TanimotoSimilarityCalculator(self.graph).get_similarity_many
            print(similarity)

        # For each edge in the line graph, compute a similarity score
        edgelist = self._edgelist  # prelookup
        sources, targets = array('l'), array('l')
        sources.extend(0 for _ in range(linegraph.ecount()))
        targets.extend(0 for _ in range(linegraph.ecount()))
        print('Computing simmilarity score for each edge.')
        with click.progressbar(linegraph.es) as bar:
            for edge in bar:
                (a, b), (c, d) = edgelist[edge.source], edgelist[edge.target]
                i = edge.index
                if a == c:
                    sources[i] = b
                    targets[i] = d
                elif a == d:
                    sources[i] = b
                    targets[i] = c
                elif b == c:
                    sources[i] = a
                    targets[i] = d
                else:  # b == d
                    sources[i] = a
                    targets[i] = c
        linegraph.es["score"] = similarity(zip(sources, targets))
        return linegraph

    def _run_single(self, threshold):
        """Runs the hierarchical link clustering algorithm on the
        graph associated to this `HLC` instance, cutting the dendrogram
        at the given threshold. Returns a generator that will generate the
        clusters one by one.

        :Parameters:
        - threshold: the level where the dendrogram will be cut
        """
        # Record the threshold in last_threshold
        self.last_threshold = threshold
        self.last_partition_density = None

        # Construct the edge similarity graph
        linegraph = self.get_edge_similarity_graph()

        # Remove unnecessary edges
        linegraph.es(score_le=threshold).delete()

        # Process the connected components of the linegraph and build the result
        clusters = linegraph.clusters()
        result = [set() for _ in range(len(clusters))]
        print("Building linegraph")
        with click.progressbar(zip(self._edgelist, clusters.membership)) as bar:
            for edge, cluster_index in bar:
                result[cluster_index].update(edge)
        return (list(cluster) for cluster in result
                if len(cluster) >= self.min_size)

    def _run_iterative(self):
        """Runs the hierarchical link clustering algorithm on the given graph,
        cutting the dendrogram at the place where the average weighted partition
        density is maximal. Returns a generator that will generate the clusters
        one by one.

        :Parameters:
        - graph: the graph being clustered
        - min_size: minimum size of clusters
        """

        # Construct the line graph
        linegraph = self.get_edge_similarity_graph()

        # Sort the scores
        sorted_edges = sorted(linegraph.es, key=itemgetter("score"),
                              reverse=True)

        # From now on, we only need the edge list of the original graph
        del linegraph

        # Set up initial configuration: every edge is a separate cluster
        clusters = EdgeClustering(self._edgelist)

        # Merge clusters, keep track of D, find maximal D
        max_d, best_threshold, best_membership = -1, None, None
        prev_score = None
        merge_edges = clusters.merge_edges  # prelookup
        print(
            'Merging clusters.'
        )
        with click.progressbar(sorted_edges) as bar:
            for edge in bar:
                score = edge["score"]

                if prev_score != score:
                    # Check whether the current D score is better than the best
                    # so far
                    if clusters.d >= max_d:
                        max_d, best_threshold = clusters.d, score
                        best_membership = list(clusters.membership)
                    prev_score = score

                # Merge the clusters
                merge_edges(edge.source, edge.target)

        del clusters
        max_d *= 2 / self.graph.ecount()

        # Record the best threshold and partition density
        self.last_threshold = best_threshold
        self.last_partition_density = max_d

        # Build the result
        result = defaultdict(set)
        for edge, cluster_index in zip(self._edgelist, best_membership):
            result[cluster_index].update(edge)
        return (list(cluster) for cluster in result.values()
                if len(cluster) >= self.min_size)


# g.to_undirected(combine_edges="sum")

algorithm = HLC(g, 10)
results = algorithm.run()

print("Threshold = %.6f" % algorithm.last_threshold)
print("D = %.6f" % algorithm.last_partition_density)

# outfile = open('results2.txt', "w")

res2 = list(results)

import geopandas as gp


def export_as_wkt(graph):
    g = ig.Graph.Read_GraphMLz(graph)


def edge_source(graph, edge):
    coordinate = (graph.vs[edge.source]['x'], graph.vs[edge.source]['y'])
    name = graph.vs[edge.source]['name']
    return [name, coordinate]


def edge_target(graph, edge):
    coordinate = (graph.vs[edge.target]['x'], graph.vs[edge.target]['y'])
    name = graph.vs[edge.target]['name']
    return [name, coordinate]


def edge_vertices_info(graph, edge):
    source = edge_source(graph, edge)
    target = edge_target(graph, edge)
    return [source, target]


from shapely.geometry import LineString, Point


def edges_to_geopandas(g):
    attributes = ['id', 'source', 'target', 'geometry']
    for attribute in g.es.attributes():
        attributes.append(attribute)
    # print(attributes)
    df = pd.DataFrame(columns=attributes)
    # with click.progressbar(g.es) as bar:
    for edge in g.es:
        info = edge_vertices_info(g, edge)
        info_list = [edge.index, info[0][0], info[1][0], LineString([Point(info[0][1]), Point(info[1][1])]),
                     edge['passenger_count'], edge['weight'],edge['trip_distance'], edge['length'], edge['community']]
        # print ()
        # dict = {}
        # for i, attribute in enumerate(attributes):
        #     # print (attribute, i)
        #     # print (info_list)
        #     dict[attribute] = [info_list[i]]
        # # print (pd.DataFrame.from_dict(dict))
        series = pd.Series(info_list, index=attributes, name=info_list[0])
        df = df.append(series)
        # print (df)
    gdf = gp.GeoDataFrame(df, geometry=df['geometry'])
    # gdf = gp.GeoDataFrame(df, geometry=LineString([Point(info[0][1]), Point(info[1][1])]))
    return gdf


# with click.progressbar(enumerate(res2)) as bar:
print(len(res2))
for i, community in enumerate(res2):
    print(i)
    sg = g.subgraph(community)
    sg.es['community'] = int(i)
    # sg.write_graphmlz('graphs/graph_'+str(i)+'.gmlz')
    df = edges_to_geopandas(sg)
    df.to_file('communities_taxi_spatial_length_3.gpkg', layer='community_' + str(i), driver='GPKG')
    # ig.plot(sg, autocurve=False, **visual_style).save('plots2/plot_'+str(i)+'.png')
