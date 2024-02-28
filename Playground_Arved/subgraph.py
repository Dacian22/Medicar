############################# Setup ########################################
# import libraries
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
import os

# reading data inside xml file to a variable under the name data
with open('Uniklinikum_Freiburg_map.osm', 'r', encoding='utf-8') as f:
    data = f.read()
 
# passing stored data inside beautifulsoup parser, storing the returned object 
bs_data = BeautifulSoup(data, "xml")

############################# Nodes ########################################

# store all nodes in a dictionary and give them a label
nds = {}
for node in bs_data.select('node'):
    # first case: store has a name
    if node.find('tag', k='name'):
        # map node id to a tuple of (lat, lon, name)
        nds[node['id']] = (node['lat'], node['lon'], node.find('tag', k='name').get('v'))
    else:
        nds[node['id']] = (node['lat'], node['lon'], None)

# create dictionary of all nodes that have a name and map id to name
named_nodes = {k: v[2] for k, v in nds.items() if v[2]}

############################# Ways ########################################

# store all ways in a dictionary and give them a label according to their highway type
ways = {}
for way in bs_data.select('way'):
    # filter out ways that don't have a highway tag
    if way.find('tag', k='highway'):
        # map way id to a tuple of (list of node ids, highway type)
        ways[way['id']] = ([nd['ref'] for nd in way.find_all('nd')],
                           way.find('tag', k='highway').get('v'))
    else:
        ways[way['id']] = ([nd['ref'] for nd in way.find_all('nd')],
                           None)
        
# list with all possible values of the key 'highway'
highway_types = list(set([way[1] for way in ways.values() if way[1] is not None]))
        
############################# Subgraph ########################################

# define the class subgraph
class Subgraph():
    def __init__(self, special_nodes, allowed_highway_types):
        self.special_nodes = special_nodes
        self.allowed_highway_types = allowed_highway_types
        
    # define function to create the graph
    def create_graph(self):
        # create a directed graph
        G = nx.DiGraph()
        # iterate over all ways
        for way_id, (nodes, highway_type) in ways.items():
            # check if the way is an allowed highway type
            if highway_type in self.allowed_highway_types:
                # add edges to the graph
                for u, v in zip(nodes[:-1], nodes[1:]):
                    G.add_edge(u, v)
        return G
    
    # define function to label edges of the graph
    def label_edges(self, G):
        # create a dictionary that maps every created edge to its highway type
        edge_labels_highways = {}
        # iterate over all edges that are highways
        for u, v in G.edges():
            # iterate over all ways
            for way_id, (nodes, highway_type) in ways.items():
                # check if the edge is in a way and add the highway type to the dictionary
                if u in nodes and v in nodes:
                    edge_labels_highways[(u, v)] = highway_type
        return edge_labels_highways

        # define function to get closest node that is part of a way to the special nodes
    def closest_node_to_special_nodes(self, G):
        closest_node = {}
        # iterate over all named nodes
        for node_id, name in [item for item in named_nodes.items() if item[1] in self.special_nodes]:
            # initialize minimum distance to infinity
            min_dist = float('inf')
            # iterate over all ways with an allowed highway type
            for way_id, way in ways.items():
                if way[1] in self.allowed_highway_types:
                    # iterate over all nodes in the way
                    for nd_id in way[0]:
                        # calculate distance
                        dist = (float(nds[nd_id][0]) - float(nds[node_id][0]))**2 + (float(nds[nd_id][1]) - float(nds[node_id][1]))**2
                        # if distance is smaller than minimum distance, update minimum distance and closest node
                        if dist < min_dist:
                            min_dist = dist
                            closest_node[node_id] = nd_id
        return closest_node
    
    # define function to plot the graph
    def plot_graph(self, G, edge_labels_highways):
        # create a figure
        plt.figure(figsize=(15,15))
        # draw the graph
        nx.draw(G, nds, with_labels=False, node_size=10, node_color='black')
        # draw the labels of the edges
        nx.draw_networkx_edge_labels(G, nds, edge_labels=edge_labels_highways, font_color='red')
        # show the plot
        plt.show()