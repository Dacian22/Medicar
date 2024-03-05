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

# define the class submap
class Submap():
    def __init__(self, special_nodes, allowed_highway_types):
        self.special_nodes = special_nodes
        self.allowed_highway_types = allowed_highway_types

    def __repr__(self):
        return f"Submap(Special nodes: {self.special_nodes}, Allowed highway types: {self.allowed_highway_types})"
    
    # define function to get closest node that is part of a way to the special nodes
    def closest_node_to_special_nodes(self):#(self, G):
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

    def create_graph(self):
        # create an undirected graph
        G = nx.Graph()
        # iterate over all ways
        for way_id, (nodes, highway_type) in ways.items():
            # check if the way is an allowed highway type
            if highway_type in self.allowed_highway_types:
                # add edges to the graph
                G.add_edges_from(zip(nodes[:-1], nodes[1:]))
        # add edges from named nodes to the closest node on a way if the nodes are not the same
        for node_id, name in [item for item in named_nodes.items() if item[1] in self.special_nodes]:
            closest_node = self.closest_node_to_special_nodes()
            if node_id != closest_node[node_id]:
                G.add_edge(node_id, closest_node[node_id])
        # add weights to the edges according to the distance between the nodes
        for u, v in G.edges():
            G[u][v]['weight'] = ((float(nds[u][0]) - float(nds[v][0]))**2 + (float(nds[u][1]) - float(nds[v][1]))**2)**0.5

        return G
    
    # define function that only keeps the largest connected component of the graph
    def keep_largest_connected_component(self, G):
        # get all connected components
        connected_components = list(nx.connected_components(G))
        # get the largest connected component
        largest_connected_component = max(connected_components, key=len)
        # create a subgraph that only contains the largest connected component
        G = G.subgraph(largest_connected_component)
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
    
    # define function to find the shortest path between two special nodes
    def find_astar_path(self, G, start_node, end_node):        
        # find node id of source and target node
        start_node_id = [key for key, value in named_nodes.items() if value == start_node][0]
        end_node_id = [key for key, value in named_nodes.items() if value == end_node][0]
        # use the a* algorithm to find the shortest path between the source and target node
        shortest_path = nx.astar_path(G, start_node_id, end_node_id, weight='weight')
        return shortest_path
    
    # define function that 'translates' the shortest path to MQTT messages
    def translate_path_to_mqtt(self, shortest_path):
        # create a list of messages
        messages = []
        # iterate over all edges in the shortest path
        edges_shortest_path = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
        for index, edge in enumerate(edges_shortest_path):
            messages.append({'edgeId': "edge_{}_{}".format(edge[0], edge[1]),
                             'sequenceId': index,
                             'startNodeId': edge[0],
                             'endNodeId': edge[1]})
        return messages
    
    # define function to plot the graph
    def plot_graph(self, G, edge_labels_highways, shortest_path):
        # plot the graph where all nodes are placed at their geographical position
        pos = {key: (float(value[0]), float(value[1])) for key, value in nds.items()}
        # correct position for the labels
        pos_labels = {key: (float(value[0]), float(value[1]) - 0.00008) for key, value in nds.items()}
        # create a figure
        plt.figure(figsize=(25, 15))
        # draw the graph
        nx.draw(G, pos=pos, with_labels=False, node_size=10, node_color='black', edge_color='black')
        # label all special nodes
        nx.draw_networkx_labels(G, pos=pos_labels, labels={key: value for key, value in named_nodes.items() if value in self.special_nodes},
                                font_size=10, font_color='green')
        # change the color of the special nodes
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[key for key, value in named_nodes.items() if value in self.special_nodes],
                               node_color='green', node_size=20)
        # draw the shortest path
        nx.draw_networkx_edges(G, pos=pos, edgelist=[(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)],
                               edge_color='red', width=3)
        # add title
        plt.title('University Hospital Freiburg', fontsize=20, fontweight='bold')
        # show the plot
        plt.show()