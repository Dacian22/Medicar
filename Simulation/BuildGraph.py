############################# Setup ########################################
# import libraries
import os

import networkx as nx
from bs4 import BeautifulSoup
import pandas as pd

def get_graph_data():
    # reading data inside xml file to a variable under the name data
    try:
        with open('Uniklinikum_Freiburg_map.osm', 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        with open(os.path.join('Simulation', 'Uniklinikum_Freiburg_map.osm'), 'r', encoding='utf-8') as f:
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
            ways[way['id']] = ([nd['ref'] for nd in way.find_all('nd')], way.find('tag', k='highway').get('v'))
        else:
            ways[way['id']] = ([nd['ref'] for nd in way.find_all('nd')], None)

    # list with all possible values of the key 'highway'
    highway_types = list(set([way[1] for way in ways.values() if way[1] is not None]))

    return nds, named_nodes, ways, highway_types


def closest_node_to_special_nodes(special_nodes, allowed_highway_types, nds, named_nodes, ways):
    closest_node = {}
    # iterate over all named nodes
    for node_id, name in [item for item in named_nodes.items() if item[1] in special_nodes]:
        # initialize minimum distance to infinity
        min_dist = float('inf')
        # iterate over all ways with an allowed highway type
        for way_id, way in ways.items():
            if way[1] in allowed_highway_types:
                # iterate over all nodes in the way
                for nd_id in way[0]:
                    # calculate distance
                    dist = (float(nds[nd_id][0]) - float(nds[node_id][0])) ** 2 + (
                            float(nds[nd_id][1]) - float(nds[node_id][1])) ** 2
                    # if distance is smaller than minimum distance, update minimum distance and closest node
                    if dist < min_dist:
                        min_dist = dist
                        closest_node[node_id] = nd_id
    return closest_node


# define function that only keeps the largest connected component of the graph
def keep_largest_connected_component(G):
    # get all connected components
    connected_components = list(nx.connected_components(G))
    # get the largest connected component
    largest_connected_component = max(connected_components, key=len)
    # create a subgraph that only contains the largest connected component
    G = G.subgraph(largest_connected_component)
    return G


# define function to label edges of the graph
# def label_edges(G, ways):
#     # create a dictionary that maps every created edge to its highway type
#     edge_labels = {}
#     # iterate over all edges that are highways
#     for u, v in  :
#         # iterate over all edges
#         # for way_id, (nodes, highway_type) in ways.items():
#             # check if the edge is in a way and add the highway type to the dictionary
#             # if u in nodes and v in nodes:
#                 # edge_labels_highways[way_id] = {'highway_type': highway_type, 'u': u, 'v': v}
#         edge_labels[way_id] = {'u': u, 'v': v}
#     return edge_labels


def build_nx_graph(allowed_highway_types, special_nodes):
    nds, named_nodes, ways, highway_types = get_graph_data()

    # create an undirected graph
    G = nx.Graph()
    # iterate over all ways
    for way_id, (nodes, highway_type) in ways.items():
        # check if the way is an allowed highway type
        if highway_type in allowed_highway_types:
            # add edges to the graph
            G.add_edges_from(zip(nodes[:-1], nodes[1:]))
    # add edges from named nodes to the closest node on a way if the nodes are not the same
    for node_id, name in [item for item in named_nodes.items() if item[1] in special_nodes]:
        closest_node = closest_node_to_special_nodes(special_nodes, allowed_highway_types, nds, named_nodes, ways)
        if node_id != closest_node[node_id]:
            G.add_edge(node_id, closest_node[node_id])
    # add weights to the edges according to the distance between the nodes
    for u, v in G.edges():
        G[u][v]['weight'] = ((float(nds[u][0]) - float(nds[v][0])) ** 2 + (
                float(nds[u][1]) - float(nds[v][1])) ** 2) ** 0.5 * 10e6 # 10e6 is a scaling factor

    # keep only the largest connected component
    G = keep_largest_connected_component(G)

    # Create dataframe with all information about the nodes: osmid, lat, lon, map_name
    nodes_df = pd.DataFrame.from_dict(nds, orient='index', columns=['lat', 'lon', 'tag'])
    nodes_df = nodes_df.drop(columns=['tag'])
    nodes_df = nodes_df.astype("float")
    # Add name from named_nodes to df
    df_named_nodes = pd.DataFrame.from_dict(named_nodes, orient='index', columns=['name'])
    nodes_df = nodes_df.join(df_named_nodes, how='left')

    # Filter nodes to only include the onees that are part of the graph
    nodes_df = nodes_df[nodes_df.index.isin(G.nodes())]
    # Switch index to int
    nodes_df.index = nodes_df.index.astype("int64")

    # Create dataframe with all information about the edges: osmid, highway_type, u, v
    edges_df = pd.DataFrame(columns=['u', 'v'])
    edges_df['u'] = [u for u, v in G.edges()]
    edges_df['v'] = [v for u, v in G.edges()]
    edges_df[['u', 'v']] = edges_df[['u', 'v']].astype("int64")

    return G, edges_df, nodes_df

    # define function that sets all weights from a given list in the graph to infinity
def set_weights_to_inf(G, edges_to_be_set_to_inf):
    if edges_to_be_set_to_inf is None:
        return G
    else:
        # print("EDGES ARE SET TO INF:")
        for edge in G.edges():
            # for edge_to_be_set_to_inf in edges_to_be_set_to_inf:
            if str(edge[0]) == str(edges_to_be_set_to_inf[0]) and str(edge[1]) == str(edges_to_be_set_to_inf[1]):
                print(f"{edge} weight was set to inf")
                G[edge[0]][edge[1]]['weight'] = float('inf')
        return G