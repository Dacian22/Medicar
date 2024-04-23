############################# Setup ########################################
# import libraries
import os

import networkx as nx
from bs4 import BeautifulSoup
import pandas as pd
import osmnx as ox
import warnings

from dotenv import load_dotenv
load_dotenv()


def get_xml_graph_data():
    # reading data inside xml file to a variable under the name data
    with open(os.path.join(os.getenv("RESOURCES"), "Uniklinikum_Freiburg_map.osm"), 'r', encoding='utf-8') as f:
        data = f.read()

    # passing stored data inside beautifulsoup parser, storing the returned object
    bs_data = BeautifulSoup(data, "xml")

    ############################# Nodes ########################################

    # store all nodes in a dictionary and give them a label
    nds_xml = {}
    for node in bs_data.select('node'):
        # first case: store has a name
        if node.find('tag', k='name'):
            # map node id to a tuple of (lat, lon, name)
            nds_xml[str(node['id'])] = (node['lat'], node['lon'], node.find('tag', k='name').get('v'))
        else:
            nds_xml[str(node['id'])] = (node['lat'], node['lon'], None)

    # create dictionary of all nodes that have a name and map id to name
    named_nodes = {k: v[2] for k, v in nds_xml.items() if v[2]}

    ############################# Ways ########################################

    # store all ways in a dictionary and give them a label according to their surface type
    ways = {}

    for way in bs_data.select('way'):
        # filter out ways that don't have a surface tag
        if way.find('tag', k='surface'):
            # map way id to a tuple of (list of node ids, surface type)
            ways[way['id']] = ([nd['ref'] for nd in way.find_all('nd')],
                                way.find('tag', k='surface').get('v'))
        else:
            ways[way['id']] = ([nd['ref'] for nd in way.find_all('nd')],
                                None)

    # list with all possible values of the key 'surface'
    surface_types = list(set([way[1] for way in ways.values() if way[1] is not None]))

    return nds_xml, named_nodes, ways, surface_types

def build_xml_graph():
    nds_xml, named_nodes, ways, surface_types = get_xml_graph_data()

    # create graph
    G_xml = nx.Graph()
    # add all nodes from xml file
    for k, v in nds_xml.items():
        G_xml.add_node(k, y = v[0], x = v[1], street_count = None)

    # drop all nodes with degree 0 if the name is None
    for node in list(G_xml.nodes(data=True)):
        if node[1]['street_count'] == None:
            if G_xml.degree[node[0]] == 0:
                G_xml.remove_node(node[0])

    # add  all edges from xml file
    for way_id, (nodes, surface) in ways.items():
        for i in range(len(nodes)-1):
            G_xml.add_edge(nodes[i], nodes[i+1], surface = surface)

    return G_xml

############################# 2. OSMNX - Graph ########################################

def build_osmnx_graph():
    # downloading map as graph object
    G_osmnx = ox.graph_from_bbox(north = 48.00877, south = 48.00373,
                                 east = 7.84336, west = 7.83252,
                                 network_type = 'all')
    # ignore future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # convert it to a Single Undirected Graph
    G_osmnx = nx.Graph(G_osmnx)

    # convert all node ids to strings
    for node in list(G_osmnx.nodes()):
        G_osmnx = nx.relabel_nodes(G_osmnx, {node: str(node)})

    # delete self loop
    G_osmnx.remove_edges_from(nx.selfloop_edges(G_osmnx))

    return G_osmnx

############################# 3. Final - Graph ########################################

# define function that only keeps the largest connected component of the graph
def keep_largest_connected_component(G):
    # get all connected components
    connected_components = list(nx.connected_components(G))
    # get the largest connected component
    largest_connected_component = max(connected_components, key=len)
    # create a deep copy of the largest connected component subgraph
    largest_subgraph = G.subgraph(largest_connected_component).copy()

    return largest_subgraph

def closest_node_to_special_nodes(special_nodes, nds, named_nodes):
    closest_node = {}
    nodes_to_check = dict(filter(lambda x: x[0] not in named_nodes.keys(), nds.items()))
    # iterate over all special nodes
    for node_id, name in [item for item in named_nodes.items() if item[1] in special_nodes]:
        # initialize minimum distance to infinity
        min_dist = float('inf')
        # iterate over all nodes that are already in the graph
        for nd_id in nodes_to_check.keys():
            # calculate euclidean distance between special node and node in graph
            dist = (float(nds[nd_id]['x']) - float(nds[node_id]['x']))**2 + (float(nds[nd_id]['y']) - float(nds[node_id]['y']))**2
            # if distance is smaller than minimum distance, update minimum distance and closest node
            if dist < min_dist:
                min_dist = dist
                closest_node[node_id] = nd_id
    return closest_node

def build_nx_graph(allowed_highway_types, allowed_surface_types, special_nodes):
    # call created functions to get data
    nds_xml, named_nodes, ways, surface_types = get_xml_graph_data()
    G_osmnx = build_osmnx_graph()
    G_xml = build_xml_graph()

    # create final graph based on osmnx graph
    G = G_osmnx.copy(as_view=False)

    # manually delete the upper part of the graph
    G.remove_node('31452921')

    # manually delete the parallel edges at the lower part of the graph
    G.remove_node('3344424273')
    G.remove_node('3344424416')
    G.remove_node('11292850730')
    G.remove_node('585142663')
    G.remove_node('523915329')
    G.remove_node('798211818')
    G.remove_node('560014161')
    G.remove_node('3344424409')
    G.remove_node('302874634')
    G.remove_node('2926060005')
    G.remove_node('3344424403')
    G.remove_node('432654264')

    G.remove_edge('3137144878', '53422897')
    G.remove_edge('3137144885', '2926060006')

    # only keep largest connected component
    G = keep_largest_connected_component(G)

    # add special nodes from xml graph
    nds_final = [item for item in named_nodes.items() if item[1] in special_nodes]
    for node_id, name in nds_final:
        G.add_node(node_id, y=nds_xml[str(node_id)][0], x=nds_xml[str(node_id)][1], street_count=None)

    # add names of special nodes to final graph
    #nx.set_node_attributes(G, nds_final, name = 'name')

    # add surface type to edges from information about ways
    edge_surface = {}
    for way_id, (nodes, surface) in ways.items():
        for i in range(len(nodes)-1):
            for j in range(1,len(nodes)):
                if (nodes[i], nodes[j]) in G.edges:
                    edge_surface[(nodes[i], nodes[j])] = surface

    # add surface type to final graph
    nx.set_edge_attributes(G, edge_surface, 'surface')

    # drop all edges that have not an allowed surface type
    for edge, data in nx.get_edge_attributes(G, 'surface').items():
        if data not in allowed_surface_types:
            G.remove_edge(edge[0], edge[1])

    # drop all edges that have not an allowed highway type
    for edge, data in nx.get_edge_attributes(G, 'highway').items():
        if data not in allowed_highway_types:
            G.remove_edge(edge[0], edge[1])

    # add edges from named nodes to the closest node on an edge
    closest_node = closest_node_to_special_nodes(special_nodes, dict(G.nodes(data=True)), named_nodes)
    for node_id, nd_id in closest_node.items():
        G.add_edge(node_id, nd_id)

    # delete self loops in the case they coincide
    G.remove_edges_from(nx.selfloop_edges(G))

    # drop all nodes that have no name and only degree 1
    # repeat until no node is dropped anymore
    while True:
        nodes_to_drop = [node for node in G.nodes() if G.degree[node] == 1 and node not in named_nodes.keys()]
        if len(nodes_to_drop) == 0:
            break
        for node in nodes_to_drop:
            G.remove_node(node)

    # Create dataframe with all information about the nodes: osmid, lat, lon, map_name
    nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    nodes_df = nodes_df.drop(columns=['street_count', 'highway'])
    nodes_df = nodes_df.rename(columns={'x': 'lon', 'y': 'lat'})
    nodes_df = nodes_df.astype("float")
    # Add name from named_nodes to df
    df_named_nodes = pd.DataFrame.from_dict(named_nodes, orient='index', columns=['name'])
    nodes_df = nodes_df.join(df_named_nodes, how='left')
    # Switch index to int
    nodes_df.index = nodes_df.index.astype("int64")

    # Create dataframe with all information about the edges: osmid, highway_type, u, v
    edges_df = pd.DataFrame(columns=['u', 'v'])
    edges_df['u'] = [u for u, v in G.edges()]
    edges_df['v'] = [v for u, v in G.edges()]
    edges_df[['u', 'v']] = edges_df[['u', 'v']].astype("int64")

    return G, edges_df, nodes_df

############################# Rerouting ########################################

    # define function that sets all weights from a given list in the graph to infinity
def set_weights_to_inf(G, edges_to_be_set_to_inf):
    if edges_to_be_set_to_inf is None:
        print("FAIL: edges not removed (are none)")
        return G
    else:
        for edge in G.edges():
            if (str(edge[0]) == str(edges_to_be_set_to_inf[0]) and str(edge[1]) == str(edges_to_be_set_to_inf[1])) or (str(edge[0]) == str(edges_to_be_set_to_inf[1]) and str(edge[1]) == str(edges_to_be_set_to_inf[0])) :
                print(f"{edge} weight was set to inf")
                G[edge[0]][edge[1]]['length'] = float('inf')
        return G