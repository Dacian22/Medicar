import json
import os
import threading
import time

import networkx as nx
import paho.mqtt.client as paho
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from paho import mqtt
import osmnx as ox
import plotly.graph_objects as go
import folium

class Routing():  # singleton class. Do not create more than one object of this class
    def __init__(self, graph, edge_labels_highways, named_nodes, nds):
        load_dotenv()
        self.graph = graph
        self.edge_labels_highways = edge_labels_highways
        self.named_nodes = named_nodes
        self.nds = nds
        self.vehicles = {}
        self.connect_to_mqtt()

    def __repr__(self):
        return f"Graph: {self.graph}"

    # define function to find the shortest path between two special nodes
    def find_astar_path(self, G, start_node, end_node):
        # find node id of source and target node
        start_node_id = [key for key, value in self.named_nodes.items() if value == start_node][0]
        end_node_id = [key for key, value in self.named_nodes.items() if value == end_node][0]
        # use the a* algorithm to find the shortest path between the source and target node
        shortest_path = nx.astar_path(G, start_node_id, end_node_id, weight='weight')
        return shortest_path
    
    def find_dijkstra_path(self, G, start_node, end_node):
        # find node id of source and target node
        start_node_id = [key for key, value in self.named_nodes.items() if value == start_node][0]
        end_node_id = [key for key, value in self.named_nodes.items() if value == end_node][0]
        
        # use Dijkstra's algorithm to find the shortest path between the source and target node
        shortest_path_length, shortest_path = nx.single_source_dijkstra(G, start_node_id, target=end_node_id, weight='weight')
        return shortest_path
    
    def find_bellman_ford_path(self, G, start_node, end_node):
        # find node id of source and target node
        start_node_id = [key for key, value in self.named_nodes.items() if value == start_node][0]
        end_node_id = [key for key, value in self.named_nodes.items() if value == end_node][0]
        
        # use Bellman-Ford algorithm to find the shortest path between the source and target node
        shortest_path = nx.bellman_ford_path(G, start_node_id, end_node_id, weight='weight')
        return shortest_path
    
    def evaluate_shortest_path_weight(G, shortest_path):
        total_weight = 0
        for i in range(len(shortest_path)-1):
            source = shortest_path[i]
            target = shortest_path[i+1]
            edge_weight = G[source][target]['weight']
            total_weight += edge_weight
        return total_weight
    

    # define function that 'translates' the shortest path to MQTT messages
    def translate_path_to_mqtt(self, shortest_path):
        # create a list of messages
        edges = []
        # iterate over all edges in the shortest path
        edges_shortest_path = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
        for index, edge in enumerate(edges_shortest_path):
            edges.append(
                {'edgeId': "edge_{}_{}".format(edge[0], edge[1]), 'sequenceId': index, 'startNodeId': edge[0],
                 'endNodeId': edge[1], 'startCoordinate': self.nds[edge[0]][:2], 'endCoordinate': self.nds[edge[1]][:2]})
        message = {'edges': edges}
        print(message)
        print(json.dumps(message))
        return message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("CONNACK received with code %s." % rc)

    def on_publish(self, client, userdata, mid, properties=None):
        print("mid: " + str(mid))

    def on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

        if msg.topic.startswith("vehicles/") and msg.topic.endswith("/status"):
            vehicle_id = msg.topic.split("/")[1]
            vehicle_status = json.loads(msg.payload.decode())
            self.vehicles[vehicle_id] = vehicle_status
            print(f"Received status of vehicle {vehicle_id}: {vehicle_status}")
        else: # received order
            print("Received new task: " + msg.payload.decode("utf-8"))
            self.handle_order(msg.payload.decode("utf-8"))

    def get_distance(self, start_node_id, end_node_id):
        return nx.astar_path_length(self.graph, start_node_id, end_node_id, weight='weight')

    def get_distance_from_vehicle_to_order(self, vehicle_id, order):
        vehicle = self.vehicles[vehicle_id]
        start_node_id = [key for key, value in self.named_nodes.items() if value == order["source"]][0]
        return self.get_distance(vehicle["targetNode"], start_node_id)

    def get_distances_from_vehicles_to_order(self, order, use_only_idle_vehicles):
        distances = {}
        for vehicle_id, vehicle in self.vehicles.items():
            if not use_only_idle_vehicles or vehicle["status"] == "idle":
                distances[vehicle_id] = self.get_distance_from_vehicle_to_order(vehicle_id, order)
        print(distances)
        return distances

    def get_vehicle_id_for_order(self, order):
        # Heuristic: vehicle with the shortest distance to source node.
        # Only if all vehicles busy, consider the (busy) vehicle with the shortest path to the target node.
        if order["vehicle_id"] == "None":
            distances = self.get_distances_from_vehicles_to_order(order, True)
            if distances == {}:
                distances = self.get_distances_from_vehicles_to_order(order, False)
            return min(distances, key=distances.get)
        else:
            return order["vehicle_id"]

    def handle_order(self, order):
        order = json.loads(order)
        print(f"Received new order: {order}")
        # find the shortest path

        shortest_path_astar = self.find_astar_path(self.graph, order["source"], order["target"])
        # shortest_path_dijkstra = self.find_dijkstra_path(self.graph, order["source"], order["target"])
        # shortest_path_bellman_ford = self.find_bellman_ford_path(self.graph, order["source"], order["target"])
        #
        # evaluation_metric_astar = self.evaluate_shortest_path_weight(self.graph, shortest_path_astar)
        # evaluation_metric_dijkstra = self.evaluate_shortest_path_weight(self.graph, shortest_path_dijkstra)
        # evaluation_metric_bellman_ford = self.evaluate_shortest_path_weight(self.graph, shortest_path_bellman_ford)

        # shortest_path = None
        # if evaluation_metric_astar >= evaluation_metric_dijkstra and evaluation_metric_astar >= evaluation_metric_bellman_ford:
        #     shortest_path = shortest_path_astar
        # elif evaluation_metric_dijkstra >= evaluation_metric_astar and evaluation_metric_dijkstra >= evaluation_metric_bellman_ford:
        #     shortest_path = shortest_path_dijkstra
        # else:
        #     shortest_path = shortest_path_bellman_ford

        # translate the shortest path to MQTT messages
        message = self.translate_path_to_mqtt(shortest_path_astar)
        # send the message to the MQTT broker
        threading.Thread(target=self.send_route_to_vehicle_async, args=(self.get_vehicle_id_for_order(order), message)).start()

    def send_route_to_vehicle_async(self, vehicle_id, route):  # please call this method async
        while self.vehicles[vehicle_id]["status"] != "idle":
            time.sleep(5)
        self.client.publish(f"vehicles/{vehicle_id}/route", json.dumps(route), qos=2)

    def connect_to_mqtt(self):
        # Connect to MQTT
        self.client = paho.Client(client_id="singleton_routing", userdata=None, protocol=paho.MQTTv5)
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

        # enable TLS for secure connection
        self.client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
        # set username and password
        self.client.username_pw_set(os.getenv("HYVE_MQTT_USR"), os.getenv("HYVE_MQTT_PWD"))
        # connect to HiveMQ Cloud on port 8883 (default for MQTT)
        self.client.connect(os.getenv("HYVE_MQTT_URL"), 8883)
        # subscribe to orders
        self.client.subscribe("order_manager/transportation/orders/#", qos=2)
        # subscribe to vehicle status
        self.client.subscribe("vehicles/+/status", qos=2)
        print("start")
        self.client.publish("hello", "simulation online", qos=2)
        threading.Thread(target=self.folium_plot).start()
        # self.folium_plot()
        print("start mqtt loop")
        self.client.loop_forever()

    # define function to plot the graph
    def plot_graph(self, G, edge_labels_highways, shortest_path):
        # plot the graph where all nodes are placed at their geographical position
        # TODO this does not work in this setting
        # pos = {key: (float(value[0]), float(value[1])) for key, value in self.graph.nodes(data=True)}
        # # correct position for the labels
        # pos_labels = {key: (float(value[0]), float(value[1]) - 0.00008) for key, value in nds.items()}
        # create a figure
        plt.figure(figsize=(25, 15))
        # draw the graph
        # nx.draw(G, pos=pos, with_labels=False, node_size=10, node_color='black', edge_color='black')
        nx.draw(G, with_labels=False, node_size=10, node_color='black', edge_color='black')

        # label all special nodes
        # TODO this does not work in this setting
        # nx.draw_networkx_labels(G, pos=pos_labels, labels={key: value for key, value in named_nodes.items() if
        #                                                    value in self.special_nodes},
        #                         font_size=10, font_color='green')
        # change the color of the special nodes
        # TODO this does not work in this setting
        # nx.draw_networkx_nodes(G, pos=pos,
        #                        nodelist=[key for key, value in named_nodes.items() if value in self.special_nodes],
        #                        node_color='green', node_size=20)
        # draw the shortest path
        # nx.draw_networkx_edges(G, pos=pos, edgelist=[(shortest_path[i], shortest_path[i + 1]) for i in
        #                                              range(len(shortest_path) - 1)],
        #                        edge_color='red', width=3)
        nx.draw_networkx_edges(G, edgelist=[(shortest_path[i], shortest_path[i + 1]) for i in
                                            range(len(shortest_path) - 1)], edge_color='red', width=3)
        # add title
        plt.title('University Hospital Freiburg', fontsize=20, fontweight='bold')
        # show the plot
        plt.show()

    def folium_plot(self):
        while True:
            print("plotting map...")
            # Downloading the map as a graph object
            G = ox.graph_from_bbox(north = 48.0081000, south = 48.0048000,
                        east = 7.8391000, west = 7.8357000, network_type = 'all')
            # filter on nodes and edges that exist in self.graph
            G = G.subgraph([int(i) for i in self.graph.nodes()])
            print(G)
            # Create a map
            m = folium.Map(location=[48.006, 7.837], zoom_start=10,
                    zoom_control=False, scrollWheelZoom=False)
            # Defining the map boundaries
            m.fit_bounds([[48.0048000, 7.8357000], [48.0081000, 7.8391000]])
            # include the car icon in the map as a marker
            for vehicle in self.vehicles.keys():
                print("vehicle: " + vehicle)
                # Create marker for vehicle using the car icon at the current vehicle position
                folium.Marker(location=[float(self.vehicles[vehicle]["position"][0]), float(self.vehicles[vehicle]["position"][1])],
                              icon=folium.features.CustomIcon('porsche-icon.svg', icon_size=(30, 30)), popup=f"Vehicle: {vehicle}").add_to(m)
            # plot the graph on the map
            map = ox.plot_graph_folium(G, graph_map=m, color="grey")
            # save the map
            map.save('map.html')
            print("map plottet")
            time.sleep(1)