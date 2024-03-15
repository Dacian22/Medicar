import json
import os

import networkx as nx
import paho.mqtt.client as paho
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from paho import mqtt


class Routing():  # singleton class. Do not create more than one object of this class
    def __init__(self, graph, edge_labels_highways, named_nodes, nds):
        load_dotenv()
        self.graph = graph
        self.edge_labels_highways = edge_labels_highways
        self.named_nodes = named_nodes
        self.nds = nds
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
        print("Received new task: " + msg.payload.decode("utf-8"))
        self.handle_order(msg.payload.decode("utf-8"))

    def handle_order(self, order):
        # TODO currently assumes exactly one vehicle named Vehicle1
        order = json.loads(order)
        print(f"Received new order: {order}")
        # find the shortest path
        shortest_path = self.find_astar_path(self.graph, order["source"], order["target"])
        # translate the shortest path to MQTT messages
        message = self.translate_path_to_mqtt(shortest_path)
        # send the message to the MQTT broker
        self.client.publish(f"vehicles/{order['vehicle_id']}/route", json.dumps(message), qos=2)

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
        # test connection
        self.client.subscribe("order_manager/transportation/orders/#", qos=2)
        print("start")
        self.client.publish("hello", "simulation online", qos=2)
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
