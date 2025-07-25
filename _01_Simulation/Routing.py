import copy
import json
import os
import re
import sys
import threading
import time
import warnings

import networkx as nx
import numpy as np
import paho.mqtt.client as mqtt
import paho.mqtt.client as paho
import plotly.graph_objects as go
from dash import Dash, dash_table
from dash import html, dcc, Output, Input, State, no_update
from dotenv import load_dotenv

import BuildGraph
import LLM_Dynamic_Weights
import LLM_Edge_Usability
import LLM_MetaModel
import TestEvaluationCsv

lock = threading.Lock()
warnings.filterwarnings("ignore")

import random


class Routing():  # singleton class. Do not create more than one object of this class

    def __init__(self, graph, edge_df, nodes_df):
        """
        Initializes the routing class with the given graph, edge, and node data, and sets up essential parameters.

        This constructor initializes the following attributes:
        - `graph`: The graph structure representing the network (e.g., buildings and routes between them).
        - `edge_df`: A DataFrame containing information about the edges in the graph.
        - `nodes_df`: A DataFrame containing information about the nodes in the graph.
        - `vehicles`: A dictionary to store the state and information about the vehicles in the network.
        - `orders`: A dictionary to store active orders and their statuses.
        - `events`: A dictionary to store any events affecting the network (e.g., accidents, roadblocks).
        - `order_queue`: A queue to manage and prioritize orders.
        - `generate_incidents`: A flag indicating whether incident generation is turned "on" or "off".
        - `current_model`: The language model currently in use (default is "gpt-few-shot").
        
        Additionally, it loads environment variables, and connects to the MQTT broker for real-time communication with vehicles and other systems.

        Args:
            graph: The graph structure representing nodes (buildings) and edges (routes between them).
            edge_df: A Pandas DataFrame containing information about the edges of the graph.
            nodes_df: A Pandas DataFrame containing information about the nodes of the graph.
        """

        load_dotenv(override=True)
        self.graph = graph
        self.edge_df = edge_df
        self.nodes_df = nodes_df
        self.vehicles = {}
        self.orders = {}
        self.events = {}
        self.generate_incidents = "off"
        self.current_model = "gpt-few-shot"
        self.dash_app_thread = None
        self.process_orders_thread = None
        self.connect_to_mqtt()

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """
        Callback function triggered when the MQTT client connects to the broker.

        Args:
            client: The MQTT client instance.
            userdata: Custom user data (not used in this function).
            flags: Response flags sent by the broker.
            rc: The connection result code indicating the status of the connection.
            properties: Optional properties for the MQTT message (default is None).
        """

        print("CONNACK received with code %s." % rc)
        # subscribe to orders
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "order_manager/transportation/orders/#", qos=2)
        # subscribe to vehicle status
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/+/status", qos=2)
        # subscribe to incidents
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/+/incident", qos=2)
        # subscribe to order finish
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/+/order_finish", qos=2)
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "hello", "simulation online", qos=2)
        if self.dash_app_thread is None:
            self.dash_app_thread = threading.Thread(target=self.get_map)
            self.dash_app_thread.start()
        if self.process_orders_thread is None:
            self.process_orders_thread = threading.Thread(target=self.process_orders)
            self.process_orders_thread.start()

    def on_publish(self, client, userdata, mid, reason_code, properties=None):
        # print("mid: " + str(mid))
        pass

    def on_subscribe(self, client, userdata, mid, reason_code_list, properties, granted_qos=None):
        # print("Subscribed: " + str(mid) + " " + str(granted_qos))
        pass

    def on_message(self, client, userdata, msg):
        """
        Callback function triggered when a message is received on an MQTT topic.

        Args:
            client: The MQTT client instance that received the message.
            userdata: Custom user data (not used in this function).
            msg: The MQTT message object containing the topic and payload.
        """

        if msg.topic.startswith(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/") and msg.topic.endswith("/status"):
            vehicle_id = msg.topic.split("/")[2]
            vehicle_status = json.loads(msg.payload.decode())
            self.vehicles[vehicle_id] = vehicle_status

        elif msg.topic.startswith(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/") and msg.topic.endswith(
                "/incident"):
            # Add incident to self.incidents
            incident = json.loads(msg.payload.decode("utf-8"))
            print(f"Received incident: {incident}")
            prompt = incident["edgeId"] + " " + incident["prompt"]
            threading.Thread(target=self.invoke_selected_model,
                             args=[prompt, self.current_model, False, incident["vehicleId"]]).start()

        elif msg.topic.startswith(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/") and msg.topic.endswith(
                "/order_finish"):
            print("Received order finished message: " + msg.payload.decode("utf-8"))
            order_id = json.loads(msg.payload.decode("utf-8"))["orderId"]
            self.orders[order_id]["status"] = "completed"

        else:  # received order
            # Add order to self.orders
            order = json.loads(msg.payload.decode("utf-8"))
            order["status"] = "waiting..."
            self.orders[order["order_id"]] = order

    def process_orders(self):
        """
        Continuously processes orders from the order queue if vehicles are available and idle.
        """
        while True:
            # If vehicles are available (idle)
            if len(self.vehicles) > 0 and any(
                    vehicle["status"] == "idle" for vehicle in self.vehicles.values()) and len(self.orders) > 0:
                for _, order in self.orders.items():
                    if order["status"] == "waiting...":
                        vehicle_id = self.get_vehicle_id_for_order(order)
                        if vehicle_id is None:
                            print("No vehicle available")
                        else:
                            self.vehicles[vehicle_id]["status"] = "busy"
                            order["vehicle_id"] = vehicle_id
                            order["status"] = "assigned"
                            self.handle_order(order)
                        break
            time.sleep(2)

    def handle_order(self, order):
        """
        Assigns a vehicle to an order, calculates the route, and sends the route to the vehicle.

        Args:
            order (dict): A dictionary containing details of the order, including 'source' and 'target' nodes.
            order_id (str): The unique identifier for the order.
        """

        path_to_target, _ = self.find_astar_path(self.graph,
                                                 self.get_node_id_from_name(order["source"]),
                                                 self.get_node_id_from_name(order["target"]))
        if path_to_target is None:
            threading.Thread(target=self.cancel_route_to_vehicle_async,
                             args=(order)).start()
            return

        # Check if vehicle is already on source node
        if self.vehicles[order["vehicle_id"]]["targetNode"] == self.get_node_id_from_name(order["source"]):
            path = path_to_target
        else:
            path_to_source, _ = self.find_astar_path(self.graph,
                                                     self.vehicles[order["vehicle_id"]]["targetNode"],
                                                     self.get_node_id_from_name(order["source"]))
            if path_to_source is None:
                threading.Thread(target=self.cancel_route_to_vehicle_async,
                                 args=(order)).start()
                return
            path = path_to_source + path_to_target

        message = self.translate_path_to_mqtt(path, order["order_id"])

        # send the message to the MQTT broker and set vehicle status to busy
        threading.Thread(target=self.send_route_to_vehicle_async,
                         args=(order["vehicle_id"], message)).start()
        order["status"] = "in progress..."

    def send_route_to_vehicle_async(self, vehicle_id, route):
        """
        Asynchronously sends a route to the specified vehicle via MQTT.

        Args:
            vehicle_id (str): The unique identifier for the vehicle.
            route (dict): The route information to be sent to the vehicle.
        """

        # if self.vehicles[vehicle_id]["status"] != "idle":
        #     print(f"ERROR: Vehicle {vehicle_id} is not idle")
        #     return
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + f"vehicles/{vehicle_id}/route", json.dumps(route),
                            qos=2)

    def cancel_route_to_vehicle_async(self, order):
        """
        Asynchronously cancels the route for the specified vehicle and updates the order status.

        Args:
            vehicle_id (str): The unique identifier for the vehicle.
            order (dict): The order information, including its ID and status.
        """
        # Set status order to unreachable
        order["status"] = "unreachable"
        print(f"\n## WARNING! Could not find a path for order {order['order_id']}\n")

        # Send empty route to vehicle
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + f"vehicles/{order['vehicle_id']}/cancel_route",
                            qos=2)

    def update_route_to_vehicle_async(self, vehicle_id, route, order):
        """
        Asynchronously updates the route for the specified vehicle and updates the order status.

        Args:
            vehicle_id (str): The unique identifier for the vehicle.
            route (dict): The new route to be sent to the vehicle.
            order (dict): The order information, including its ID and status.
        """

        # Send the new route to the vehicle
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + f"vehicles/{vehicle_id}/update_route",
                            json.dumps(route),
                            qos=2)

        # Update the order status
        order["status"] = "rerouted..."
        return

    def connect_to_mqtt(self):
        """Establishes a connection to the MQTT broker with TLS encryption and authentication."""

        # Connect to MQTT
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=paho.MQTTv5)
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

        # enable TLS for secure connection
        self.client.tls_set()  # tls_version=mqtt.client.ssl.PROTOCOL_TLS
        self.client.tls_insecure_set(True)

        # set username and password
        self.client.username_pw_set(os.getenv("HYVE_MQTT_USR"), os.getenv("HYVE_MQTT_PWD"))
        # connect to HiveMQ Cloud on port 8883 (default for MQTT)
        self.client.connect(os.getenv("HYVE_MQTT_URL"), 8883)
        self.client.loop_forever()

    def get_distance(self, start_node_id, end_node_id):
        """
        Calculates the shortest path length between two nodes in the graph using the A* algorithm.

        Args:
            start_node_id (str): The identifier of the starting node.
            end_node_id (str): The identifier of the destination node.

        Returns:
            float: The length of the shortest path between the start and end nodes, considering edge weights.
        """

        return nx.astar_path_length(self.graph, str(start_node_id), str(end_node_id), weight='weight')

    def get_distance_from_vehicle_to_order(self, vehicle_id, order):
        """
        Calculates the shortest path length from a vehicle's current location to the source node of an order.

        Args:
            vehicle_id (str): The identifier of the vehicle.
            order (dict): A dictionary containing order details, including the source node's name.

        Returns:
            float: The distance between the vehicle's current location and the source node of the order.
        """

        vehicle = self.vehicles[vehicle_id]
        start_node_id = self.get_node_id_from_name(order["source"])
        return self.get_distance(vehicle["targetNode"], start_node_id)

    def find_astar_path(self, G, start_node_id, end_node_id):
        """
        Finds the shortest path between two nodes in a graph using the A* algorithm.

        Args:
            G (networkx.Graph): The graph on which to perform the A* search.
            start_node_id (str): The ID of the starting node.
            end_node_id (str): The ID of the ending node.

        Returns:
            tuple: A tuple containing:
                - shortest_path (list): The list of nodes representing the shortest path from the start node to the end node.
                - astar_time (float): The time in seconds taken to compute the path.
        """
        start_time = time.perf_counter()
        # use the a* algorithm to find the shortest path between the source and target node
        shortest_path = nx.astar_path(G, str(start_node_id), str(end_node_id), weight='length')
        end_time = time.perf_counter()
        astar_time = end_time - start_time
        return shortest_path, astar_time

    def find_dijkstra_path(self, G, start_node_id, end_node_id):
        """
        Finds the shortest path between two nodes in a graph using the Dijkstra algorithm.

        Args:
            G (networkx.Graph): The graph on which to perform the Dijkstra search.
            start_node_id (str): The ID of the starting node.
            end_node_id (str): The ID of the ending node.

        Returns:
            tuple: A tuple containing:
                - shortest_path (list): The list of nodes representing the shortest path from the start node to the end node.
                - dijkstra_time (float): The time in seconds taken to compute the path.
        """

        start_time = time.perf_counter()
        # use Dijkstra's algorithm to find the shortest path between the source and target node
        shortest_path_length, shortest_path = nx.single_source_dijkstra(G, str(start_node_id), target=str(end_node_id),
                                                                        weight='length')
        end_time = time.perf_counter()
        dijkstra_time = end_time - start_time
        return shortest_path, dijkstra_time

    def find_bellman_ford_path(self, G, start_node_id, end_node_id):
        """
        Finds the shortest path between two nodes in a graph using the Bellman-Ford algorithm.

        Args:
            G (networkx.Graph): The graph on which to perform the Bellman-Ford search.
            start_node_id (str): The ID of the starting node.
            end_node_id (str): The ID of the ending node.

        Returns:
            tuple: A tuple containing:
                - shortest_path (list): The list of nodes representing the shortest path from the start node to the end node.
                - bellman_ford_time (float): The time in seconds taken to compute the path.
        """

        start_time = time.perf_counter()
        # use Bellman-Ford algorithm to find the shortest path between the source and target node
        shortest_path = nx.bellman_ford_path(G, str(start_node_id), str(end_node_id), weight='length')
        end_time = time.perf_counter()
        bellman_ford_time = end_time - start_time
        return shortest_path, bellman_ford_time

    def evaluate_shortest_path_weight(self, G, shortest_path):
        """
        Evaluates the total weight of a given shortest path in the graph.

        Args:
            G (networkx.Graph): The graph containing the edges with weights.
            shortest_path (list): A list of nodes representing the shortest path in the graph.

        Returns:
            float: The total weight of the shortest path. This is the sum of the weights of all edges along the path.
        """
        total_weight = 0
        for i in range(len(shortest_path) - 1):
            source = shortest_path[i]
            target = shortest_path[i + 1]
            if source != target:
                edge_weight = G[source][target]['length']
            else:
                edge_weight = 0
            total_weight += edge_weight
        return total_weight

    def translate_path_to_mqtt(self, shortest_path, order_id):
        """
        Translates a shortest path into a message format suitable for MQTT communication.

        Args:
            shortest_path (list): A list of nodes representing the shortest path in the graph.
            order_id (str): The ID of the order associated with the path.

        Returns:
            dict: A dictionary representing the MQTT message. Contains the path details as a list of edges and the order ID.
        """

        # Create a list of messages
        edges = []
        # Iterate over all edges in the shortest path
        edges_shortest_path = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
        for index, edge in enumerate(edges_shortest_path):
            edges.append(
                {'edgeId': "edge_{}_{}".format(edge[0], edge[1]), 'sequenceId': index, 'startNodeId': edge[0],
                 'endNodeId': edge[1], 'startCoordinate': tuple(self.nodes_df.loc[int(edge[0])].loc[["lat", "lon"]]),
                 'endCoordinate': tuple(self.nodes_df.loc[int(edge[1])].loc[["lat", "lon"]])})
        message = {'edges': edges, 'orderId': order_id}
        return message

    def get_node_id_from_name(self, name):
        """Retrieves the node ID from the node name."""

        return self.nodes_df[self.nodes_df["name"] == name].index[0]

    def get_distances_from_vehicles_to_order(self, order, use_only_idle_vehicles):
        """
        Calculates distances from vehicles to a given order.

        Args:
            order (dict): A dictionary containing order details. It should include the "source" key, 
                        which represents the starting point for the distance calculation.
            use_only_idle_vehicles (bool): If True, only calculates distances for vehicles with an "idle" status. 
                                            If False, calculates distances for all vehicles regardless of their status.

        Returns:
            dict: A dictionary where keys are vehicle IDs and values are the distances from the vehicle to the order's source location.
        """
        distances = {}
        for vehicle_id, vehicle in self.vehicles.items():
            if not use_only_idle_vehicles or vehicle["status"] == "idle":
                distances[vehicle_id] = self.get_distance_from_vehicle_to_order(vehicle_id, order)
        return distances

    def get_vehicle_id_for_order(self, order):
        """
        Determines the best vehicle to assign to an order based on distance to the order's source or target location.

        Args:
            order (dict): A dictionary containing order details. Should include:
                        - "source": The starting point of the order.
                        - "vehicle_id": The ID of the vehicle assigned to the order, or "None" if no vehicle is assigned.

        Returns:
            str: The ID of the vehicle best suited to handle the order based on the distance heuristic.
        """
        # Heuristic: vehicle with the shortest distance to source node.
        # Only if all vehicles busy, consider the (busy) vehicle with the shortest path to the target node.
        if order["vehicle_id"] == "None":
            distances = self.get_distances_from_vehicles_to_order(order, True)
            if distances == {}:
                return None
            return min(distances, key=distances.get)
        else:
            return order["vehicle_id"]

    def parse_edge(self, llm_output, edge_id=None):
        """
        Parses an edge identifier from a given LLM output.

        Args:
            llm_output (str): The output from the LLM containing edge information. 
            edge_id (str, optional): An edge identifier in the format "edge_X_Y" to be parsed if provided.

        Returns:
            tuple or list: A tuple containing the edge ID as (source_id, target_id) or a list of such tuples if multiple edges are found. 
                        Returns "ERROR" if no valid edge information could be parsed.
        """

        pattern1 = r"\([`']?(\d+)[`']?,.?[`']?(\d+)[`']?\)"
        pattern2 = r"(?:edge_)?([0-9]+)_([0-9]+)"
        if edge_id is None:
            try:
                matches = re.findall(pattern2, llm_output)
                if len(matches) == 1:
                    edge_id = int(re.findall(pattern2, llm_output)[0][0]), int(re.findall(pattern2, llm_output)[0][1])
                elif len(matches) > 1:
                    edge_id = []
                    for match in matches:
                        edge_id.append((int(match[0]), int(match[1])))
            except IndexError:
                try:
                    edge_id = int(re.findall(pattern1, llm_output)[0][0]), int(re.findall(pattern1, llm_output)[0][1])
                except IndexError:
                    print("ERROR: Could not find edge in LLM output")
                    return "ERROR"
        else:
            edge_id = int(re.findall(pattern2, edge_id)[0][0]), int(re.findall(pattern2, edge_id)[0][1])
        return edge_id

    def apply_llm_output(self, llm_output, prompt, edgeId, human=True, vehicleId=None, dynamic=False):
        """
        Applies the output from an LLM to update the graph and handle incidents.

        This method processes the output from a large language model (LLM) to determine if an edge in the graph 
        should be removed or if the graph should remain unchanged. It updates the graph, handles incidents, 
        and reroutes vehicles if necessary.

        Args:
            llm_output (str): The output received from the LLM, which indicates whether an edge should be removed.
            prompt (str): The prompt sent to the LLM, used for context.
            edgeId (tuple): The ID of the edge (start_node_id, end_node_id) to be potentially updated.
            human (bool, optional): Indicates if the output is from a human or a vehicle. Defaults to True (human).
            vehicleId (str, optional): The ID of the vehicle, used if the output is from a vehicle. Defaults to None.
            dynamic (bool, optional): Indicates if the output is dynamic. If True, uses dynamic parsing. Defaults to False.

        Returns:
            str: A status message indicating the result of applying the LLM output:
                - "SUCCESS" if the edge was successfully removed and vehicles were rerouted.
                - "ERROR" if the edge could not be removed.
                - "NO_CHANGE" if no changes were made.
        """

        self.events[edgeId] = {
            "status": "N/A",
            "value": 0,
            # timestamp in HH:MM:SS
            "timestamp": time.strftime('%H:%M:%S', time.localtime()),
            "origin": "Human" if human else "Vehicle " + str(vehicleId),
            "prompt": prompt
        }
        # Parse the output
        if not dynamic:
            parsed_res = TestEvaluationCsv.parse_output(llm_output)
            print(parsed_res)
        else:
            parsed_res = LLM_Dynamic_Weights.parse_output(llm_output)
            print(parsed_res)
        if not parsed_res:  # set edge weight to infinity
            self.graph, success_message = BuildGraph.set_weights_to_inf(self.graph, edgeId)
            if success_message == "SUCCESS":
                # Add incident to incidents
                self.events[edgeId] = {
                    "status": "Yes",
                    "value": "inf",
                    # timestamp in HH:MM:SS
                    "timestamp": time.strftime('%H:%M:%S', time.localtime()),
                    "origin": "Human" if human else "Vehicle " + str(vehicleId),
                    "prompt": prompt
                }

                # Reroute vehicles
                self.reroute_vehicles(edgeId)
                return "SUCCESS"
            else:
                print("ERROR: Could not remove edge")
                return "ERROR"
        self.events[edgeId] = {
            "status": "No",
            "value": 0,
            # timestamp in HH:MM:SS
            "timestamp": time.strftime('%H:%M:%S', time.localtime()),
            "origin": "Human" if human else "Vehicle " + str(vehicleId),
            "prompt": prompt
        }
        return "NO_CHANGE"

    def apply_llm_output_dynamic(self, llm_output_second_stage, llm_output_first_stage, prompt, method, edgeId,
                                 human=True, vehicleId=None):
        """
    Applies the dynamic output from an LLM to update the graph with new edge weights and handles incidents.

    Args:
        llm_output_second_stage (str): The output from the second stage of the LLM, indicating the new weight for the edge.
        llm_output_first_stage (str): The output from the first stage of the LLM, used for initial parsing or validation.
        prompt (str): The prompt sent to the LLM, used for context.
        method (str): The method used for setting the edge weight.
        edgeId (tuple): The ID of the edge (start_node_id, end_node_id) to be updated.
        human (bool, optional): Indicates if the output is from a human or a vehicle. Defaults to True (human).
        vehicleId (str, optional): The ID of the vehicle, used if the output is from a vehicle. Defaults to None.

    Returns:
        str: A status message indicating the result of applying the LLM output:
            - "SUCCESS" if the edge weight was successfully updated and vehicles were rerouted.
            - "ERROR" if the weight could not be set.
    """

        # Parse the output
        parsed_value = LLM_Dynamic_Weights.parse_output_weights(llm_output_second_stage)

        if parsed_value <= 1:
            # Event is not significant
            self.events[edgeId] = {
                "status": "No",
                "value": str(parsed_value) + " " + str(method),
                # timestamp in HH:MM:SS
                "timestamp": time.strftime('%H:%M:%S', time.localtime()),
                "origin": "Human" if human else "Vehicle " + str(vehicleId),
                "prompt": prompt
            }
            return "SUCCESS"

        # Set the weights in the graph
        self.graph, success_message = BuildGraph.set_weight_to_value(self.graph, edgeId, parsed_value, method)

        if success_message == "SUCCESS":
            # Add incident to incidents
            self.events[edgeId] = {
                "status": "Yes",
                "value": str(parsed_value) + " " + str(method),
                # timestamp in HH:MM:SS
                "timestamp": time.strftime('%H:%M:%S', time.localtime()),
                "origin": "Human" if human else "Vehicle " + str(vehicleId),
                "prompt": prompt
            }

            # Reroute vehicles
            self.reroute_vehicles(edgeId)
            return "SUCCESS"
        else:
            print("ERROR: Could not set weight")
            return "ERROR"

    def apply_llm_output_meta(self, llm_output_usability, llm_output_dynamic, llm_output_length, llm_output_time,
                              llm_output_nodes, llm_output_nodes_time, prompt, method, edgeId,
                              human=True, vehicleId=None):
        """
        Applies the meta output from an LLM to update edge weights and handle incidents based on various types of input.

        Args:
            llm_output_usability (str): The output from the LLM related to edge usability, not used in this method.
            llm_output_dynamic (str): The output from the LLM related to dynamic weights, not used in this method.
            llm_output_length (str): The output from the LLM related to edge length.
            llm_output_time (str): The output from the LLM related to travel time.
            llm_output_nodes (str): The output from the LLM related to node information, not used in this method.
            llm_output_nodes_time (str): The output from the LLM related to nodes and time.
            prompt (str): The prompt sent to the LLM, used for context.
            method (str): The method used for setting the edge weight.
            edgeId (tuple or list): The ID(s) of the edge(s) to be updated. Can be a single edge ID or a list of edge IDs.
            human (bool, optional): Indicates if the output is from a human or a vehicle. Defaults to True (human).
            vehicleId (str, optional): The ID of the vehicle, used if the output is from a vehicle. Defaults to None.

        Returns:
            str: A status message indicating the result of applying the LLM output:
                - "SUCCESS" if the edge weights were successfully updated and vehicles were rerouted.
                - "ERROR" if the weight could not be set.
        """

        parsed_value = None
        if llm_output_length is not None:
            parsed_value = LLM_Dynamic_Weights.parse_output_weights(llm_output_length)
        elif llm_output_time is not None:
            parsed_value = LLM_Dynamic_Weights.parse_output_weights(llm_output_time)
        elif llm_output_nodes_time is not None:
            parsed_value = LLM_Dynamic_Weights.parse_output_weights(llm_output_nodes_time)

        if parsed_value is None or parsed_value <= 1:
            # Event is not significant
            self.events[edgeId] = {
                "status": "No",
                "value": str(parsed_value) + " " + str(method),
                # timestamp in HH:MM:SS
                "timestamp": time.strftime('%H:%M:%S', time.localtime()),
                "origin": "Human" if human else "Vehicle " + str(vehicleId),
                "prompt": prompt
            }
            return "SUCCESS"

        results = {}

        if isinstance(edgeId, list):
            for edge in edgeId:
                self.graph, success_message = BuildGraph.set_weight_to_value(self.graph, edge, parsed_value, method)

                if success_message == "SUCCESS":
                    # Add incident to incidents
                    self.events[edge] = {
                        "status": "Yes",
                        "value": str(parsed_value) + " " + str(method),
                        # timestamp in HH:MM:SS
                        "timestamp": time.strftime('%H:%M:%S', time.localtime()),
                        "origin": "Human" if human else "Vehicle " + str(vehicleId),
                        "prompt": prompt
                    }

                    # Reroute vehicles
                    self.reroute_vehicles(edge)
                    results[edge] = "SUCCESS"
                else:
                    print("ERROR: Could not set weight")
                    return "ERROR"

            if results:
                return "SUCCESS"
        else:
            self.graph, success_message = BuildGraph.set_weight_to_value(self.graph, edgeId, parsed_value, method)

            if success_message == "SUCCESS":
                # Add incident to incidents
                self.events[edgeId] = {
                    "status": "Yes",
                    "value": str(parsed_value) + " " + str(method),
                    # timestamp in HH:MM:SS
                    "timestamp": time.strftime('%H:%M:%S', time.localtime()),
                    "origin": "Human" if human else "Vehicle " + str(vehicleId),
                    "prompt": prompt
                }

                # Reroute vehicles
                self.reroute_vehicles(edgeId)
                return "SUCCESS"
            else:
                print("ERROR: Could not set weight")
                return "ERROR"

    def reroute_vehicles(self, obstacle_edge_id):
        """
        Reroutes vehicles that are affected by an obstacle on a given edge.

        Args:
            obstacle_edge_id (tuple): The ID of the edge that is blocked or has an obstacle, represented as a tuple 
                                    (start_node_id, end_node_id)
        """
        affected_vehicles = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle["currentTask"] is not None:
                # Look up the current edge of the vehicle from the vehicle[sequenceId]
                for edge in vehicle["currentTask"]["edges"]:
                    if vehicle["currentSequenceId"] is None:
                        print(f"ERROR: Could not find current sequence id for vehicle {vehicle_id}")
                        break
                    if edge["sequenceId"] > vehicle["currentSequenceId"]:
                        if (str(edge["startNodeId"]) == str(obstacle_edge_id[0]) and str(edge["endNodeId"]) == str(
                                obstacle_edge_id[1])) or (
                                str(edge["startNodeId"]) == str(obstacle_edge_id[1]) and str(edge["endNodeId"]) == str(
                            obstacle_edge_id[0])):
                            affected_vehicles.append(vehicle_id)
                            break

        print("Affected vehicles:", affected_vehicles)

        # Tell every affected vehicle to stop
        for vehicle_id in affected_vehicles:
            self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + f"vehicles/{vehicle_id}/stop", "stop", qos=2)

        # Recalculate routes for affected vehicles
        for vehicle_id in affected_vehicles:
            threading.Thread(target=self.handle_rerouting, args=vehicle_id).start()

    def handle_rerouting(self, vehicle_id):
        """
        Handles the rerouting of a vehicle that has encountered an obstacle.

        This method waits until the specified vehicle has stopped, determines the current node of the vehicle,
        and calculates a new path to the destination, considering whether the vehicle has already visited the 
        source node of the current order. It updates the vehicle's route or cancels it if a new path cannot be 
        found.

        Args:
            vehicle_id (str): The ID of the vehicle that needs to be rerouted.
        """
        while not self.vehicles[vehicle_id]["status"] == "stopped":
            time.sleep(0.1)

        current_node = None
        current_node_index = None
        vehicle = self.vehicles[vehicle_id]
        order_id = vehicle['currentTask']['orderId']
        order = self.orders.get(order_id)
        already_visited = False
        for edge in vehicle["currentTask"]["edges"]:
            # Check if vehicles has already visited the order['source']
            if edge["endNodeId"] == self.get_node_id_from_name(order["source"]):
                print(f"Vehicle {vehicle_id} has already visited the source node of order {order_id}")
                already_visited = True
            if edge["sequenceId"] == vehicle["currentSequenceId"]:
                current_node = edge["endNodeId"]
                current_node_index = edge['sequenceId']
                break
        if current_node_index is None or current_node is None:
            print(f"ERROR: Could not find current node (index) for vehicle {vehicle_id}")

        if already_visited:
            path, _ = self.find_astar_path(self.graph,
                                           current_node,
                                           self.get_node_id_from_name(order["target"]))
            if path is None:
                threading.Thread(target=self.cancel_route_to_vehicle_async,
                                 args=(vehicle_id, order)).start()
                return
        else:
            path_to_source, _ = self.find_astar_path(self.graph,
                                                     current_node,
                                                     self.get_node_id_from_name(order["source"]))
            path_to_target, _ = self.find_astar_path(self.graph,
                                                     self.get_node_id_from_name(order["source"]),
                                                     self.get_node_id_from_name(order["target"]))
            if path_to_source is None or path_to_target is None:
                threading.Thread(target=self.cancel_route_to_vehicle_async,
                                 args=(vehicle_id, order)).start()
                return
            path = path_to_source + path_to_target

        message = self.translate_path_to_mqtt(path, order_id)
        # send the message to the MQTT broker and set vehicle status to busy
        print(f"Rerouting vehicle {vehicle_id}...")
        threading.Thread(target=self.update_route_to_vehicle_async,
                         args=(vehicle_id, message, order)).start()

    def invoke_selected_model(self, value_prompt, value_model, human=True, vehicleId=None, edge_id=None):
        """
        Invokes a specified model to generate LLM output and applies it to update edge weights or handle incidents.

        Args:
            value_prompt (str): The prompt to be sent to the LLM for generating output.
            value_model (str): The model to be used for generating the output. Can be one of:
                - 'gpt-few-shot'
                - 'llama-few-shot'
                - 'gpt-zero-shot'
                - 'llama-zero-shot'
                - 'gpt-few-shot-dynamic'
                - 'llama-few-shot-dynamic'
                - 'gpt-zero-shot-dynamic'
                - 'llama-zero-shot-dynamic'
                - 'meta-model'
            human (bool, optional): Indicates if the output is from a human or a vehicle. Defaults to True (human).
            vehicleId (str, optional): The ID of the vehicle, used if the output is from a vehicle. Defaults to None.
            edge_id (str or list of str, optional): The ID(s) of the edge(s) to be updated. If None, it will be parsed from the prompt.

        Returns:
            tuple: A tuple containing:
                - str: The status of the operation or error message.
                - dict: Styling information for the output display.
        """

        if value_model == 'gpt-few-shot':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output = LLM_Edge_Usability.invoke_llm(value_prompt, "openai", "fewshot")
            success_code = self.apply_llm_output(llm_output, value_prompt, human=human, vehicleId=vehicleId,
                                                 edgeId=edge_id)
        elif value_model == 'llama-few-shot':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output = LLM_Edge_Usability.invoke_llm(value_prompt, "llama2", "fewshot")
            success_code = self.apply_llm_output(llm_output, value_prompt, human=human, vehicleId=vehicleId,
                                                 edgeId=edge_id)
        elif value_model == 'gpt-zero-shot':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output = LLM_Edge_Usability.invoke_llm(value_prompt, "openai", "zeroshot")
            success_code = self.apply_llm_output(llm_output, value_prompt, human=human, vehicleId=vehicleId,
                                                 edgeId=edge_id)
        elif value_model == 'llama-zero-shot':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output = LLM_Edge_Usability.invoke_llm(value_prompt, "llama2", "zeroshot")
            success_code = self.apply_llm_output(llm_output, value_prompt, human=human, vehicleId=vehicleId,
                                                 edgeId=edge_id)
        elif value_model == 'gpt-few-shot-dynamic':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output_second_stage, llm_output, method = LLM_Dynamic_Weights.invoke_llm_chain(value_prompt, "openai",
                                                                                               "fewshot")
            success_code = self.apply_llm_output_dynamic(llm_output_second_stage, llm_output, value_prompt, human=human,
                                                         method=method, vehicleId=vehicleId, edgeId=edge_id)
            llm_output = llm_output + "\n" + llm_output_second_stage
        elif value_model == 'llama-few-shot-dynamic':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output_second_stage, llm_output, method = LLM_Dynamic_Weights.invoke_llm_chain(value_prompt, "llama2",
                                                                                               "fewshot")
            success_code = self.apply_llm_output_dynamic(llm_output_second_stage, llm_output, value_prompt,
                                                         human=human, method=method, vehicleId=vehicleId,
                                                         edgeId=edge_id)
            llm_output = llm_output + "\n" + llm_output_second_stage
        elif value_model == 'gpt-zero-shot-dynamic':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output_second_stage, llm_output, method = LLM_Dynamic_Weights.invoke_llm_chain(value_prompt, "openai",
                                                                                               "zeroshot")
            success_code = self.apply_llm_output_dynamic(llm_output_second_stage, llm_output, value_prompt, human=human,
                                                         method=method, vehicleId=vehicleId, edgeId=edge_id)
            llm_output = llm_output + "\n" + llm_output_second_stage
        elif value_model == 'llama-zero-shot-dynamic':
            if edge_id is None:
                edge_id = self.parse_edge(value_prompt)
            llm_output_second_stage, llm_output, method = LLM_Dynamic_Weights.invoke_llm_chain(value_prompt, "llama2",
                                                                                               "zeroshot")
            success_code = self.apply_llm_output_dynamic(llm_output_second_stage, llm_output, value_prompt,
                                                         human=human, method=method, vehicleId=vehicleId,
                                                         edgeId=edge_id)
            llm_output = llm_output + "\n" + llm_output_second_stage
        elif value_model == 'meta-model':
            llm_output_usability, llm_output_dynamic, llm_output_length, llm_output_time, llm_output_nodes, llm_output_nodes_time, method = LLM_MetaModel.invoke_llm(
                value_prompt)
            if edge_id is None:
                if llm_output_nodes is not None:
                    edge_id = self.parse_edge(llm_output_nodes)
                else:
                    edge_id = self.parse_edge(value_prompt)
            success_code = self.apply_llm_output_meta(llm_output_usability, llm_output_dynamic, llm_output_length,
                                                      llm_output_time,
                                                      llm_output_nodes, llm_output_nodes_time, value_prompt,
                                                      method=method,
                                                      edgeId=edge_id, human=human, vehicleId=vehicleId
                                                      )
            if llm_output_dynamic is not None:
                llm_output = llm_output_usability + "\n" + llm_output_dynamic
                if llm_output_length is not None:
                    llm_output = llm_output + "\n" + llm_output_length
                else:
                    llm_output = llm_output + "\n" + llm_output_time
            elif llm_output_nodes is not None:
                llm_output = llm_output_usability + "\n" + llm_output_nodes
                if llm_output_nodes_time is not None:
                    llm_output = llm_output + "\n" + llm_output_nodes_time
            else:
                llm_output = llm_output_usability



        else:
            print("ERROR: Model not found")
            return "ERROR: Model not found", {'whiteSpace': 'pre-line', 'padding': 5, 'backgroundColor': 'lightgrey',
                                              'font-family': 'Arial, sans-serif', 'display': 'flex', 'flexGrow': 1,
                                              'minHeight': '40px', 'borderRadius': '15px', 'marginTop': '5px'}

        if success_code == "SUCCESS":
            return llm_output, {'whiteSpace': 'pre-line', 'padding': 5, 'backgroundColor': 'lightgrey',
                                'font-family': 'Arial, sans-serif', 'display': 'flex', 'flexGrow': 1,
                                'minHeight': '40px', 'borderRadius': '15px', 'marginTop': '5px'}
        elif success_code == "NO_CHANGE":
            return "Does not affect the graph.", {'whiteSpace': 'pre-line', 'padding': 5,
                                                  'backgroundColor': 'lightgrey',
                                                  'font-family': 'Arial, sans-serif', 'display': 'flex', 'flexGrow': 1,
                                                  'minHeight': '40px', 'borderRadius': '15px', 'marginTop': '5px'}
        else:
            print("ERROR: Could not apply LLM output")
            return "ERROR: Could not apply LLM output", {'whiteSpace': 'pre-line', 'padding': 5,
                                                         'backgroundColor': 'lightgrey',
                                                         'font-family': 'Arial, sans-serif', 'display': 'flex',
                                                         'flexGrow': 1,
                                                         'minHeight': '40px', 'borderRadius': '15px',
                                                         'marginTop': '5px'}

    def get_map(self):
        """
        Generates and returns an interactive map visualization using Plotly and Dash.

        - **Edges**: Displayed as lines connecting nodes. Color and width are defined by `graph_color`.
        - **Nodes**: Displayed as markers. Special nodes have different styling.
        - **Vehicles**: Displayed as markers with colors assigned from `vehicle_colors` list.
        - **Routes**: Show the current route of each vehicle, with different colors for each vehicle.
        - **Incidents**: Displayed as lines and markers, indicating the status of incidents on the edges.

        The map is embedded in a Dash web application with tabs for different views and an interval component for live updates.

        Returns:
            dash.Dash: A Dash application instance with the map visualization embedded in it.
        """

        vehicle_colors = [
            "darkorange", "green", "yellow", "darkred", "lightslategray", "purple", "red", "royalblue",
            "burlywood", "darkslategray", "lemonchiffon", "lightsteelblue", "powderblue", "olivedrab",
            "peru", "gold", "mediumseagreen", "lavenderblush", "skyblue", "tomato", "orange", "darkslategrey",
            "lightgoldenrodyellow", "darkkhaki", "slategray"
        ]
        graph_color = '#4b42f5'

        def get_map_plot():

            # Add weights to the edges_df
            # with lock:
            for _, row in self.edge_df.iterrows():
                try:
                    self.edge_df.at[_, "length"] = self.graph[str(int(row["u"]))][str(int(row["v"]))]["length"]
                except KeyError:
                    self.edge_df.at[_, "length"] = np.NaN

            fig = go.Figure()

            # Add edges to the map
            lats = []
            lons = []
            names = []
            for irow, row in self.edge_df.iterrows():
                lons.append(self.nodes_df.loc[row["u"]]["lon"])
                lons.append(self.nodes_df.loc[row["v"]]["lon"])
                lons.append(None)
                lats.append(self.nodes_df.loc[row["u"]]["lat"])
                lats.append(self.nodes_df.loc[row["v"]]["lat"])
                lats.append(None)
                names.append(f"edge_{int(self.nodes_df.loc[row['u']].name)}_{int(self.nodes_df.loc[row['v']].name)}")
                names.append(None)
            fig.add_trace(go.Scattermapbox(mode='lines',
                                           lon=lons,
                                           lat=lats,
                                           line={'color': graph_color, 'width': 3},  # if on_routes else 3
                                           hoverlabel={'namelength': -1},
                                           text=names,
                                           hoverinfo="text"
                                           ))

            # Add nodes to the map
            for index_node, row in self.nodes_df.iterrows():
                is_special_node = row["name"] is not np.NaN
                if is_special_node:  # currently only display nodes that are special nodes
                    is_special_node = row["name"] is not np.NaN
                    fig.add_trace(go.Scattermapbox(mode='markers',  # if is_special_node else 'markers' markers+text
                                                   lon=[row["lon"]],
                                                   lat=[row["lat"]],
                                                   marker={'color': graph_color, 'size': 20 if is_special_node else 10,
                                                           'allowoverlap': False, 'symbol': 'circle'},
                                                   text=row["name"] if is_special_node else index_node,
                                                   name=row["name"] if is_special_node else index_node,
                                                   hoverinfo="text",
                                                   textposition='bottom center',
                                                   textfont=dict(size=20, color='black')
                                                   ))

            # Add vehicles to the map
            # with lock:
            for vehicle_id, vehicle in self.vehicles.items():
                color_vehicle = vehicle_colors[int(vehicle_id) - 1 % len(vehicle_colors)]
                fig.add_trace(go.Scattermapbox(mode='markers',
                                               lon=[vehicle["position"][1]],
                                               lat=[vehicle["position"][0]],
                                               marker={'color': color_vehicle, 'size': 25, 'allowoverlap': True},
                                               text=f"Vehicle: {vehicle_id}",
                                               name=vehicle_id
                                               ))

            # Add routes to the map
            # with lock:
            for vehicle_id, vehicle in self.vehicles.items():
                if vehicle["currentTask"] is not None:
                    lons = []
                    lats = []
                    current_sequence_id = vehicle["currentSequenceId"]
                    for edge in vehicle["currentTask"]["edges"]:
                        if edge["sequenceId"] >= current_sequence_id:
                            lons.append(self.nodes_df.loc[int(edge["startNodeId"])]["lon"])
                            lons.append(self.nodes_df.loc[int(edge["endNodeId"])]["lon"])

                            lats.append(self.nodes_df.loc[int(edge["startNodeId"])]["lat"])
                            lats.append(self.nodes_df.loc[int(edge["endNodeId"])]["lat"])

                    fig.add_trace(go.Scattermapbox(mode='lines',
                                                   lon=lons,
                                                   lat=lats,
                                                   line={'color': vehicle_colors[
                                                       int(vehicle_id) - 1 % len(vehicle_colors)], 'width': 5},
                                                   name=f"Route of vehicle {vehicle}",
                                                   hoverinfo='name',
                                                   hoverlabel={'namelength': -1}
                                                   ))

            # Add incidents to the map
            # with lock:
            for edge_id, incident_value in self.events.items():
                lons = [self.nodes_df.loc[edge_id[0]]["lon"], self.nodes_df.loc[edge_id[1]]["lon"]]
                lats = [self.nodes_df.loc[edge_id[0]]["lat"], self.nodes_df.loc[edge_id[1]]["lat"]]
                # Case 1: Incident status = 'Yes'
                if incident_value["status"] == "Yes":
                    fig.add_trace(go.Scattermapbox(mode='lines',
                                                   lon=lons,
                                                   lat=lats,
                                                   line={'color': 'black', 'width': 5},
                                                   name=f"Incident on edge_{int(edge_id[0])}_{int(edge_id[1])}",
                                                   hoverinfo='name',
                                                   hoverlabel={'namelength': -1}
                                                   ))

                    # Add markers in the middle of the edge
                    fig.add_trace(go.Scattermapbox(
                        lon=[(self.nodes_df.loc[edge_id[0]]["lon"] + self.nodes_df.loc[edge_id[1]]["lon"]) / 2],
                        lat=[(self.nodes_df.loc[edge_id[0]]["lat"] + self.nodes_df.loc[edge_id[1]]["lat"]) / 2],
                        marker=go.scattermapbox.Marker(
                            size=17,
                            color='rgb(255, 0, 0)',
                            opacity=1
                        )
                    ))

                    fig.add_trace(go.Scattermapbox(
                        lon=[(self.nodes_df.loc[edge_id[0]]["lon"] + self.nodes_df.loc[edge_id[1]]["lon"]) / 2],
                        lat=[(self.nodes_df.loc[edge_id[0]]["lat"] + self.nodes_df.loc[edge_id[1]]["lat"]) / 2],
                        marker=go.scattermapbox.Marker(
                            size=8,
                            color='rgb(242, 177, 172)',
                            opacity=1
                        ),
                        # parse everything after edge_x_y
                        text=re.sub(r"edge_[0-9]+_[0-9]+", "", incident_value["prompt"]),
                        hoverinfo='text'
                    ))

                # Case 2: Incident status = 'No'
                else:
                    # Add markers in the middle of the edge
                    fig.add_trace(go.Scattermapbox(
                        lon=[(self.nodes_df.loc[edge_id[0]]["lon"] + self.nodes_df.loc[edge_id[1]]["lon"]) / 2],
                        lat=[(self.nodes_df.loc[edge_id[0]]["lat"] + self.nodes_df.loc[edge_id[1]]["lat"]) / 2],
                        marker=go.scattermapbox.Marker(
                            size=17,
                            color='rgb(0, 255, 0)',
                            opacity=1
                        )
                    ))

                    fig.add_trace(go.Scattermapbox(
                        lon=[(self.nodes_df.loc[edge_id[0]]["lon"] + self.nodes_df.loc[edge_id[1]]["lon"]) / 2],
                        lat=[(self.nodes_df.loc[edge_id[0]]["lat"] + self.nodes_df.loc[edge_id[1]]["lat"]) / 2],
                        marker=go.scattermapbox.Marker(
                            size=8,
                            color='rgb(177, 242, 172)',
                            opacity=1
                        ),
                        text=re.sub(r"edge_[0-9]+_[0-9]+", "", incident_value["prompt"]),
                        hoverinfo='text'
                    ))

            fig.update_layout(mapbox_style="open-street-map",
                              mapbox_zoom=15.7,
                              mapbox_center_lat=48.00632,
                              mapbox_center_lon=7.838,
                              margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              autosize=True,
                              # width=1050,
                              # height=850,
                              showlegend=False)

            fig['layout']['uirevision'] = 'currentZoom'

            return fig

        app = Dash(__name__)

        app.layout = html.Div([
            html.H1(children='Intelligent Hospital Logistics',
                    style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                           'height': '20px'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='live-update-graph',
                              responsive=True,
                              style={
                                  'width': '100%',
                                  'height': '100%'}),
                    dcc.Interval(
                        id='interval-component',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    )
                ], style={'padding': 5, 'flex': 2}),
                html.Div([
                    dcc.Tabs(id="tabs",
                             value='tab-1',
                             colors={'border': '#d6d6d6', 'primary': '#99C554', 'background': '#f9f9f9'},
                             children=[
                                 dcc.Tab(label='Incidents', value='tab-1'),
                                 dcc.Tab(label='Events', value='tab-events'),
                                 dcc.Tab(label='Status', value='tab-2'),
                                 dcc.Tab(label='Prompts', value='tab-prompts')
                             ],
                             style={
                                 'height': '100%'
                             }),
                    html.Div(id='tabs-content',
                             style={
                                 'height': '100%',
                                 'maxHeight': 'calc(95vh - 100px)',
                                 'overflowY': 'auto'
                             })
                ], style={'padding': 5, 'flex': 1, 'flexDirection': 'row', 'minWidth': '100px', 'maxWidth': '500px',
                          'height': '100%'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'stretch',
                      'height': 'calc(100% - 20px)'
                      })
        ], style={'fontFamily': 'Arial, sans-serif',
                  'width': '100%',
                  # 'height': '100%'
                  'height': '95vh',
                  'maxHeight': '95vh'
                  })

        # For buttons
        button_style = {'padding': 5, 'backgroundColor': '#99C554', 'border': 'none', 'color': 'white',
                        'borderRadius': '15px'}

        # For radio items
        radio_items_style = {'display': 'inline-block', 'marginRight': '15px', 'color': 'white',
                             'fontFamily': 'Arial, sans-serif'}

        # Apply the styles to the buttons and radio items
        html.Button('Submit', id='press-invoke-llm', n_clicks=0, style=button_style)
        dcc.RadioItems(["off", "on"], self.generate_incidents, id='generate-incidents', style=radio_items_style)

        @app.callback(Output('tabs-content', 'children'),
                      Input('tabs', 'value'))
        def render_content(tab):
            if tab == 'tab-1':
                return html.Div([
                    html.H2("Ongoing Incidents",
                            style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                   'marginTop': '30px'}),
                    html.Div(dash_table.DataTable(id='incidents-tbl', cell_selectable=False,
                                                  style_data_conditional=[
                                                      {
                                                          "if": {"state": "selected"},
                                                          "backgroundColor": "inherit !important",
                                                          "border": "inherit !important",
                                                      }
                                                  ],
                                                  style_header={
                                                      'backgroundColor': 'lightgrey',
                                                      'fontWeight': 'bold'
                                                  },
                                                  style_cell={
                                                      'backgroundColor': 'white',
                                                      'color': 'black',
                                                      'border': '1px solid grey',
                                                      'maxWidth': '150px',
                                                      'overflow': 'hidden',
                                                      'textOverflow': 'ellipsis',
                                                      'whiteSpace': 'normal'
                                                  },
                                                  style_table={
                                                      'height': '100%',
                                                      'overflowY': 'scroll'
                                                  },
                                                  fixed_rows={'headers': True},
                                                  ),
                             style={'width': '100%'}),
                    dcc.Interval(
                        id='table-update-interval-incidents',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    ),
                ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
            elif tab == 'tab-2':
                return html.Div([
                    html.H2("Orders",
                            style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                   'marginTop': '30px',
                                   }),
                    html.Div(dash_table.DataTable(id='tbl', cell_selectable=False,
                                                  style_data_conditional=[
                                                      {
                                                          "if": {"state": "selected"},
                                                          "backgroundColor": "inherit !important",
                                                          "border": "inherit !important",
                                                      }
                                                  ],
                                                  style_header={
                                                      'backgroundColor': 'lightgrey',
                                                      'fontWeight': 'bold'
                                                  },
                                                  style_cell={
                                                      'backgroundColor': 'white',
                                                      'color': 'black',
                                                      'border': '1px solid grey',
                                                      'maxWidth': '150px',
                                                      'overflow': 'hidden',
                                                      'textOverflow': 'ellipsis',
                                                      'whiteSpace': 'normal'
                                                  },
                                                  style_table={
                                                      'overflowY': 'scroll'
                                                  },
                                                  fixed_rows={'headers': True},
                                                  ),
                             style={
                                 'width': '100%',
                             }
                             ),
                    dcc.Interval(
                        id='table-update-interval',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    ),
                    html.H2("Vehicles",
                            style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                   'marginTop': '30px',
                                   }),
                    html.Div(dash_table.DataTable(id='vehicle-table', cell_selectable=False,
                                                  style_data_conditional=[
                                                      {
                                                          "if": {"state": "selected"},
                                                          "backgroundColor": "inherit !important",
                                                          "border": "inherit !important",
                                                      }
                                                  ],
                                                  style_header={
                                                      'backgroundColor': 'lightgrey',
                                                      'fontWeight': 'bold'
                                                  },
                                                  style_cell={
                                                      'backgroundColor': 'white',
                                                      'color': 'black',
                                                      'border': '1px solid grey'
                                                  },
                                                  style_table={
                                                      'overflowY': 'scroll'
                                                  },
                                                  fixed_rows={'headers': True},
                                                  ),
                             style={
                                 'width': '100%',
                             }),
                    dcc.Interval(
                        id='vehicle-table-update-interval',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    )
                ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
            elif tab == 'tab-events':
                return html.Div([
                    html.H2("Events",
                            style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                   'marginTop': '30px'}),
                    html.Div(dash_table.DataTable(id='events-tbl', cell_selectable=False,
                                                  style_data_conditional=[
                                                      {
                                                          "if": {"state": "selected"},
                                                          "backgroundColor": "inherit !important",
                                                          "border": "inherit !important",
                                                      },
                                                      {
                                                          'if': {
                                                              'filter_query': '{Res.} = Yes',
                                                              'column_id': 'Res.'
                                                          },
                                                          'backgroundColor': 'red'
                                                      },
                                                      {
                                                          'if': {
                                                              'filter_query': '{Res.} = No',
                                                              'column_id': 'Res.'
                                                          },
                                                          'backgroundColor': 'green'
                                                      }
                                                  ],
                                                  style_header={
                                                      'backgroundColor': 'lightgrey',
                                                      'fontWeight': 'bold'
                                                  },
                                                  style_cell={
                                                      'backgroundColor': 'white',
                                                      'color': 'black',
                                                      'border': '1px solid grey',
                                                      'maxWidth': '150px',
                                                      'overflow': 'hidden',
                                                      'textOverflow': 'ellipsis',
                                                      'whiteSpace': 'normal'
                                                  },
                                                  style_table={
                                                      'height': 'height',
                                                      'overflowY': 'scroll'
                                                  },
                                                  fixed_rows={'headers': True},
                                                  ),
                             style={'maxWidth': '100%'}),
                    dcc.Interval(
                        id='table-update-interval-events',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    ),
                ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
            elif tab == 'tab-prompts':
                return html.Div([
                    html.H2("LLM",
                            style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                   'marginTop': '30px'}),
                    html.Div([
                        dcc.Textarea(id='input-prompt', value='Prompt...',
                                     style={'height': 60, 'padding': 5, 'flex': 1, 'borderRadius': '15px',
                                            'marginRight': '5px'}),
                        html.Div([
                            dcc.Dropdown(
                                id='llm-model-dropdown',
                                options=[
                                    {'label': 'GPT-3.5-Few', 'value': 'gpt-few-shot'},
                                    {'label': 'LLAMA-2-Few', 'value': 'llama-few-shot'},
                                    {'label': 'GPT-3.5-Zero', 'value': 'gpt-zero-shot'},
                                    {'label': 'LLAMA-2-Zero', 'value': 'llama-zero-shot'},
                                    {'label': 'GPT-3.5-Few-Dynamic', 'value': 'gpt-few-shot-dynamic'},
                                    {'label': 'LLAMA-2-Few-Dynamic', 'value': 'llama-few-shot-dynamic'},
                                    {'label': 'GPT-3.5-Zero-Dynamic', 'value': 'gpt-zero-shot-dynamic'},
                                    {'label': 'LLAMA-2-Zero-Dynamic', 'value': 'llama-zero-shot-dynamic'},
                                    {'label': 'GPT-3.5-Meta', 'value': 'meta-model'},
                                ],
                                value='gpt-few-shot',
                                style={'borderRadius': '15px', 'padding': 5, 'width': '95%'}
                            ),
                            html.Button('Submit', id='press-invoke-llm', n_clicks=0,
                                        style={'padding': 5, 'backgroundColor': '#99C554', 'border': 'none',
                                               'color': 'white', 'borderRadius': '15px', 'width': '95%'}),
                        ], style={'alignItems': 'center', 'flex': 1, 'flexDirection': 'column', 'display': 'flex'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'}),
                    html.Label(id='llm-output',
                               style={'whiteSpace': 'pre-line', 'padding': 5, 'backgroundColor': 'lightgrey',
                                      'font-family': 'Arial, sans-serif', 'display': 'none', 'flexGrow': 1,
                                      'minHeight': '40px', 'borderRadius': '15px', 'marginTop': '5px'}),
                    html.Div([
                        html.H2("Prompt Generation",
                                style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                       'marginTop': '60px'}),
                        html.Div([
                            html.Div([
                                html.Label('Seed used by all vehicles:',
                                           style={'font-family': 'Arial, sans-serif', 'color': 'black',
                                                  'marginRight': '20px', 'width': '35%'}),
                                dcc.Input(id='random-seed', type='number', value=42,
                                          style={'marginRight': '5px', 'width': '10%', 'borderRadius': '15px'}),
                                html.Button('Update Seed', id='update-seed-button', n_clicks=0,
                                            style={'padding': 5, 'backgroundColor': '#99C554', 'border': 'none',
                                                   'color': 'white', 'borderRadius': '15px', 'marginRight': '5px'}),
                                html.Button('Randomize Seed', id='choose-random-seed-button', n_clicks=0,
                                            style={'padding': 5, 'backgroundColor': '#99C554', 'border': 'none',
                                                   'color': 'white', 'borderRadius': '15px'})
                            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'left'}),
                            html.Div([
                                html.Label('Generate prompts from vehicles:',
                                           style={'font-family': 'Arial, sans-serif', 'color': 'black',
                                                  'marginRight': '20px', 'width': '35%'}),
                                dcc.RadioItems(["off", "on"], self.generate_incidents, id='generate-incidents',
                                               inline=True, style={'width': '10%'})
                            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'left',
                                      'marginTop': '30px'}),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'left',
                                  'marginTop': '10px'})
                    ])
                ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})

        @app.callback(Output('live-update-graph', 'figure'),
                      Input('interval-component', 'n_intervals'))
        def update_metrics(_):  # don't care about the input
            return get_map_plot()

        @app.callback(
            [Output('llm-output', 'children'),
             Output('llm-output', 'style')],
            Input('press-invoke-llm', 'n_clicks'),
            [State('input-prompt', 'value'),
             State('llm-model-dropdown', 'value')],
            prevent_initial_call=True
        )
        def update_output(_, value_prompt, value_model):
            return self.invoke_selected_model(value_prompt, value_model)

        @app.callback(
            Output('choose-random-seed-button', 'style'),
            [Input('llm-model-dropdown', 'value'),
             Input('choose-random-seed-button', 'style')],
            prevent_initial_call=True,
        )
        def update_model(value_model, style_hack):
            self.current_model = value_model
            print(self.current_model)
            return style_hack

        @app.callback(
            Output('tbl', 'data'),
            Input('table-update-interval', 'n_intervals')
        )
        def update_table(_):  # don't care about the input
            # Convert the orders dictionary to a list of dictionaries, which is the format required by DataTable
            orders_list = [v for v in self.orders.values()]

            orders_list_copy = copy.deepcopy(orders_list)

            for order in orders_list_copy:
                # Set vehicle_id to vehicle color
                if order["vehicle_id"] is not None:
                    try:
                        vehicle_id_int = int(order["vehicle_id"])
                        order["vehicle_id"] = vehicle_colors[vehicle_id_int - 1 % len(vehicle_colors)]
                    except ValueError:
                        pass
                    order["Vehicle"] = order["vehicle_id"]
                    del order["vehicle_id"]
                if 'timestamp' in order:
                    del order['timestamp']
                if 'order_id' in order:
                    del order['order_id']

            return orders_list_copy

        @app.callback(
            Output('tbl', 'style_data_conditional'),
            Input('tbl', 'data')
        )
        def update_order_table_style(data):
            style_data_conditional = []
            for row in data:
                style_data_conditional.append({
                    'if': {'column_id': 'Vehicle', 'row_index': data.index(row)},
                    'backgroundColor': row['Vehicle']
                })
            return style_data_conditional

        @app.callback(
            Output('vehicle-table', 'data'),
            Input('vehicle-table-update-interval', 'n_intervals')
        )
        def update_vehicle_table(_):  # don't care about the input
            # Convert the vehicles dictionary to a list of dictionaries, which is the format required by DataTable
            vehicles_list = [v for v in self.vehicles.values()]
            vehicle_ids = [k for k in self.vehicles.keys()]

            # check if vehicles are empty
            if len(vehicles_list) == 0:
                return no_update

            vehicle_ids_copy = copy.deepcopy(vehicle_ids)
            vehicles_list_copy = copy.deepcopy(vehicles_list)

            # sort both lists by vehicle_id
            vehicle_ids_copy, vehicles_list_copy = zip(*sorted(zip(vehicle_ids_copy, vehicles_list_copy)))

            for index, vehicle in enumerate(vehicles_list_copy):
                # Set vehicle_id to vehicle color
                vehicle["vehicle_id"] = vehicle_ids_copy[index]
                if vehicle["vehicle_id"] is not None:
                    try:
                        vehicle["vehicle_id"] = vehicle_colors[int(vehicle["vehicle_id"]) - 1 % len(vehicle_colors)]
                    except ValueError:
                        print(f"Could not convert vehicle_id to color: {vehicle['vehicle_id']}")
                    vehicle["Vehicle"] = vehicle["vehicle_id"]
                    del vehicle["vehicle_id"]
                if 'timestamp' in vehicle:
                    del vehicle['timestamp']
                if 'headerId' in vehicle:
                    del vehicle['headerId']
                if 'currentTask' in vehicle:
                    del vehicle['currentTask']
                if 'order_id' in vehicle:
                    del vehicle['order_id']
                if 'speeed' in vehicle:
                    del vehicle['speeed']
                if 'currentSequenceId' in vehicle:
                    del vehicle['currentSequenceId']
                if 'position' in vehicle:
                    vehicle['position'] = f"({vehicle['position'][0]:.8f}, {vehicle['position'][1]:.8f})"
                if 'vehicleId' in vehicle:
                    del vehicle['vehicleId']

            return vehicles_list_copy

        @app.callback(
            Output('vehicle-table', 'style_data_conditional'),
            Input('vehicle-table', 'data')
        )
        def update_table_style(data):
            style_data_conditional = []
            for row in data:
                style_data_conditional.append({
                    'if': {'column_id': 'Vehicle', 'row_index': data.index(row)},
                    'backgroundColor': row['Vehicle']
                })
            return style_data_conditional

        @app.callback(
            Output('incidents-tbl', 'data'),
            Input('table-update-interval-incidents', 'n_intervals')
        )
        def update_incidents_table(_):
            incidents_edge_ids = [f"edge_{edge[0]}_{edge[1]}" for edge in self.events.keys()]
            incidents_attributes = [incident for incident in self.events.values()]

            incidents_edge_ids_copy = copy.deepcopy(incidents_edge_ids)
            incidents_attributes_copy = copy.deepcopy(incidents_attributes)

            # Filter the incidents with status "Incident"
            incidents_attributes_copy = [incident for incident in incidents_attributes_copy if
                                         incident.get('status') == 'Yes']

            for index, incident in enumerate(incidents_attributes_copy):
                # Set vehicle_id to vehicle color
                incident["Edge_ID"] = incidents_edge_ids_copy[index]
                if 'status' in incident:
                    del incident['status']
                if 'prompt' in incident:
                    del incident['prompt']
                if 'timestamp' in incident:
                    incident["Timestamp"] = incident["timestamp"]
                    del incident['timestamp']
                if 'origin' in incident:
                    incident["Origin"] = incident["origin"]
                    del incident['origin']
                if 'value' in incident:
                    incident["Val."] = incident["value"]
                    del incident['value']

            return incidents_attributes_copy

        @app.callback(
            Output('events-tbl', 'data'),
            Input('table-update-interval-events', 'n_intervals')
        )
        def update_events_table(_):  # don't care about the input
            incidents_edge_ids = [f"edge_{edge[0]}_{edge[1]}" for edge in self.events.keys()]
            incidents_attributes = [incident for incident in self.events.values()]

            incidents_edge_ids_copy = copy.deepcopy(incidents_edge_ids)
            incidents_attributes_copy = copy.deepcopy(incidents_attributes)

            for index, incident in enumerate(incidents_attributes_copy):
                # Set vehicle_id to vehicle color
                if 'status' in incident:
                    incident["Res."] = incident["status"]
                    del incident['status']
                if 'value' in incident:
                    incident["Val."] = incident["value"]
                    del incident['value']
                if 'timestamp' in incident:
                    incident["Timestamp"] = incident["timestamp"]
                    del incident['timestamp']
                if 'origin' in incident:
                    incident["Origin"] = incident["origin"]
                    del incident['origin']
                if 'prompt' in incident:
                    incident["Prompt"] = incident["prompt"]
                    del incident['prompt']

            return incidents_attributes_copy

        @app.callback(
            Output('random-seed', 'value', allow_duplicate=True),
            Input('choose-random-seed-button', 'n_clicks'),
            State('random-seed', 'value'),
            prevent_initial_call=True
        )
        def choose_random_seed(n_clicks, value):
            value = random.randrange(sys.maxsize)
            return value

        @app.callback(
            Output('random-seed', 'value'),
            Input('update-seed-button', 'n_clicks'),
            State('random-seed', 'value'),
            prevent_initial_call=True
        )
        def update_random_seed(n_clicks, value):
            for index, vehicle_id in enumerate(self.vehicles.keys()):
                self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + f"vehicles/{vehicle_id}/random_seed",
                                    value + index, qos=2)
                time.sleep(0.1)
            return value

        @app.callback(
            Output('generate-incidents', 'value'),
            Input('generate-incidents', 'value'),
            prevent_initial_call=True
        )
        def update_generate_incidents(value):
            self.generate_incidents = value
            for vehicle_id in self.vehicles.keys():
                self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + f"vehicles/{vehicle_id}/generate_incidents",
                                    value, qos=2)
                time.sleep(0.1)
            return value

        app.run(debug=False, dev_tools_hot_reload=False, dev_tools_silence_routes_logging=True)
