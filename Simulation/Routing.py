import json
import os
import time
import warnings

import networkx as nx
import paho.mqtt.client as paho
from dotenv import load_dotenv
from paho import mqtt

# from Simulation import LLM
import LLM

warnings.filterwarnings("ignore")

import threading

lock = threading.Lock()

from dash import Dash, html, dcc, Output, Input, State
import plotly.graph_objects as go


class Routing():  # singleton class. Do not create more than one object of this class
    def __init__(self, graph, edge_df, nodes_df):
        load_dotenv()
        self.graph = graph
        self.edge_df = edge_df
        self.nodes_df = nodes_df
        self.vehicles = {}
        self.connect_to_mqtt()

    def __repr__(self):
        return f"Graph: {self.graph}"

    # define function to find the shortest path between two special nodes
    def find_astar_path(self, G, start_node_id, end_node_id):
        # use the a* algorithm to find the shortest path between the source and target node
        shortest_path = nx.astar_path(G, str(start_node_id), str(end_node_id), weight='weight')
        return shortest_path

    def find_dijkstra_path(self, G, start_node_id, end_node_id):
        # use Dijkstra's algorithm to find the shortest path between the source and target node
        shortest_path_length, shortest_path = nx.single_source_dijkstra(G, start_node_id, target=end_node_id,
                                                                        weight='weight')
        return shortest_path

    def find_bellman_ford_path(self, G, start_node_id, end_node_id):
        # use Bellman-Ford algorithm to find the shortest path between the source and target node
        shortest_path = nx.bellman_ford_path(G, start_node_id, end_node_id, weight='weight')
        return shortest_path

    def evaluate_shortest_path_weight(G, shortest_path):
        total_weight = 0
        for i in range(len(shortest_path) - 1):
            source = shortest_path[i]
            target = shortest_path[i + 1]
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
                 'endNodeId': edge[1], 'startCoordinate': tuple(self.nodes_df.loc[int(edge[0])].loc[["lat", "lon"]]),
                 'endCoordinate': tuple(self.nodes_df.loc[int(edge[1])].loc[["lat", "lon"]])})
        message = {'edges': edges}
        # print(message)
        # print(json.dumps(message))
        return message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("CONNACK received with code %s." % rc)

    def on_publish(self, client, userdata, mid, properties=None):
        # print("mid: " + str(mid))
        pass

    def on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(self, client, userdata, msg):
        # print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

        if msg.topic.startswith(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/") and msg.topic.endswith("/status"):
            vehicle_id = msg.topic.split("/")[2]
            vehicle_status = json.loads(msg.payload.decode())
            self.vehicles[vehicle_id] = vehicle_status
            # print(f"Received status of vehicle {vehicle_id}: {vehicle_status}")
        else:  # received order
            # print("Received new task: " + msg.payload.decode("utf-8"))
            self.handle_order(msg.payload.decode("utf-8"))

    def get_distance(self, start_node_id, end_node_id):
        return nx.astar_path_length(self.graph, str(start_node_id), str(end_node_id), weight='weight')

    def get_distance_from_vehicle_to_order(self, vehicle_id, order):
        vehicle = self.vehicles[vehicle_id]
        start_node_id = self.get_node_id_from_name(order["source"])
        return self.get_distance(vehicle["targetNode"], start_node_id)

    def get_node_id_from_name(self, name):
        return self.nodes_df[self.nodes_df["name"] == name].index[0]

    def get_distances_from_vehicles_to_order(self, order, use_only_idle_vehicles):
        distances = {}
        for vehicle_id, vehicle in self.vehicles.items():
            if not use_only_idle_vehicles or vehicle["status"] == "idle":
                distances[vehicle_id] = self.get_distance_from_vehicle_to_order(vehicle_id, order)
        # print(distances)
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
        # print(f"Received new order: {order}")
        # find the shortest path

        shortest_path_astar = self.find_astar_path(self.graph, self.get_node_id_from_name(order["source"]),
                                                   self.get_node_id_from_name(order["target"]))
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
        vehicle_id = self.get_vehicle_id_for_order(order)
        # Set vehicle status to busy
        threading.Thread(target=self.send_route_to_vehicle_async, args=(vehicle_id, message)).start()

    def send_route_to_vehicle_async(self, vehicle_id, route):  # please call this method async
        while self.vehicles[vehicle_id]["status"] != "idle":
            time.sleep(5)
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + f"vehicles/{vehicle_id}/route", json.dumps(route),
                            qos=2)
        self.vehicles[vehicle_id]["status"] = "busy"

    def connect_to_mqtt(self):
        # Connect to MQTT
        self.client = paho.Client(protocol=paho.MQTTv5)
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
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "order_manager/transportation/orders/#", qos=2)
        # subscribe to vehicle status
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/+/status", qos=2)
        print("simulation online")
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "hello", "simulation online", qos=2)
        threading.Thread(target=self.folium_plot).start()
        # self.folium_plot()
        # print("start mqtt loop")
        threading.Thread(target=LLM.main, args=[self]).start()
        self.client.loop_forever()

    def folium_plot(self):

        def getmap():
            fig = go.Figure()
            with lock:
                for _, row in self.edge_df.iterrows():
                    fig.add_trace(go.Scattermapbox(mode='lines',
                                                   lon=[self.nodes_df.loc[row["u"]]["lon"],
                                                        self.nodes_df.loc[row["v"]]["lon"]],
                                                   lat=[self.nodes_df.loc[row["u"]]["lat"],
                                                        self.nodes_df.loc[row["v"]]["lat"]],
                                                   marker={'color': "grey", 'size': 15, 'allowoverlap': True}
                                                   # name=row['source_name']
                                                   ))

            # fig = px.line_mapbox(self.nodes_df.loc[self.edge_df["u"]], lat="lat", lon="lon", hover_data="name", zoom=3, height=300)

            # Add vehicles to the map
            with lock:
                for vehicle in self.vehicles.keys():
                    fig.add_trace(go.Scattermapbox(mode='markers',
                                                   lon=[self.vehicles[vehicle]["position"][1]],
                                                   lat=[self.vehicles[vehicle]["position"][0]],
                                                   marker={'color': "red", 'size': 15, 'allowoverlap': True},
                                                   text=f"Vehicle: {vehicle}",
                                                   name=vehicle
                                                   ))

            # fig.update_traces(marker_symbol="car", selector=dict(type='scattermapbox'))

            fig.update_layout(mapbox_style="open-street-map",
                              mapbox_zoom=17.5,
                              mapbox_center_lat=48.00632,
                              mapbox_center_lon=7.838,
                              margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              width=900, height=800,
                              showlegend=False)

            return fig

        app = Dash(__name__)
        app.layout = html.Div([
            html.H1(children='Intelligent Hospital Logistics',
                    style={'textAlign': 'left', 'font-family': 'Arial, sans-serif'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='live-update-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=0.5 * 1000,  # in milliseconds
                        n_intervals=0
                    )
                ], style={'padding': 10, 'flex': 1}),
                html.Div([
                    html.H2("LLM", style={'textAlign': 'left', 'font-family': 'Arial, sans-serif'}),
                    html.Div([
                        dcc.Textarea(id='input-prompt', value='Prompt...', style={'height': 60, 'padding': 10, 'flex': 10}),
                        html.Button('Submit', id='press-invoke-llm', n_clicks=0, style={'padding': 10, 'flex': 1}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'padding': 10}),
                    html.Label(id='llm-output', style={'whiteSpace': 'pre-line', 'padding': 10})
                ], style={'padding': 10, 'flex': 1})
            ], style={'display': 'flex', 'flexDirection': 'row'})
            ])

        @app.callback(Output('live-update-graph', 'figure'),
                      Input('interval-component', 'n_intervals'))
        def update_metrics(n):
            print("updating map...")
            return getmap()

        @app.callback(
            Output('llm-output', 'children'),
            Input('press-invoke-llm', 'n_clicks'),
            State('input-prompt', 'value'),
            prevent_initial_call=True
        )
        def update_output(n_clicks, value):
            return value

        app.run(debug=False)
