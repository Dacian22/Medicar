import copy
import json
import os
import re
import threading
import time
import warnings

import networkx as nx
import numpy as np
import paho.mqtt.client as paho
import plotly.graph_objects as go
from dash import Dash, dash_table
from dash import html, dcc, Output, Input, State, no_update
from dotenv import load_dotenv
from paho import mqtt

import BuildGraph
import Playground_LLM_Dacian
import TestEvaluationCsv

lock = threading.Lock()
warnings.filterwarnings("ignore")


class Routing():  # singleton class. Do not create more than one object of this class
    def __init__(self, graph, edge_df, nodes_df):
        load_dotenv(override=True)
        self.graph = graph
        self.edge_df = edge_df
        self.nodes_df = nodes_df
        self.vehicles = {}
        self.orders = {}
        self.connect_to_mqtt()

    def __repr__(self):
        return f"Graph: {self.graph}"

    # define function to find the shortest path between two special nodes
    def find_astar_path(self, G, start_node_id, end_node_id):
        # use the a* algorithm to find the shortest path between the source and target node
        shortest_path = nx.astar_path(G, str(start_node_id), str(end_node_id), weight='length')
        return shortest_path

    def find_dijkstra_path(self, G, start_node_id, end_node_id):
        # use Dijkstra's algorithm to find the shortest path between the source and target node
        shortest_path_length, shortest_path = nx.single_source_dijkstra(G, start_node_id, target=end_node_id,
                                                                        weight='length')
        return shortest_path

    def find_bellman_ford_path(self, G, start_node_id, end_node_id):
        # use Bellman-Ford algorithm to find the shortest path between the source and target node
        shortest_path = nx.bellman_ford_path(G, start_node_id, end_node_id, weight='length')
        return shortest_path

    def evaluate_shortest_path_weight(G, shortest_path):
        total_weight = 0
        for i in range(len(shortest_path) - 1):
            source = shortest_path[i]
            target = shortest_path[i + 1]
            edge_weight = G[source][target]['length']
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
        else:  # received order
            # Add order to self.orders
            order = json.loads(msg.payload.decode("utf-8"))
            self.orders[order["order_id"]] = order
            self.handle_order(order)

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
        vehicle_id = self.get_vehicle_id_for_order(order)
        # Update vehicle id in self.orders
        order["vehicle_id"] = vehicle_id

        shortest_path_astar = self.find_astar_path(self.graph, self.get_node_id_from_name(order["source"]),
                                                   self.get_node_id_from_name(order["target"]))
        if self.vehicles[order["vehicle_id"]]["targetNode"] != self.get_node_id_from_name(order["source"]):
            shortest_path_astar_to_start_node = self.find_astar_path(self.graph,
                                                                     self.vehicles[order["vehicle_id"]]["targetNode"],
                                                                     self.get_node_id_from_name(order["source"]))
            # add the two paths together
            shortest_path_astar = shortest_path_astar_to_start_node + shortest_path_astar
        # translate the shortest path to MQTT messages
        print("Old shortest path:", shortest_path_astar)
        message = self.translate_path_to_mqtt(shortest_path_astar)
        # send the message to the MQTT broker and set vehicle status to busy
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
        threading.Thread(target=self.get_map).start()
        self.client.loop_forever()

    def apply_llm_output(self, llm_output):
        # Parse the output
        print(llm_output)
        parsed_res = TestEvaluationCsv.parse_output(llm_output)
        print(parsed_res)
        if not parsed_res:  # remove edge
            try:
                # Get edge from llm_output
                re_str = r"edge_([0-9]+)_([0-9]+)"
                result = re.findall(re_str, llm_output)
                print(result)
                edge_id = int(re.findall(re_str, llm_output)[0][0]), int(re.findall(re_str, llm_output)[0][1])
            except IndexError:
                pattern = r"\([`']?\d+[`']?,.?[`']?\d+[`']?\)"
                edge_id = re.findall(pattern, llm_output)
                edge_id = edge_id[0].strip("'´")
                print(edge_id)
            print(f"trying to remove edge {edge_id}...")

            # Update graph in the routing
            self.graph = BuildGraph.set_weights_to_inf(self.graph, edge_id)
            self.reroute_vehicles(edge_id)

    def reroute_vehicles(self, obstacle_edge_id):
        print("Obstacle: ", obstacle_edge_id)
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle["currentTask"] is not None:
                print(f"Vehicle {vehicle_id} current task edges:")
                for edge in vehicle["currentTask"]["edges"]:
                    print(edge)
            else:
                print(f"Vehicle {vehicle_id} has no current task.")
        # Get all vehicles that are currently moving and will pass through the obstacle edge
        affected_vehicles = []
        # Detect Obstacle on Edge
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle["status"] == "moving":
                print("Vehicle moving:", vehicle_id)
                current_position = vehicle['position']
                print("Current node", current_position)
                current_node_index = None
                obstacle_index = None
                for task_edge in vehicle["currentTask"]["edges"]:
                    # Extract start and end nodes of the current task edge
                    task_start_node = task_edge["startNodeId"]
                    task_end_node = task_edge["endNodeId"]
                    task_start_coord = task_edge["startCoordinate"]
                    task_end_coord = task_edge["endCoordinate"]

                    if task_start_coord == current_position:
                        current_node = task_start_node
                        if current_node_index is None:
                            current_node_index = task_edge["sequenceId"]
                            print("Current node index", current_node_index)
                    elif task_end_coord == current_position:
                        current_node = task_end_node
                        if current_node_index is None:
                            current_node_index = task_edge["sequenceId"]
                            print("Current node index", current_node_index)
                    if (task_start_node == str(obstacle_edge_id[0]) and task_end_node == str(obstacle_edge_id[1])) or \
                            (task_start_node == str(obstacle_edge_id[1]) and task_end_node == str(obstacle_edge_id[0])):
                        obstacle_index = task_edge["sequenceId"]
                        print("obstacle index", obstacle_index)

                if current_node_index is not None and obstacle_index is not None:
                    if obstacle_index >= current_node_index:
                        affected_vehicles.append(vehicle_id)

        print("Affected vehicle:", affected_vehicles)

        # Recalculate routes for affected vehicles
        for vehicle_id in affected_vehicles:
            vehicle = self.vehicles[vehicle_id]
            vehicle["status"] = "idle"
            target_destination = vehicle["currentTask"]["edges"][-1]["endNodeId"]
            new_shortest_path = self.find_astar_path(self.graph, current_node, target_destination)
            print("New shortest path", new_shortest_path)
            # Update vehicle's current task with the new shortest path
            vehicle["currentTask"]["edges"] = new_shortest_path
            # Translate and send updated route to the vehicle
            updated_route = self.translate_path_to_mqtt(new_shortest_path)
            # self.send_route_to_vehicle_async(vehicle_id, updated_route)
            print("updated route: ", updated_route)

            threading.Thread(target=self.send_route_to_vehicle_async, args=(vehicle_id, updated_route)).start()

    def get_map(self):

        vehicle_colors = ["red", "green", "blue", "goldenrod", "magenta"]

        def get_map_plot():

            # Add weights to the edges_df
            with lock:
                for _, row in self.edge_df.iterrows():
                    self.edge_df.at[_, "length"] = self.graph[str(int(row["u"]))][str(int(row["v"]))]["length"]

            fig = go.Figure()

            # Add edges to the map
            lats = []
            lons = []
            for irow, row in self.edge_df.iterrows():
                lons.append(self.nodes_df.loc[row["u"]]["lon"])
                lons.append(self.nodes_df.loc[row["v"]]["lon"])
                lons.append(None)
                lats.append(self.nodes_df.loc[row["u"]]["lat"])
                lats.append(self.nodes_df.loc[row["v"]]["lat"])
                lats.append(None)
            fig.add_trace(go.Scattermapbox(mode='lines',
                                           lon=lons,
                                           lat=lats,
                                           line={'color': 'grey', 'width': 3},  # if on_routes else 3
                                           hoverlabel={'namelength': -1}
                                           ))

            # Add nodes to the map
            for index_node, row in self.nodes_df.iterrows():
                is_special_node = row["name"] is not np.NaN
                if is_special_node:  # currently only display nodes that are special nodes
                    color_node = "grey"
                    symbol_node = "circle"
                    is_special_node = row["name"] is not np.NaN
                    fig.add_trace(go.Scattermapbox(mode='markers',  # if is_special_node else 'markers' markers+text
                                                   lon=[row["lon"]],
                                                   lat=[row["lat"]],
                                                   marker={'color': color_node, 'size': 20 if is_special_node else 10,
                                                           'allowoverlap': False, 'symbol': symbol_node},
                                                   text=row["name"] if is_special_node else index_node,
                                                   name=row["name"] if is_special_node else index_node,
                                                   hoverinfo="text",
                                                   textposition='bottom center',
                                                   textfont=dict(size=20, color='black')
                                                   ))

            # Add vehicles to the map
            with lock:
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
            with lock:
                for vehicle_id, vehicle in self.vehicles.items():
                    if vehicle["currentTask"] is not None:
                        lons = []
                        lats = []
                        for edge in vehicle["currentTask"]["edges"]:
                            lons.append(self.nodes_df.loc[int(edge["startNodeId"])]["lon"])
                            lons.append(self.nodes_df.loc[int(edge["endNodeId"])]["lon"])
                            # lons.append(None)
                            lats.append(self.nodes_df.loc[int(edge["startNodeId"])]["lat"])
                            lats.append(self.nodes_df.loc[int(edge["endNodeId"])]["lat"])
                            # lats.append(None)
                        fig.add_trace(go.Scattermapbox(mode='lines',
                                                       lon=lons,
                                                       lat=lats,
                                                       line={'color': vehicle_colors[
                                                           int(vehicle_id) - 1 % len(vehicle_colors)], 'width': 5},
                                                       name=f"Route of vehicle {vehicle}",
                                                       hoverinfo='name',
                                                       hoverlabel={'namelength': -1}
                                                       ))

            fig.update_layout(mapbox_style="open-street-map",
                              mapbox_zoom=16,
                              mapbox_center_lat=48.00632,
                              mapbox_center_lon=7.838,
                              margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              width=1000,
                              height=850,
                              showlegend=False)

            fig['layout']['uirevision'] = 'currentZoom'

            return fig

        app = Dash(__name__)

        app.layout = html.Div([
            html.H1(children='Intelligent Hospital Logistics',
                    style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='live-update-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    )
                ], style={'padding': 5, 'flex': 1}),
                html.Div([
                    html.H2("LLM", style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554'}),
                    html.Div([
                        dcc.Dropdown(
                            id='llm-model-dropdown',
                            options=[
                                {'label': 'GPT-3.5', 'value': 'gpt'},
                                {'label': 'LLAMA-2', 'value': 'llama'},
                            ],
                            value='gpt',
                            style={'flex': 2, 'marginRight': '5px', 'borderRadius': '15px', 'padding': 5}
                        ),
                        dcc.Textarea(id='input-prompt', value='Prompt...',
                                     style={'height': 60, 'padding': 5, 'flex': 7, 'borderRadius': '15px',
                                            'marginRight': '5px'}),
                        html.Button('Submit', id='press-invoke-llm', n_clicks=0,
                                    style={'padding': 5, 'flex': 1, 'backgroundColor': '#99C554', 'border': 'none',
                                           'color': 'white', 'borderRadius': '15px'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'}),
                    html.Label(id='llm-output',
                               style={'whiteSpace': 'pre-line', 'padding': 5, 'backgroundColor': 'lightgrey',
                                      'font-family': 'Arial, sans-serif', 'display': 'none', 'flexGrow': 1,
                                      'minHeight': '40px', 'borderRadius': '15px', 'marginTop': '5px'}),
                    html.H2("Orders",
                            style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                   'marginTop': '60px'}),
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
                                                      'border': '1px solid grey'
                                                  },
                                                  )),
                    dcc.Interval(
                        id='table-update-interval',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    ),
                    html.H2("Vehicles",
                            style={'textAlign': 'left', 'font-family': 'Arial, sans-serif', 'color': '#99C554',
                                   'marginTop': '60px'}),
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
                                                  )),
                    dcc.Interval(
                        id='vehicle-table-update-interval',
                        interval=2 * 1000,  # in milliseconds
                        n_intervals=0
                    )
                ], style={'padding': 5, 'flex': 2})
            ], style={'display': 'flex', 'flexDirection': 'row'})
        ])

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
            if value_model == 'gpt':
                llm_output = Playground_LLM_Dacian.invoke_llm(value_prompt)
            else:
                llm_output = Playground_LLM_Dacian.invoke_llm(value_prompt)
            self.apply_llm_output(llm_output)
            return llm_output, {'whiteSpace': 'pre-line', 'padding': 5, 'backgroundColor': 'lightgrey',
                                'font-family': 'Arial, sans-serif', 'display': 'flex', 'flexGrow': 1,
                                'minHeight': '40px', 'borderRadius': '15px', 'marginTop': '5px'}

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

            return orders_list_copy

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
                        pass
                    vehicle["Vehicle"] = vehicle["vehicle_id"]
                    del vehicle["vehicle_id"]
                if 'timestamp' in vehicle:
                    del vehicle['timestamp']
                if 'headerId' in vehicle:
                    del vehicle['headerId']
                if 'currentTask' in vehicle:
                    del vehicle['currentTask']
                if 'position' in vehicle:
                    vehicle['position'] = f"({vehicle['position'][0]:.8f}, {vehicle['position'][1]:.8f})"

            return vehicles_list_copy

        app.run(debug=False)
