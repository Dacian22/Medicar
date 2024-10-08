import datetime
import json
import os
import random
import threading
import time
import warnings

import paho.mqtt.client as mqtt
import paho.mqtt.client as paho
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")


class Vehicle:
    """
    This is a class for Vehicle. It gets commands from the simulation ("_2_Simulation") via MQTT and executes them.
    """

    vehicle_id = None  # String
    current_position = None  # tuple of floats (latitude, longitude)
    current_speed = 5 * 10e-6  # meter per second
    current_task = None
    client = None  # mqtt client
    status = "idle"  # one of "idle", "busy", "moving"
    target_node = None  # String
    currentSequenceId = None  # integer
    generate_incidents = False
    generate_incidents_interval = 6  # how often should incidents be generated (on average)
    generate_incidents_seed = None  # integer or none, if none, do not use predefined one. This is for drawing the incidents at random
    current_order_counter = 0
    current_do_task_thread = None

    def __init__(self, _vehicle_id, _current_position, _current_target_node):
        '''
        Constructor for the Vehicle class
        :param _vehicle_id: String
        :param _current_position: tuple of floats (latitude, longitude)
        :param _current_target_node: String
        '''
        load_dotenv()
        self.vehicle_id = _vehicle_id
        self.current_position = _current_position
        self.target_node = _current_target_node

        df = pd.read_csv(os.path.join(os.getenv("RESOURCES"), 'EvaluationDataset.csv'), delimiter=';')

        self.possible_incident_list = [test[0] for _, test in df.iterrows()]

    def on_connect(self, client, userdata, flags, rc, properties=None):
        '''
        Callback function for the MQTT client. It is called when the client connects to the MQTT broker.
        :param client: mqtt.Client
        :param userdata: any
        :param flags: dict
        :param rc: int
        :param properties: dict
        '''
        # test connection
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/route", qos=2)
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/random_seed",
                              qos=2)
        self.client.subscribe(
            os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/generate_incidents", qos=2)
        self.send_vehicle_status()
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/stop",
                              qos=2)
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/update_route",
                              qos=2)
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/cancel_route",
                              qos=2)
        if self.current_task is not None:
            self.current_do_task_thread = threading.Thread(target=self.dotask, args=[self.current_task])
            self.current_do_task_thread.start()

    def on_publish(self, client, userdata, mid, reason_code, properties=None):
        '''
        Callback function for the MQTT client. It is called when the client publishes a message to the MQTT broker.
        :param client: mqtt.Client
        :param userdata: any
        :param mid: int
        :param reason_code: int
        :param properties: dict
        '''
        pass

    def on_subscribe(self, client, userdata, mid, reason_code_list, properties, granted_qos=None):
        '''
        Callback function for the MQTT client. It is called when the client subscribes to a topic on the MQTT broker.
        :param client: mqtt.Client
        :param userdata: any
        :param mid: int
        :param reason_code_list: list
        :param properties: dict
        :param granted_qos: list
        '''
        pass

    def on_message(self, client, userdata, msg):
        '''
        Callback function for the MQTT client. It is called when the client receives a message from the MQTT broker.
        :param client: mqtt.Client
        :param userdata: any
        :param msg: mqtt.MQTTMessage
        '''
        threading.Thread(target=self.message_worker, args=[msg]).start()

    def message_worker(self, msg):
        '''
        Worker function for the on_message callback function. It processes the received message.
        :param msg: mqtt.MQTTMessage
        '''
        if msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/route":
            self.receive_route(msg.payload.decode("utf-8"))
        elif msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/random_seed":
            self.generate_incidents_seed = int(msg.payload.decode("utf-8"))
            random.seed(self.generate_incidents_seed)
        elif msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/generate_incidents":
            received_value = msg.payload.decode("utf-8")
            if received_value == "off":
                self.generate_incidents = False
            elif received_value == "on":
                self.generate_incidents = True
            else:
                print("ERROR in generate incidents message: " + received_value)
        elif msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/stop":
            self.send_stop()
        elif msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/update_route":
            self.receive_route(msg.payload.decode("utf-8"))
        elif msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/cancel_route":
            if self.status == "moving":
                self.send_stop()
            self.current_task = None
            self.status = "idle"
            self.currentSequenceId = None
            self.send_vehicle_status()
        else:
            print("ERROR: Received unsupported message: " + msg.topic)

    def send_stop(self):
        '''
        Sends a stop command to the vehicle.
        '''
        if self.status == "moving":
            print(f"Vehicle {self.vehicle_id}: Received stop command")
            self.status = "stopping"
            while self.current_task is not None and self.current_do_task_thread.is_alive():
                time.sleep(0.1)

            print(f"Vehicle {self.vehicle_id}: Executed stop command")
            self.send_vehicle_status()
        else:
            print(f"Vehicle {self.vehicle_id}: ERROR: Received stop command while not moving")

    def send_vehicle_status(self):
        '''
        Sends the current status of the vehicle to the simulation.
        '''
        payload = dict()
        # VDA 5050 standard properties
        payload["headerId"] = "NAN"
        payload["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["vehicleId"] = self.vehicle_id
        payload["position"] = self.current_position
        payload["speeed"] = self.current_speed

        # custom properties
        payload["status"] = self.status
        payload["targetNode"] = self.target_node
        payload["currentTask"] = self.current_task
        payload["currentSequenceId"] = self.currentSequenceId

        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/status",
                            json.dumps(payload), qos=0)

    def send_incident(self, incident_prompt, edgeId):
        '''
        Sends an incident to the simulation.
        :param incident_prompt: String
        :param edgeId: String
        '''
        payload = dict()
        payload["prompt"] = incident_prompt
        payload["edgeId"] = edgeId
        payload["vehicleId"] = self.vehicle_id
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/incident",
                            json.dumps(payload), qos=2)

    def receive_route(self, message):
        '''
        Receives a route from the simulation.
        :param message: String
        '''
        payload = json.loads(message)
        self.current_order_counter += 1

        if self.status != 'idle':  # cancel current task
            print(f"RECEIVED UPDATE TO ROUTE! Vehicle {self.vehicle_id}")

        while self.current_task is not None and self.current_do_task_thread.is_alive():
            self.status = "stopping"
            print(f"ERROR Vehicle {self.vehicle_id}: Received new task while still busy on the old one")
            time.sleep(0.1)

        self.current_task = payload
        self.send_vehicle_status()
        self.current_do_task_thread = threading.Thread(target=self.dotask, args=[payload])
        self.current_do_task_thread.start()

    def connect_to_mqtt(self):
        '''
        Connects the vehicle to the MQTT broker.
        '''
        # Connect to MQTT
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=paho.MQTTv5)
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

        # enable TLS for secure connection
        self.client.tls_set()  # tls_version=mqtt.client.ssl.PROTOCOL_TLS
        self.client.tls_insecure_set(True)

        # Set queue size
        self.client.max_inflight_messages_set(20)
        self.client.max_queued_messages_set(100_000)

        # set username and password
        self.client.username_pw_set(os.getenv("HYVE_MQTT_USR"), os.getenv("HYVE_MQTT_PWD"))
        # connect to HiveMQ Cloud on port 8883 (default for MQTT)
        self.client.connect(os.getenv("HYVE_MQTT_URL"), 8883)

        self.client.loop_forever()  # loop start (if constantly sending status)

    def get_linear_function_for_edge(self, edge):
        '''
        Calculates the linear function for an edge.
        :param edge: dict
        :return: tuple of floats (slope, y-intercept)
        '''
        divider = (float(edge["endCoordinate"][0]) - float(edge["startCoordinate"][0]))
        if divider == 0:  # edge is vertical
            divider = 0.0000001
        m = (float(edge["endCoordinate"][1]) - float(edge["startCoordinate"][1])) / divider  # slope
        n = float(edge["startCoordinate"][1]) - m * float(edge["startCoordinate"][0])  # y-intercept
        if m == 0:  # edge is vertical
            m = 0.0000001
        return m, n

    def move_along_edge(self, edge):
        '''
        Moves the vehicle along an edge.
        :param edge: dict
        '''
        edge["startCoordinate"][0] = float(edge["startCoordinate"][0])
        edge["startCoordinate"][1] = float(edge["startCoordinate"][1])
        edge["endCoordinate"][0] = float(edge["endCoordinate"][0])
        edge["endCoordinate"][1] = float(edge["endCoordinate"][1])
        negative_x = edge["startCoordinate"][0] > edge["endCoordinate"][0]
        m, n = self.get_linear_function_for_edge(edge)
        # calculated adjusted increase needed for x axis to match the speed
        x_increase = abs(self.current_speed / m)

        while True:
            end = False
            # calculate new position for positive slope
            if not negative_x:
                new_x = self.current_position[0] + x_increase
                new_y = m * new_x + n
            else:  # calculate new position for negative_x slope
                new_x = self.current_position[0] - x_increase
                new_y = m * new_x + n

            # check if new position is larger than the target position
            if negative_x and new_x < edge["endCoordinate"][0] or not negative_x and new_x > edge["endCoordinate"][0]:
                new_x = edge["endCoordinate"][0]
                new_y = edge["endCoordinate"][1]
                end = True
            self.current_position = [new_x, new_y]
            self.send_vehicle_status()
            time.sleep(1)
            if end:
                break

    def dotask(self, task):
        '''
        Executes a task.
        :param task: dict
        '''
        print(task)
        if task["edges"] is None or len(task["edges"]) == 0:  # edges are empty => do not move
            print(f"Vehicle {self.vehicle_id}: ERROR: Received task with no edges")
            return
        else:  # work on the task
            self.status = "moving"
            self.target_node = task["edges"][-1]["endNodeId"]
            edges = task["edges"]
            for edge in edges:
                self.currentSequenceId = edge["sequenceId"]
                self.move_along_edge(edge)
                if self.generate_incidents and random.randint(0, self.generate_incidents_interval) == 0:
                    threading.Thread(target=self.send_incident,
                                     args=[random.choice(self.possible_incident_list), edge["edgeId"]]).start()
                if self.status == "stopping":
                    self.status = "stopped"
                    self.target_node = edge["endNodeId"]
                    self.send_vehicle_status()
                    return
        self.status = "idle"
        self.currentSequenceId = None
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/order_finish",
                            payload=json.dumps({"orderId": task["orderId"]}), qos=2)
        self.current_task = None
        self.send_vehicle_status()
