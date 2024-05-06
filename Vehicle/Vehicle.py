import datetime
import json
import os
import random
import time

import paho.mqtt.client as paho
from dotenv import load_dotenv

import threading

import pandas as pd
import paho.mqtt.client as mqtt


class Vehicle:
    """
    This is a class for Vehicle. It gets commands from the simulation ("_2_Simulation") via MQTT and executes them.
    """

    vehicle_id = None  # String
    current_position = None  # tuple of floats (latitude, longitude)
    current_speed = 5 * 10e-6  # meter per second
    current_task = None
    client = None  # mqtt client
    status = None  # one of "idle", "busy", "moving"
    target_node = None  # String
    currentSequenceId = None # integer
    generate_incidents = False
    generate_incidents_interval = 6  # how often should incidents be generated (on average)
    generate_incidents_seed = None # integer or none, if none, do not use predefined one. This is for drawing the incidents at random
    current_order_counter = 0
    current_do_task_thread = None

    def __init__(self, _vehicle_id, _current_position, _current_target_node):
        load_dotenv()
        self.vehicle_id = _vehicle_id
        self.current_position = _current_position
        self.target_node = _current_target_node

        df = pd.read_csv(os.path.join(os.getenv("RESOURCES"), 'EvaluationDataset.csv'), delimiter=';')

        self.possible_incident_list = [test[0] for _, test in df.iterrows()]

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("CONNACK received with code %s." % rc)
        # test connection
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/route", qos=2)
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/random_seed",
                              qos=2)
        self.client.subscribe(
            os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/generate_incidents", qos=2)
        self.status = "idle"
        self.send_vehicle_status()

    def on_publish(self, client, userdata, mid, reason_code, properties=None):
        # print("mid: " + str(mid))
        pass

    def on_subscribe(self, client, userdata, mid, reason_code_list, properties, granted_qos=None):
        # print("Subscribed: " + str(mid) + " " + str(granted_qos))
        pass

    def on_message(self, client, userdata, msg):
        threading.Thread(target=self.message_worker, args=[msg]).start()

    def message_worker(self, msg):
        # print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
        if msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/route":
            print("Received new task: " + msg.payload.decode("utf-8"))
            # self.receive_route(msg.payload.decode("utf-8"))
            self.receive_route(msg.payload.decode("utf-8"), )
        elif msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/random_seed":
            # print("Received new random seed: " + msg.payload.decode("utf-8"))
            self.generate_incidents_seed = int(msg.payload.decode("utf-8"))
        elif msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/generate_incidents":
            # print("Received new generate_incidents: " + msg.payload.decode("utf-8"))
            received_value = msg.payload.decode("utf-8")
            if received_value == "off":
                self.generate_incidents = False
            elif received_value == "on":
                self.generate_incidents = True
            else:
                print("ERROR in generate incidents message: " + received_value)
        else:
            print("ERROR: Received unsupported message: " + msg.topic)

    def send_vehicle_status(self):
        payload = dict()
        # VDA 5050 standard properties
        payload["headerId"] = "NAN"  # TODO: headerId not implemented (not VDA 5050 compliant)
        payload["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["vehicleId"] = self.vehicle_id
        payload["position"] = self.current_position  # TODO: format is not VDA 5050 compliant
        payload["speeed"] = self.current_speed

        # custom properties
        payload["status"] = self.status
        payload["targetNode"] = self.target_node
        payload["currentTask"] = self.current_task
        payload["currentSequenceId"] = self.currentSequenceId

        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/status", json.dumps(payload), qos=0)

    def send_incident(self, incident_prompt, edgeId):
        payload = dict()
        payload["prompt"] = incident_prompt
        payload["edgeId"] = edgeId
        payload["vehicleId"] = self.vehicle_id
        print("Sending incident: " + json.dumps(payload))
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/incident", json.dumps(payload), qos=2)

    def receive_route(self, message):
        payload = json.loads(message)
        self.current_order_counter += 1

        if self.status != 'idle': # cancle current task
            print(f"RECEIVED UPDATE TO ROUTE! Vehicle {self.vehicle_id}")

        self.status = "busy"

        while self.current_task is not None and self.current_do_task_thread.is_alive():
            print("Waiting for do task to terminate...")
            time.sleep(0.1)

        self.current_task = payload
        self.send_vehicle_status()
        self.current_do_task_thread = threading.Thread(target=self.dotask, args=[payload])
        self.current_do_task_thread.start()
        # self.dotask(self.current_order_counter)

    def connect_to_mqtt(self):
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

        self.client.loop_forever() # loop start (if constantly sending status)

    def get_linear_function_for_edge(self, edge):
        divider = (float(edge["endCoordinate"][0]) - float(edge["startCoordinate"][0]))
        if divider == 0: # edge is vertical
            divider = 0.0000001
        m = (float(edge["endCoordinate"][1]) - float(edge["startCoordinate"][1])) / divider  # slope
        n = float(edge["startCoordinate"][1]) - m * float(edge["startCoordinate"][0])  # y-intercept
        if m == 0:  # edge is vertical
            m = 0.0000001
        return m, n

    def move_along_edge(self, edge):
        edge["startCoordinate"][0] = float(edge["startCoordinate"][0])
        edge["startCoordinate"][1] = float(edge["startCoordinate"][1])
        edge["endCoordinate"][0] = float(edge["endCoordinate"][0])
        edge["endCoordinate"][1] = float(edge["endCoordinate"][1])
        negative_x = edge["startCoordinate"][0] > edge["endCoordinate"][0]
        m, n = self.get_linear_function_for_edge(edge)
        # calculated adjusted increase needed for x axis to match the speed
        x_increase = abs(self.current_speed / m)

        while True:
            end=False
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
            if self.status != "moving":
                return
            time.sleep(1)
            if end:
                break
            if self.status != "moving":
                return

    def dotask(self, task):
        print(task)
        if task["edges"] is None or len(task["edges"]) == 0:  # edges are empty => do not move
            print("No edges in task")
        else:  # work on the task
            self.status = "moving"
            self.target_node = task["edges"][-1]["endNodeId"]
            edges = task["edges"]
            for edge in edges:
                self.currentSequenceId = edge["sequenceId"]
                self.move_along_edge(edge)
                if self.generate_incidents and random.randint(0, self.generate_incidents_interval) == 0:
                    threading.Thread(target=self.send_incident, args=[random.choice(self.possible_incident_list), edge["edgeId"]]).start()
                if self.status != "moving":
                    return
        self.status = "idle"
        self.currentSequenceId = None
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/order_finish", payload=json.dumps({"orderId": task["orderId"]}) , qos=2)
        self.current_task = None
        self.send_vehicle_status()
