import datetime
import json
import os
import time

import paho.mqtt.client as paho
from dotenv import load_dotenv
from paho import mqtt

import threading


class Vehicle:
    """
    This is a class for Vehicle. It gets commands from the simulation ("_2_Simulation") via MQTT and executes them.
    """

    vehicle_id = None  # String
    current_position = [48.00686, 7.8371425]  # tuple of floats (latitude, longitude)
    current_speed = 5 * 10e-6  # meter per second
    current_task = None
    client = None  # mqtt client
    status = None  # one of "idle", "busy", "moving"
    target_node = None  # String

    def __init__(self, _vehicle_id):
        load_dotenv()
        self.vehicle_id = _vehicle_id

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("CONNACK received with code %s." % rc)

    def on_publish(self, client, userdata, mid, properties=None):
        print("mid: " + str(mid))

    def on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
        if msg.topic == os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/route":
            print("Received new task: " + msg.payload.decode("utf-8"))
            # self.receive_route(msg.payload.decode("utf-8"))
            threading.Thread(target=self.receive_route, args=(msg.payload.decode("utf-8"),)).start()
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

        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/" + self.vehicle_id + "/status", json.dumps(payload), qos=2)

    def send_incident(self, incident):
        # TODO not implemented
        pass

    def receive_route(self, message):
        print(message)
        payload = json.loads(message)

        if self.current_task is not None:
            print("Vehicle is busy")
            self.send_vehicle_status()
            return

        self.current_task = payload
        self.status = "busy"
        self.send_vehicle_status()
        self.dotask()

    def connect_to_mqtt(self):
        # Connect to MQTT
        self.client = paho.Client(client_id=self.vehicle_id, userdata=None, protocol=paho.MQTTv5)
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
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" +"vehicles/" + self.vehicle_id + "/route", qos=2)
        print("start")
        self.status = "idle"
        self.send_vehicle_status()
        self.client.loop_forever() # loop start (if constantly sending status)

    def get_linear_function_for_edge(self, edge):
        m = (float(edge["endCoordinate"][1]) - float(edge["startCoordinate"][1])) / (float(edge["endCoordinate"][0]) - float(edge["startCoordinate"][0]))  # slope
        n = float(edge["startCoordinate"][1]) - m * float(edge["startCoordinate"][0])  # y-intercept
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
                print("reached end of edge")
                new_x = edge["endCoordinate"][0]
                new_y = edge["endCoordinate"][1]
                end = True
            self.current_position = [new_x, new_y]
            print("newposition: " + str(self.current_position))
            self.send_vehicle_status()
            time.sleep(1)  # sleep for 1 second before next call
            if end:
                break

    def dotask(self):
        print(self.current_task)
        self.status = "moving"
        self.target_node = self.current_task["edges"][-1]["endNodeId"]
        if self.current_task["edges"] is None:  # edges are empty => do not move
            return
        else:  # work on the task
            edges = self.current_task["edges"]
            for edge in edges:
                self.move_along_edge(edge)
        self.status = "idle"
        self.current_task = None
        self.send_vehicle_status()
