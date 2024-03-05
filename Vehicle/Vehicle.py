import datetime
import json
import os
import time

import paho.mqtt.client as paho
from dotenv import load_dotenv
from paho import mqtt


class Vehicle:
    """
    This is a class for Vehicle. It gets commands from the simulation ("_2_Simulation") via MQTT and executes them.
    """

    vehicle_id = None  # String
    current_position = None  # nodeId or edgeId
    current_speed = 60  # nodes per minute
    current_task = None
    client = None
    status = None  # one of "idle", "busy", "moving"

    # moving = False

    def __init__(self, _vehicle_id):
        load_dotenv()
        self.vehicle_id = _vehicle_id
        self.connect_to_mqtt()

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("CONNACK received with code %s." % rc)

    def on_publish(self, client, userdata, mid, properties=None):
        print("mid: " + str(mid))

    def on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
        if msg.topic == "vehicles/" + self.vehicle_id + "/route":
            print("Received new task: " + msg.payload.decode("utf-8"))
            self.receive_route(msg.payload.decode("utf-8"))
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

        self.client.publish("vehicles/" + self.vehicle_id + "/status", json.dumps(payload), qos=2)

    def send_incident(self, incident):
        # TODO not implemented
        pass

    def receive_route(self, message):
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
        self.client.subscribe("vehicles/" + self.vehicle_id + "/route", qos=2)
        print("start")
        self.status = "idle"
        self.send_vehicle_status()
        self.client.loop_forever()

    def dotask(self):
        print(self.current_task)
        self.status = "moving"
        if self.current_task["edges"] is None: # edges are empty => do not move
            pass
        else:
            end_node_ids = [edge["endNodeId"] for edge in self.current_task["edges"]]
            for node in end_node_ids:
                time.sleep(60 / self.current_speed)
                self.current_position = node
                self.send_vehicle_status()
                print("newposition: " + json.dumps(self.current_position))
        self.status = "idle"
        self.current_task = None
        self.send_vehicle_status()
