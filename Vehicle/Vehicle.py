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
    current_position = None  # NetworkXNode or NetworkXEdge as string
    current_speed = 60  # nodes per minute
    current_task = None
    client = None
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
        if msg.topic == "vehicles/" + self.vehicle_id + "/newtask":
            if self.current_task is None:
                print("Received new task: " + msg.payload.decode("utf-8"))
                self.dotask(msg.payload.decode("utf-8"))
            else:
                print("Vehicle is busy")
                client.publish("vehicles/" + self.vehicle_id + "/status", "busy", qos=2)
        elif msg.topic == "vehicles/" + self.vehicle_id + "/canceltask":
            print("Received cancel task")
            self.canceltask()

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
        self.client.publish("vehicles/" + self.vehicle_id + "/status", "online", qos=2)
        self.client.subscribe("vehicles/" + self.vehicle_id + "/newtask", qos=2)
        self.client.subscribe("vehicles/" + self.vehicle_id + "/hello", qos=2)
        print("start")
        self.client.loop_forever()

    def dotask(self, route):
        self.current_task = route.split(",")
        print(self.current_task)
        for step in self.current_task:
            time.sleep(60 / self.current_speed)
            self.current_position = step
            self.client.publish("vehicles/" + self.vehicle_id + "/position", self.current_position, qos=2)
            print("newposition: " + self.current_position)
        self.current_task = None
        self.client.publish("vehicles/" + self.vehicle_id + "/status", "idle", qos=2)

    def canceltask(self):
        pass