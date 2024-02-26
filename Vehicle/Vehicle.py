import os

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
    current_task = None  # Task as string
    moving = False

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

    def connect_to_mqtt(self):
        # Connect to MQTT
        client = paho.Client(client_id=self.vehicle_id, userdata=None, protocol=paho.MQTTv5)
        client.on_connect = self.on_connect
        client.on_publish = self.on_publish
        client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message

        # enable TLS for secure connection
        client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
        # set username and password
        client.username_pw_set(os.getenv("HYVE_MQTT_USR"), os.getenv("HYVE_MQTT_PWD"))
        # connect to HiveMQ Cloud on port 8883 (default for MQTT)
        client.connect(os.getenv("HYVE_MQTT_URL"), 8883)
        # test connection
        client.subscribe("vehicles/" + self.vehicle_id + "/commands", qos=2)
        client.publish("vehicles/" + self.vehicle_id + "/status", "online", qos=2)
        print(client.is_connected())
        client.loop_forever()
