import paho.mqtt.client as paho
import paho.mqtt.client as mqtt
from order import Order
import threading
import time
import json
import csv

import os


class OrderManager:

    time_between_orders = 2  # seconds
    
    
    def __init__(self, mqtt_broker_url, mqtt_username, mqtt_password,heuristics_file):
        # create a MQTT client
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=paho.MQTTv5)
        self.client.on_connect = self.on_connect
        # self.client.on_publish = self.on_publish
        # self.client.on_message = self.on_message  # added on_message callback
        # self.client.on_subscribe = self.on_subscribe  # added on_subscribe callback

        # enable TLS for secure connection
        self.client.tls_set()  # tls_version=mqtt.client.ssl.PROTOCOL_TLS

        # set username and password
        self.client.username_pw_set(mqtt_username, mqtt_password)

        # connect the client to MQTT broker
        self.client.connect(mqtt_broker_url, 8883)

        # subscribe to topic to get the status of the vehicles
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/+/status", qos=2)

        self.heuristics_file = heuristics_file
        # load the heuristics file
        self.heuristics = self.load_heuristics()

        # store the list of idle vehicles
        self.idle_vehicles = []

        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("Connected with result code " + str(rc))

    # def on_publish(self, client, userdata, mid):
    #     # print("Message published")
    #     pass

    # def on_message(self, client, userdata, message):
    #    pass

    # def on_subscribe(self, client, userdata, mid, granted_qos, properties):
    #     # print("Subscribed to topic with QoS:", granted_qos)
    #     pass
        
    def load_heuristics(self):
        heuristics = []
        #open the heuristics file and read the rows
        with open(self.heuristics_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) #skip the first row
            for row in reader:
                #divide the heuristics
                heuristics.append(row)
        return heuristics
    
    def process_heuristics(self):

        for idx, heuristic in enumerate(self.heuristics):
            # initialize OrderManager with the current heuristic
            order_instance = Order(heuristic, idx)
        
            # create orders from the current heuristic
            order_instance.create_order()

            # send the order to the transportation manager
            self.send_order(order_instance)
            time.sleep(self.time_between_orders) # TODO very dirty hack

        self.client.loop_start()
    
    def send_order(self, order):
        # convert the order instance to a dictionary
        order_dict = order.to_dict()
        # convert the order dictionary to a JSON string
        order_json = json.dumps(order_dict)
        
        # publish the order to the MQTT broker
        topic = f"order_manager/transportation/orders/{order.order_id}"
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + topic, order_json, qos=2)
        print("Order sent:", order_json)

    #assign the vehicle id to current order when a message with the closest vehicle
    # is received from simulation
    def closest_vehicle_callback(self, client, userdata, message):
        vehicle_id = json.loads(message.payload.decode())
        return vehicle_id
