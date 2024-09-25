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
        """
        Initializes an MQTT client, connects to the broker, and subscribes to vehicle status topics.
        Also loads heuristics from a provided file and initializes idle vehicle tracking.

        Args:
            mqtt_broker_url (str): The URL or IP address of the MQTT broker.
            mqtt_username (str): The username for MQTT broker authentication.
            mqtt_password (str): The password for MQTT broker authentication.
            heuristics_file (str): Path to the file containing heuristics data.

        Attributes:
            client (mqtt.Client): The MQTT client used for communication with the broker.
            heuristics_file (str): The file path of the heuristics.
            heuristics (dict): The loaded heuristics data from the file.
            idle_vehicles (list): A list to track idle vehicles.
        """

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=paho.MQTTv5)
        self.client.on_connect = self.on_connect
        self.client.tls_set()  
        self.client.username_pw_set(mqtt_username, mqtt_password)
        self.client.connect(mqtt_broker_url, 8883)
        self.client.subscribe(os.getenv("MQTT_PREFIX_TOPIC") + "/" + "vehicles/+/status", qos=2)

        self.heuristics_file = heuristics_file
        self.heuristics = self.load_heuristics()

       
        self.idle_vehicles = []
        self.client.loop_start()

   
    def on_connect(self, client, userdata, flags, rc, properties=None):
        """
        Callback function that is triggered when the MQTT client successfully connects to the broker.

        Args:
            client (mqtt.Client): The MQTT client instance.
            userdata (Any): The user data (unused in this case).
            flags (dict): Response flags sent by the broker.
            rc (int): The result code of the connection attempt. A result code of 0 indicates a successful connection.
            properties (mqtt.Properties, optional): MQTT v5 properties (unused in this case).

        Side Effect:
            Prints the result code of the connection to the console.
        """
        print("Connected with result code " + str(rc))

    
    def on_publish(self, client, userdata, mid):
        pass

    def on_message(self, client, userdata, message):
        pass

    def on_subscribe(self, client, userdata, mid, granted_qos, properties):
        pass

    def load_heuristics(self):
        """
        Loads heuristic data from a CSV file and returns it as a list of rows.

        The CSV file is expected to have a header row, which is skipped. Each subsequent row is read and
        appended to a list.

        Returns:
            list: A list of rows, where each row is a list of values from the CSV file.
        """
        heuristics = []
        with open(self.heuristics_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                heuristics.append(row)
        return heuristics
    
    
    def process_heuristics(self):
        """
        Processes each heuristic in the loaded heuristics list by creating and sending orders.

        For each heuristic, an `Order` object is instantiated, and the order is created using the `create_order` method.
        The order is then sent using the `send_order` method, followed by a delay between orders.

        The method also starts the MQTT client loop to continue processing messages after sending the orders.

        Attributes:
            self.heuristics (list): The list of heuristics loaded from a file.
            self.time_between_orders (float or int): The time interval in seconds to wait between processing orders.
        """

        for idx, heuristic in enumerate(self.heuristics):
            order_instance = Order(heuristic, idx)
            order_instance.create_order()

            self.send_order(order_instance)
            time.sleep(self.time_between_orders)

        self.client.loop_start()
    
    
    def send_order(self, order):
        """
        Sends the order to the specified MQTT topic in JSON format.

        The order is first converted to a dictionary using the `to_dict` method, then serialized into a JSON string.
        The order is published to an MQTT topic using the MQTT client, with a QoS level of 2 to ensure message delivery.

        Args:
            order (Order): The `Order` object to be sent, which contains order details like source, target, items, and vehicle ID.

        Attributes:
            self.client (mqtt.Client): The MQTT client used for publishing messages.

        Side Effect:
            Prints the JSON representation of the sent order.

        """

        order_dict = order.to_dict()
        order_json = json.dumps(order_dict)
        
        topic = f"order_manager/transportation/orders/{order.order_id}"
        self.client.publish(os.getenv("MQTT_PREFIX_TOPIC") + "/" + topic, order_json, qos=2)
        print("Order sent:", order_json)



    def closest_vehicle_callback(self, client, userdata, message):
        """
        Callback function triggered when a message regarding the closest vehicle is received.

        This function decodes the incoming MQTT message payload, which is expected to be in JSON format,
        and extracts the `vehicle_id` of the closest vehicle.

        Args:
            client (mqtt.Client): The MQTT client instance that received the message.
            userdata (Any): The user data passed to the callback (unused in this case).
            message (mqtt.Message): The message object containing the topic, payload, and other metadata.

        Returns:
            vehicle_id (str): The ID of the closest vehicle extracted from the message payload.

        """

        vehicle_id = json.loads(message.payload.decode())
        return vehicle_id
