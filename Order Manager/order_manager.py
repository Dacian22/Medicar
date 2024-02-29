import paho.mqtt.client as mqtt
from order import Order
import threading
import time
import json
import csv


class OrderManager:
    
    
    def __init__(self, mqtt_broker_url, mqtt_username, mqtt_password,heuristics_file):
        # create a MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.on_message = self.on_message  # added on_message callback
        self.client.on_subscribe = self.on_subscribe  # added on_subscribe callback

        # enable TLS for secure connection
        self.client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)

        # set username and password
        self.client.username_pw_set(mqtt_username, mqtt_password)

        # connect the client to MQTT broker
        self.client.connect(mqtt_broker_url, 8883)

        # subscribe to topic
        self.client.subscribe("order_manager/transportation/orders", qos = 2)
        
        self.heuristics_file = heuristics_file
        self.heuristics = self.load_heuristics()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

    def on_publish(self, client, userdata, mid):
        print("Message published")

    def on_message(self, client, userdata, message):
        print("Received message:", message.payload.decode())

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed to topic with QoS:", granted_qos)
    
    def load_heuristics(self):
        heuristics = []
        with open(self.heuristics_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                heuristics.append(row[0])
        return heuristics

    
    def process_heuristics(self,heuristics):

        for idx, heuristic in enumerate(self.heuristics):
            # initialize OrderManager with the current heuristic
            order_instance = Order(heuristic)

            # create orders from the current heuristic
            order_instance.create_orders_from_heuristics()

            # start a new thread to send orders periodically for the current heuristic
            thread = threading.Thread(target=self.send_order_periodically,args=(order_instance,))
            thread.start()
    
    def send_order(self, order):
        # convert the order instance to a dictionary
        order_dict = order.to_dict()
        # convert the order dictionary to a JSON string
        order_json = json.dumps(order_dict)
        
        # publish the order to the MQTT broker
        topic = f"order_manager/transportation/orders/{order.order_id}"
        self.client.publish(topic, order_json, qos = 2)
        print(Order sent:", order_json)

    def send_order_periodically(self, order_instance):
      while True:
         interval = order_instance.order_interval
         # send order only if the order interval has elapsed
         if interval > 0:
                self.send_order(order_instance)
                # sleep for the interval sec before sending the next order
                time.sleep(interval)
