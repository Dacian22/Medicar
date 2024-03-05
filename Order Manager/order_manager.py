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

        # subscribe to topic to get the status of the vehicles
        self.client.subscribe("vehicles/+/status", qos=2)
        # get the closest vehicle from the message sent by simulation
        self.client.message_callback_add("simulation/closest_vehicle", self.closest_vehicle_callback)
        
        self.heuristics_file = heuristics_file
        # load the heuristics file
        self.heuristics = self.load_heuristics()

        # store the list of idle vehicles
        self.idle_vehicles = []

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

    def on_publish(self, client, userdata, mid):
        print("Message published")

    def on_message(self, client, userdata, message):
       # get the message sent by the topic and decode it
       topic = message.topic
       payload = json.loads(message.payload.decode())

       # create a dictionary that contains the status for every vehicle id
       if topic.startswith("vehicles/") and topic.endswith("/status"):
            vehicle_id = topic.split("/")[1]
            vehicle_status = payload["status"]
            self.update_vehicle_status(vehicle_id, vehicle_status)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed to topic with QoS:", granted_qos)
        
    def load_heuristics(self):
        heuristics = []
        with open(self.heuristics_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) #skip the first row
            for row in reader:
                #divide the heuristics
                heuristics.append(row[0]) 
        return heuristics
    
    def process_heuristics(self):

        for idx, heuristic in enumerate(self.heuristics):
            # initialize OrderManager with the current heuristic
            order_instance = Order(heuristic)
        
            # create orders from the current heuristic
            order_instance.create_order()

            # assign a vehicle to the order
            self.assign_vehicle(order_instance)

            #get the vehicle id from simulation
            try:
                json.dumps(self.closest_vehicle_callback)
                order_instance.vehicle_id = self.closest_vehicle_callback
            except TypeError:
                order_instance.vehicle_id = 1  #temporary

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
        self.client.publish(topic, order_json, qos=2)
        print("Order sent:", order_json)

    def send_order_periodically(self, order_instance):
      while True:
         interval = order_instance.order_interval
         # send order only if the order interval has elapsed
         if interval > 0:
                self.send_order(order_instance)
                # sleep for the interval before sending the next order
                time.sleep(interval)
    
    def update_vehicle_status(self, vehicle_id, status):
       """
       Update the status of the given vehicle in the list of idle vehicles.
       If the vehicle is now idle and was not previously, add it to the list.
       If the vehicle is no longer idle and was previously, remove it from the list.
       """
       if status == "idle" and vehicle_id not in self.idle_vehicles:
           self.idle_vehicles.append(vehicle_id)
       elif status != "idle" and vehicle_id in self.idle_vehicles:
           self.idle_vehicles.remove(vehicle_id)


    # require from the simulation which vehicle is the closest to the order source
    def assign_vehicle(self, order):
        payload = {
            "order_id": order.order_id,
            "source": order.source,
            "idle_vehicles": self.idle_vehicles
        }
        self.client.publish("simulation/get_closest_vehicle", json.dumps(payload), qos=2)
    
    #assign the vehicle id to current order when a message with the closest vehicle
    # is received from simulation
    def closest_vehicle_callback(self, client, userdata, message):
        vehicle_id = json.loads(message.payload.decode())
        return vehicle_id

    

