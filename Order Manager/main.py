from order_manager import OrderManager
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()


if __name__ == "__main__":
    mqtt_broker_url = os.getenv("HYVE_MQTT_URL")
    mqtt_username = os.getenv("HYVE_MQTT_USR")
    mqtt_password = os.getenv("HYVE_MQTT_PWD")
    heuristics_file = "heuristics.txt"
    
   
    
    with open(heuristics_file, 'r') as file:
        heuristics = file.readlines()
        
    order_manager = OrderManager(mqtt_broker_url, mqtt_username, mqtt_password,heuristics)
    # process heuristics concurrently
    order_manager.process_heuristics(heuristics)

