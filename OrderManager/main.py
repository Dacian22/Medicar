import os

from dotenv import load_dotenv

from order_manager import OrderManager

# load environment variables from .env file
load_dotenv()


def main():
    mqtt_broker_url = os.getenv("HYVE_MQTT_URL")
    mqtt_username = os.getenv("HYVE_MQTT_USR")
    mqtt_password = os.getenv("HYVE_MQTT_PWD")
    heuristics_file = os.path.join(os.getenv("RESOURCES"), "orders.csv")

    order_manager = OrderManager(mqtt_broker_url, mqtt_username, mqtt_password, heuristics_file)
    # process heuristics concurrently
    order_manager.process_heuristics()


if __name__ == "__main__":
    main()
