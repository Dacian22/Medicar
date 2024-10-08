import os

from dotenv import load_dotenv

from order_manager import OrderManager

load_dotenv()


def main():
    """
    Main entry point of the application that sets up and starts the `OrderManager` instance.

    This function:
        1. Retrieves configuration values for the MQTT broker URL, username, and password from environment variables.
        2. Constructs the path to the heuristics file by joining the `RESOURCES` environment variable with the filename "orders.csv".
        3. Creates an instance of the `OrderManager` class with the obtained configuration values and heuristics file path.
        4. Calls the `process_heuristics` method on the `OrderManager` instance to start processing the heuristics.

    Environment Variables:
        - HYVE_MQTT_URL: The URL of the MQTT broker.
        - HYVE_MQTT_USR: The username for MQTT authentication.
        - HYVE_MQTT_PWD: The password for MQTT authentication.
        - RESOURCES: The directory path where the heuristics file is located.
    """

    mqtt_broker_url = os.getenv("HYVE_MQTT_URL")
    mqtt_username = os.getenv("HYVE_MQTT_USR")
    mqtt_password = os.getenv("HYVE_MQTT_PWD")
    heuristics_file = os.path.join(os.getenv("RESOURCES"), "orders.csv")

    _ = OrderManager(mqtt_broker_url, mqtt_username, mqtt_password, heuristics_file)


if __name__ == "__main__":
    main()
