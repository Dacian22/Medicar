import re
import time


class Order:
    order_id_dict = {}

    def __init__(self, heuristic, order_id):
        """
        Initialize an instance of the class with the provided heuristic and order ID.

        Args:
            heuristic (str): The heuristic or strategy used for processing the order.
            order_id (int or str): A unique identifier for the order.

        Attributes:
            heuristic (str): The strategy used for the order processing.
            order_id (int or str): The unique identifier for the order.
            timestamp (str): The timestamp when the object is created, in the format "YYYY-MM-DD HH:MM:SS".
            source (str): The source location of the order (default is an empty string).
            target (str): The target location of the order (default is an empty string).
            items (list): A list of items associated with the order (default is an empty list).
            order_interval (int): The time interval for the order processing (default is 0).
            vehicle_id (None or str): The ID of the vehicle assigned to the order (default is None).
        """

        self.heuristic = heuristic
        self.order_id = order_id
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.source = ""
        self.target = ""
        self.items = []
        self.order_interval = 0
        self.vehicle_id = None

    def extract_order(self):
        """
        Extracts and processes the order details from the heuristic attribute.

        The heuristic is expected to be a list-like structure containing:
            0: items (list): The list of items for the order.
            1: source (str): The source location of the order.
            2: target (str): The target location of the order.
            3: interval_str (str): The time interval for the order in the format "{value} {unit}",
                                where unit can be 'hours' or 'min'.
            4: vehicle_id (str): The ID of the vehicle assigned to the order.

        Returns:
            tuple: A tuple containing the following values:
                source (str): The source location.
                target (str): The target location.
                items (list): The list of items.
                value (int): The time interval for the order in minutes.
                vehicle_id (str): The vehicle ID.

        Raises:
            ValueError: If the interval string format is invalid or does not match the expected pattern.
        """

        components = self.heuristic
        items = components[0]
        source = components[1]
        target = components[2]
        interval_str = components[3]
        vehicle_id = components[4]
        match = re.match(r'(\d+)\s+(hours|min)', interval_str)

        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if unit == 'min':
                value *= 60  # convert minutes to hours
            return source, target, items, value, vehicle_id
        else:
            raise ValueError("Invalid interval format")

    def create_order(self):
        """
        Creates an order by extracting the necessary components from the heuristic and 
        assigning them to the corresponding attributes of the object.

        The method uses the `extract_order` method to retrieve the following details:
            - source (str): The source location of the order.
            - target (str): The target location of the order.
            - items (list): The list of items for the order.
            - interval (int): The time interval for the order in minutes.
            - vehicle_id (str): The ID of the vehicle assigned to the order.

        After extracting the data, it assigns these values to the object's attributes:
            - self.source
            - self.target
            - self.items
            - self.order_interval
            - self.vehicle_id
        """

        source, target, items, interval, vehicle_id = self.extract_order()
        self.source = source
        self.target = target
        self.items = items
        self.order_interval = interval
        self.vehicle_id = vehicle_id

    def to_dict(self):
        """
         Converts the current order object into a dictionary format, capturing relevant details.

         Returns:
             dict: A dictionary containing the following key-value pairs:
                 - "order_id" (str or int): The unique identifier for the order.
                 - "timestamp" (str): The current timestamp in the format "YYYY-MM-DD HH:MM:SS".
                 - "source" (str): The source location of the order.
                 - "target" (str): The target location of the order.
                 - "items" (list): The list of items for the order.
                 - "vehicle_id" (str): The ID of the vehicle assigned to the order (only included if not None).

         Raises:
             ValueError: If `self.vehicle_id` is None, the method raises an error as the vehicle ID is considered required.
         """

        order_dict = {
            "order_id": self.order_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source": self.source,
            "target": self.target,
            "items": self.items,
        }

        if self.vehicle_id is not None:
            order_dict["vehicle_id"] = self.vehicle_id
            return order_dict
        else:
            raise ValueError("Vehicle ID is None")
