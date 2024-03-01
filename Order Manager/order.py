import re
import time

class Order:
    # dictionary to store order IDs for each heuristic
    order_id_dict = {}
    
    def __init__(self,heuristic):
        
        # read heuristics from the provided file
        self.heuristic = heuristic
        
        # initialize order ID for the current heuristic
        if heuristic not in Order.order_id_dict:
            Order.order_id_dict[heuristic] = 1

        # set the order ID for the current instance
        self.order_id = Order.order_id_dict[heuristic]
        
        #declare class attributes
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.origin = ""
        self.destination = ""
        self.objects = []
        self.order_interval = 0
        
    #extract the objects, origin, destination and interval of the order
    def extract_order(self, heuristic):
         # split the CSV row into individual components
        components = heuristic.split(",")
        
        objects = components[0]
        origin = components[1]
        destination = components[2]
        interval_str = components[3]
        match = re.match(r'(\d+)\s+(hours|min)', interval_str)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if unit == 'min':
                value *= 60  # convert minutes to hours
            return origin, destination, objects, value
        else:
            raise ValueError("Invalid interval format")

    #create an order from the extracted information
    def create_order(cls, heuristic):
        origin, destination, objects, interval = cls.extract_order(heuristic)
        order = cls(heuristic)
        order.origin = origin
        order.destination = destination
        order.objects = objects
        order.order_interval = interval
        return order
    
    
    # convert the Order instance to a dictionary
    def to_dict(self):
       return {
           "order_id": self.order_id,
           "timestamp": self.timestamp,
           "origin": self.origin,
           "destination": self.destination,
           "objects": self.objects,
           "order_interval": self.order_interval
       }
                  




