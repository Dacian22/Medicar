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
        
    
    def extract_order(self, heuristic):
        origin = re.search(r'from\s+(.*?)\s+to', heuristic, re.IGNORECASE).group(1)
        destination = re.search(r'to\s+(.*?)\s+every', heuristic, re.IGNORECASE).group(1)
        objects = re.search(r'Transport\s+(.*?)\s+from', heuristic, re.IGNORECASE).group(1).split(',')
        order_interval = re.search(r'every\s+(\d+)\s+(\w+)', heuristic, re.IGNORECASE)
        interval = int(order_interval.group(1))
        unit = order_interval.group(2)
        if unit == 'min':
           interval *= 60
        print(origin, destination, objects, interval)
        return origin, destination, objects, interval
   
    def create_orders_from_heuristics(self):
        origin, destination, objects, interval = self.extract_order(self.heuristic)
        #create order dictionary
        order = {
            "from_location": origin,
            "to_location": destination,
            "objects": objects,
            "interval": interval
        }
        self.order_interval = interval
        self.origin = origin
        self.destination = destination
        self.objects = objects
        self.order = order
        return self.order
    
    
    # Convert the Order instance to a dictionary
    def to_dict(self):
       return {
           "order_id": self.order_id,
           "timestamp": self.timestamp,
           "origin": self.origin,
           "destination": self.destination,
           "objects": self.objects,
           "order_interval": self.order_interval
       }
                  




