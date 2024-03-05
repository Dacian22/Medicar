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
            Order.order_id_dict[heuristic] = len(Order.order_id_dict)+1

        # set the order ID for the current instance
        self.order_id = Order.order_id_dict[heuristic]
        
        #declare class attributes
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.source = ""
        self.target = ""
        self.items = []
        self.order_interval = 0
        self.vehicle_id  = None
    
   #extract the objects, origin, destination and interval of the order
    def extract_order(self):
        # split the CSV row into individual components
        components = self.heuristic.split(",")
        
        items = components[0]
        source = components[1]
        target = components[2]
        interval_str = components[3]
        match = re.match(r'(\d+)\s+(hours|min)', interval_str)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if unit == 'min':
                value *= 60  # convert minutes to hours
            return source, target, items, value
        else:
            raise ValueError("Invalid interval format")
    
    # create an order from the extracted information
    def create_order(self):
        source, target, items, interval = self.extract_order()
        self.source = source
        self.target = target
        self.items = items
        self.order_interval = interval
    
    
    # convert the Order instance to a dictionary
    def to_dict(self):
       order_dict = {
           "order_id": self.order_id,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
           "source": self.source,
           "target": self.target,
           "items": self.items,
       }
       #add the vehicle id to the order
       if self.vehicle_id is not None:
            order_dict["vehicle_id"] = self.vehicle_id
       else:
           order_dict["vehicle_id"] = 1 #assign 1 if the vehicle id is not give -- temporary
       return order_dict        




