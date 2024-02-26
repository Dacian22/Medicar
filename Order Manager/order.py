import re


class Order:
    
    
    def __init__(self,heuristic, order_id_counter):
        
        # read heuristics from the provided file
        self.heuristic = heuristic
        
        #declare class attributes
        self.order_id_counter = order_id_counter
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
        # increment order ID counter for the next order
        self.order_id_counter += 1
        self.order_interval = interval
        self.origin = origin
        self.destination = destination
        self.objects = objects
        self.order = order
        return self.order
    
    
    # Convert the Order instance to a dictionary
    def to_dict(self):
       return {
           "order_id_counter": self.order_id_counter,
           "origin": self.origin,
           "destination": self.destination,
           "objects": self.objects,
           "order_interval": self.order_interval
       }
                  




