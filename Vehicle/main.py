from Vehicle import Vehicle
import multiprocessing

num_vehicles = 2

def start_vehicle(vehicle):
    vehicle.connect_to_mqtt()

if __name__ == '__main__':
    # Create num_vehicles vehicles
    vehicles = [Vehicle(str(i)) for i in range(num_vehicles)]

    # Start them all using multiple Processes

    # Spin up vehicles (multiple threads)
    with multiprocessing.Pool(num_vehicles) as ex:
        _ = ex.map(start_vehicle, vehicles) # never returns
