import multiprocessing

from Vehicle import Vehicle

num_vehicles = 3

def start_vehicle(vehicle):
    vehicle.connect_to_mqtt()


def main():
    # Create num_vehicles vehicles
    vehicles = [Vehicle(str(i)) for i in range(1, num_vehicles + 1)]

    # Start them all using multiple Processes

    # Spin up vehicles (multiple threads)
    with multiprocessing.Pool(num_vehicles) as ex:
        _ = ex.map(start_vehicle, vehicles)  # never returns


if __name__ == '__main__':
    main()