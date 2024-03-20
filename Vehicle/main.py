import multiprocessing

from Vehicle import Vehicle

num_vehicles = 3

startNodes = [7112443020, 6185709507,
                          388528852, 7047823175]
startCoordinates = [[48.005975,7.837238],
                    [48.006860,7.837142],
                    [48.006648,7.838115],
                    [48.005446,7.838718]
                    ]

def start_vehicle(vehicle):
    vehicle.connect_to_mqtt()


def main():
    # Create num_vehicles vehicles
    vehicles = [Vehicle(str(i), startCoordinates[i % len(startCoordinates)], startNodes[i % len(startNodes)]) for i in range(1, num_vehicles + 1)]

    # Start them all using multiple Processes

    # Spin up vehicles (multiple threads)
    with multiprocessing.Pool(num_vehicles) as ex:
        _ = ex.map(start_vehicle, vehicles)  # never returns


if __name__ == '__main__':
    main()