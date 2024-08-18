import multiprocessing

from Vehicle import Vehicle

num_vehicles = 5

startNodes = [1854588738, 5910155409,
              7108797192, 388528852,
              5861976268]

startCoordinates = [[48.007039,7.835593],[48.004773,7.833172],
                    [48.005305,7.837666], [48.006648,7.838115],
                    [48.004689,7.840123]]
              
            
def start_vehicle(vehicle):
    vehicle.connect_to_mqtt()


def main():
    # Create num_vehicles vehicles
    vehicles = [Vehicle(str(i), startCoordinates[i % len(startCoordinates)], startNodes[i % len(startNodes)]) for i in range(1, num_vehicles + 1)]

    # Spin up vehicles (multiple threads)
    with multiprocessing.Pool(num_vehicles) as ex:
        _ = ex.map(start_vehicle, vehicles)  # never returns


if __name__ == '__main__':
    main()