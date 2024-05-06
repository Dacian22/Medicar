import multiprocessing

from Vehicle import Vehicle

num_vehicles = 3

startNodes = [7112443020, 6185709507, 388528852, 7047823175, 9954286428, 1280377548,1854588738,
              3350449602,5327549888, 7880503849, 7158216520,7152639585, 5910155409, 9622545344,
               7191650493, 7126868397, 5327549890,2401035988, 7113069806, 7952832724, 7112443021,
                   3378944507,3436131035, 7112443018, 7126868392]
startCoordinates = [[48.005975,7.837238],
                    [48.006860,7.837142],
                    [48.006648,7.838115],
                    [48.005446,7.838718],
                    [48.004394,7.832625],
                    [48.005588,7.839469],
                    [48.007039,7.835593],
                    [48.005814,7.839474],
                    [48.008281,7.836216],
                    [48.004892,7.834016],
                    [48.004146,7.837532],
                    [48.007047,7.840072],
                    [48.004773,7.833172],
                    [48.007369,7.837768],
                    [48.005474,7.839320],
                    [48.007905,7.838185],
                    [48.008358,7.836480],
                    [48.005344,7.835426],
                    [48.007364,7.838701],
                    [48.004000,7.839739],
                    [48.006722,7.837288],
                    [48.004835,7.839897],
                    [48.005425,7.837624],
                    [48.005461,7.835026],
                    [48.005210,7.842642]

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