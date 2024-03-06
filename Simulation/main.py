from Simulation import BuildGraph
from Simulation import Routing

parameters = {
    "subgraph_params": {
        'special_nodes': ['Zwischen den Räumen', 'Stimme vom Berg',
                          'Emmaus Kapelle', 'Klinik für Frauenheilkunde [1]',
                          'Ernst von Bergmann'],
        'allowed_highway_types': ['footway', 'service']
    },
}


def main():
    G, edge_labels_highways, named_nodes = BuildGraph.build_nx_graph(
        parameters['subgraph_params']['allowed_highway_types'], parameters['subgraph_params']['special_nodes'])
    _ = Routing.Routing(G, edge_labels_highways, named_nodes)


if __name__ == "__main__":
    main()
