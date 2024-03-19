# from Simulation import BuildGraph
# from Simulation import Routing

import BuildGraph
import Routing

parameters = {
    "subgraph_params": {
        'special_nodes': ['Zwischen den Räumen', 'Stimme vom Berg',
                          'Emmaus Kapelle', 'Klinik für Frauenheilkunde [1]'],
        'allowed_highway_types': ['footway', 'service']
    },
}


def main():
    G, edge_df, nodes_df = BuildGraph.build_nx_graph(
        parameters['subgraph_params']['allowed_highway_types'], parameters['subgraph_params']['special_nodes'])
    _ = Routing.Routing(G, edge_df, nodes_df)


if __name__ == "__main__":
    main()
