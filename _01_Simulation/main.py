import BuildGraph
import Routing

import logging

parameters = {
    "subgraph_params": {
        'special_nodes': ['Emmaus Kapelle', 'Apotheke des Universitätsklinikums',
                 'Neurozentrum', 'Café am Ring', 'Die Andere Galerie', 'Augenklinik / HNO',
                 'Tonus', 'Neurozentrum', 'Café am Eck', 'Bistro am Lorenzring',
                 'Urologie', 'Luther Kindergarten', 'Kiosk Frauenklinik', 'Ernährungsmedizin',
                 'Medienzentrum', '3SAM Tagespflege', 'Klinik für Onkologische Rehabilitation',
                 'Stimme vom Berg', 'Klinik für Frauenheilkunde', 'Cafeteria im Casino',
                 'Sympathy', 'Die Himmelsleiter', 'Zwischen den Räumen',
                 'Terrakotta', 'Große Kugelkopfsäule', 'Freischwimmer', 'Notaufnahme', 'Gum',
                 'Tripylon', 'Notfallpraxis der niedergelassenen Ärzte', 'Klinik für Palliativmedizin',
                 'Lebensalter', 'Blutspende Freiburg', 'Christian Daniel Nussbaum','Das große Spiel',
                 'Hippokrates von Kos', 'Theodor Billroth',
                 'Adolf Lorenz', 'Universitätsklinikum Freiburg - Klinik für Innere Medizin'],
        'allowed_highway_types': ['footway', 'unclassified', 'service', 'platform',
                                  'steps', 'residential', 'construction', 'path', 'secondary_link',
                                  'tertiary', 'pedestrian', 'secondary', 'cycleway'],
        'allowed_surface_types': [None, 'grass_paver', 'paving_stones', 'asphalt', 'cobblestone', 'sett']},
}


def main():
    """Main function to build the graph and initialize routing"""
    
    G, edge_df, nodes_df = BuildGraph.build_nx_graph(
        parameters['subgraph_params']['allowed_highway_types'],
        parameters['subgraph_params']['allowed_surface_types'],
        parameters['subgraph_params']['special_nodes'])

    _ = Routing.Routing(G, edge_df, nodes_df)


if __name__ == "__main__":
    main()
