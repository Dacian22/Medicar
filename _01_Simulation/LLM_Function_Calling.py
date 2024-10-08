import networkx as nx
import openai
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import BuildGraph
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Build the graph from the special nodes
G, edge_df, nodes_df = BuildGraph.build_nx_graph(
        parameters['subgraph_params']['allowed_highway_types'],
        parameters['subgraph_params']['allowed_surface_types'],
        parameters['subgraph_params']['special_nodes'])


def get_neighbor_nodes(given_node):
    """Return the neighbors of the specified node in the graph."""
    
    if given_node in G:
        impacted_nodes = list(G.neighbors(given_node))
        impacted_nodes = impacted_nodes.append(given_node)
        neighbor_nodes = {
            "neighbor_nodes": list(G.neighbors(given_node)),
        }
    else:
        neighbor_nodes = {
            "neighbor_nodes": "",
        }

    return json.dumps(neighbor_nodes)

function_descriptions = [
    {
        "name": "get_neighbor_nodes",
        "descripton": "Get the impacted nodes from a specific node in a graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "given_node": {
                    "type": "string",
                    "description": "A specific node in a graph. ",
                },
            },
        },
    }
]

user_prompt = input("Please enter your prompt: ")
context = """You are a graph expert and you are given the graph of a university hospital
            campus. Nodes are the buildings in the graph and edges are the routes between
            the buidlings. You will be given some information that something is happening 
            at a specific node. You need to determine if what is happening impacts other buildings in 
            the graph. The event impacts other buildings if it causes people crowds outside the buildings.
            If not, answer that the only impacted node is the given node. 
            Otherwise, give the impacted nodes including the given node.
            Give a definite answer."""

full_prompt = f"{context} \n {user_prompt}"

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# Get the original response from the model
response1 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [{'role': 'user', 'content': full_prompt}],
    max_tokens=300,
    temperature=0,
)

output = response1.choices[0].message

response_content = output.content.strip().lower()

# Check if other nodes are impacted as well
if "only impacted node " not in response_content:
    
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [{'role': 'user', 'content': full_prompt}],
        max_tokens=300,
        functions=function_descriptions,
        function_call="auto",
    )


output = response1.choices[0].message

# Check if the model has called the function
if output.function_call:
    params = json.loads(output.function_call.arguments)
    chosen_function = eval(output.function_call.name)
    answer = chosen_function(**params)


    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [{'role': 'user', 'content': full_prompt},
                    {'role': "function", "name": output.function_call.name, "content": answer},
        ],
        max_tokens=300,
        functions=function_descriptions,
    )

    print(response2.choices[0].message.content)

else:
    print(output.content)

