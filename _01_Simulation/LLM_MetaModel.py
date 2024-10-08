import json
import os
import warnings

from openai import OpenAI

import BuildGraph

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
                          'Lebensalter', 'Blutspende Freiburg', 'Christian Daniel Nussbaum', 'Das große Spiel',
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

method = None


def get_connected_edges(given_node):
    """
    Return the edges connected to the specified node in the graph.

    Args:
        given_node (str): The node for which connected edges are to be retrieved.

    Returns:
        str: A JSON string containing a list of edges connected to the given node.
              If the node does not exist in the graph, returns an empty list.
    """

    if given_node in G:
        connected_edges = list(G.edges(given_node))
        connected_edges = {
            "connected_edges": list(G.edges(given_node)),
        }
    else:
        connected_edges = {
            "connected_edges": "",
        }

    return json.dumps(connected_edges)


function_descriptions = [
    {
        "name": "get_connected_edges",
        "descripton": "Get the connected edges to a specific node in a graph.",
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


def invoke_llm(prompt):
    """
    Invokes the OpenAI API to process a prompt and determine the impact of events on the accessibility of edges in a graph.

    Args:
        prompt (str): The prompt containing information about an event affecting the graph.

    Returns:
        tuple: A tuple containing the results of various analyses based on the prompt:
            - output_usability (str): Determines if the event impacts transportation usability.
            - output_dynamic (str): Determines if the event impacts the whole length of an edge or just a part of it.
            - output_length (str): Provides a value between 0 and 100 indicating the extent of the impact on edge accessibility.
            - output_time (str): Provides a value in minutes indicating the time delay for vehicles passing through the edge.
            - output_nodes (str): Lists the edges impacted by an event occurring at a specific node.
            - output_nodes_time (str): Provides a time delay in minutes for vehicles due to events at specific nodes.
            - method (str): The method used for quantifying the impact, either "factor" for accessibility or "minutes" for time delay.
    """

    output_usability = None
    output_dynamic = None
    output_length = None
    output_time = None
    output_nodes = None
    output_nodes_time = None

    context_usability = f"""You are a graph expert and you are given the graph of a university hospital
                    campus. Nodes are the buildings in the graph and edges are the routes between
                    the buildings.  You will be given some information that something is happening 
                    at a specific node or edge. You need to determine if what is happening will have an
                    impact in transporting goods through edges. 
                    Only take into consideration transportation outside the buildings and not within buildings. 
                    You can rely on the given examples to determine the importance of an event. 
                    Examples: 
                    Question: Someone fell at edge edge_N3_N4. Does this impact the transportation? \n
                    Answer: True, the event will have an impact in transportation and the edge is not usable.\n
                    Question: Someone dropped their ice cream at edge_N1_N2. Does this impact the transportation? \n 
                    Answer:   False, the event won't have an impact in transportation and the edge is usable. \n
                    Question: There is a fire alarm going off at node 2. Does it impact the transportation? \n
                    Answer:  True, the event will have an impact in transportation and other nodes are impacted as well. \n
                    Question: There is a surgery going on at node C. Does it impact the transportation? \n 
                    Anwer: False, the event won't have an impact in transportation and only the current node is impacted by this event. \n"""

    user_prompt = prompt
    full_prompt_usability = f"{context_usability} \n {user_prompt}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response_usability = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': full_prompt_usability}],
        max_tokens=300,
        temperature=0,
    )

    output_usability = response_usability.choices[0].message.content

    response_content = output_usability.strip().lower()

    # Check if the event impacts the transportation and is happening on an edge
    if "true" in response_content and "edge" in response_content:
        context_dynamic = f"""As a professional graph modeler, you're tasked with determining the 
        accessibility of edges in a transportation network. You are given an event that impacts
        the usability of the edge. Now, you must determine whether this event wuld impact the whole
        length of the edge or it happens in a single point of the edge. 
        You can rely on the given examples to determine the importance of an event. 

        Examples: 
        Question: Someone fell at edge edge_N3_N4. Does this impact the whole length of the edge? \n
        Answer:   False, the event will impact only part of the edge edge_N3_N4.\n
        Question: At edge edge_A_B a barrier blocks the entrance. Does this impact the whole length of the edge? \n 
        Answer:   True, the event will impact the whole edge edge_A_B. \n
        Question: At edge edge_A_B the pathway is covered in thick mud due to recent rain. Does this impact the whole length of the edge? \n
        Answer:   True, the event will impact the whole edge edge_A_B. \n
        Question: At edge edge_A_B a bicycle is in the midle of the edge.Does this impact the whole length of the edge? \n 
        Anwer:    False, the event will impact only part of the edge edge_A_B. \n"""

        full_prompt_dynamic = f"{context_dynamic} \n {user_prompt}"

        response_dynamic = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{'role': 'user', 'content': full_prompt_dynamic}],
            max_tokens=300,
            temperature=0,
        )

        output_dynamic = response_dynamic.choices[0].message.content
        response_content = output_dynamic.strip().lower()

        # Checks if the event impacts the whole length of the edge
        if "true" in response_content:
            context_length = f"""As a professional graph modeler, you're tasked with determining the 
            accessibility of edges in a transportation network. You must determine how much was the 
            provided edge affected based on how important the event given as input is. 
            The values are between 0-100 with 100 being the most affected, values between 0-25 are for 
            events that affect the accessibility of the edge a little bit, values between 25-50 are for 
            events that moderately affect the accessibility of the edge, values between 50-75 are for 
            events that seriously affect the accessibility of the edge and values between 75-100
            affect the accessibility of the edge critically. 
            You can rely on the given examples to determine the importance of an event. 

                Examples: 
                Question: At edge edge_A_B a barrier blocks the entrance. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected. \n 
                Answer:   The value is 98. \n
                Question: At edge edge_A_B the pathway is covered in thick mud due to recent rain. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected. \n
                Answer:   The value is 70. \n

                Please provide a mandatory single value between 0 and 100 for how much the accessibility of 
                the edge for the transportation vehicles is affected. Format it exactly like this: The value is X."""

            full_prompt_length = f"{context_length} \n {user_prompt}"

            response_length = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': full_prompt_length}],
                max_tokens=300,
                temperature=0,
            )

            method = "factor"
            output_length = response_length.choices[0].message.content

        # If the event impacts only a part of the edge
        else:
            context_time = f"""As a professional graph modeler, you're tasked with determining the 
            accessibility of edges in a transportation network. You are given an event that impacts the
            accessability of an edge. Now, you must determine based on the event given what time penalty 
            should be applied to a vehicle if it passes through it.
            You can rely on the given examples to determine the importance of an event.

            Examples: 
            Question: Someone fell at edge edge_N3_N4. Please provide a mandatory single value in minutes for how much time will the vehicle be delayed. \n
            Answer:   The value is 15 minutes.\n
            Question: At edge edge_A_B a bicycle is in the midle of the edge.Please provide a mandatory single value in minutes for how much time will the vehicle be delayed. \n 
            Anwer:    The value is 30 minutes. \n

            Please provide a mandatory single value in minutes for how much is the accessibility of the edge 
            for the transportation vehicles is affected. Format it exactly like this: The value is X minutes."""

            full_prompt_time = f"{context_time} \n {user_prompt}"

            response_time = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': full_prompt_time}],
                max_tokens=300,
                temperature=0,
            )

            method = "minutes"
            output_time = response_time.choices[0].message.content

    # Checks if the event impacts the transportation and it is happening on a node
    elif "true" in response_content and "node" in response_content:
        context_nodes = f"""You are a graph expert and you are given the graph of a university hospital
                campus. Nodes are the buildings in the graph and edges are the routes between
                the buidlings. Autonomous vehicles are transporting goods throughout the edges.
                You will be given some information that something is happening at a specific node. Now,
                you need to determine which edges are impacted from this event. Be concise and only give the 
                list of impacted edges in this format for each edge 'edge_node1_node2'."""

        full_prompt_nodes = f"{context_nodes} \n {user_prompt}"

        response_nodes = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{'role': 'user', 'content': full_prompt_nodes}],
            max_tokens=300,
            temperature=0,
            functions=function_descriptions,
            function_call="auto",
        )

        message = response_nodes.choices[0].message

        # Checks if the model called the function
        if message.function_call:
            params = json.loads(message.function_call.arguments)
            chosen_function = eval(message.function_call.name)
            answer = chosen_function(**params)

            response_nodes = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': full_prompt_nodes},
                          {'role': "function", "name": message.function_call.name, "content": answer},
                          ],
                max_tokens=300,
                functions=function_descriptions,
            )

            output_nodes = response_nodes.choices[0].message.content

            context_nodes_time = f"""You're a graph expert and you are given a graph representing a hospital 
            campus where nodes are buildings and edges are the routes between buildings. Autonomous vehicles 
            transport goods along these edges. You will be given certain events happening in buildings that
            cause people to go outside, leading to crowds on the surrounding edges and delaying transportation.
            You need to determine the severity of the event and estimate a time delay for the vehicle based
            on the severity.
            You can rely on the given examples to determine the severity of an event.

                Examples: 
                Question: The ceiling has collapsed on node 2. Please provide a mandatory single value in minutes for how much time will the vehicle be delayed. \n
                Answer:   The value is 120 minutes.\n
                Question: A smoke detection alarm is going off at node 1.Please provide a mandatory single value in minutes for how much time will the vehicle be delayed. \n 
                Anwer:    The value is 30 minutes. \n
                Question: Routine maintenance is happening at node 3. Please provide a mandatory single value in minutes for how much time will the vehicle be delayed.
                Answer: The value is 15 minutes.

            Please provide a mandatory single value in minutes for how much the vehicle will be delayed. 
            Format it exactly like this: The value is X minutes.
                """

            full_prompt_nodes_time = f"{context_nodes_time} \n {user_prompt}"

            response_nodes_time = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': full_prompt_nodes_time}],
                max_tokens=300,
                temperature=0.3,
            )

            method = 'minutes'
            output_nodes_time = response_nodes_time.choices[0].message.content

    else:  # False
        output_dynamic = None
        output_length = None
        output_time = None
        output_nodes = None
        output_nodes_time = None
        method = None

    return output_usability, output_dynamic, output_length, output_time, output_nodes, output_nodes_time, method


if __name__ == "__main__":
    output_usability, output_dynamic, output_length, output_time, output_nodes, output_nodes_time, method = invoke_llm(
        input("Enter prompt: "))
    print(output_usability, output_dynamic, output_length, output_time, output_nodes, output_nodes_time)
