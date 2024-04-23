import time
from typing import Any, Dict

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import re
import os

import ast


from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI,ChatOpenAI
from BuildGraph import set_weights_to_inf
from langsmith import Client
from langserve import RemoteRunnable
llama2 = RemoteRunnable("http://127.0.0.1:8489/llama2")
llama3 = RemoteRunnable("http://127.0.0.1:8489/llama3")
from LLamaLLMWrapper import LLama

load_dotenv(override=True)
#client=Client(api_key=os.getenv("LANGCHAIN_API_KEY"))


def get_examples():
    examples = [
    {
        "question": "At edge edge_N2_N3 someone fell on the floor blocking it. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N2_N3?
    Intermediate answer: Extremely, someone falling down would seriously affect the accessibility of edge edge_N2_N3 because transportation vehicles would not be able to easily go past.
    So the final answer is: The value for how affected the accessibility of edge_N2_N3 is by the event is 95.
    """,
        "reasoning":"""
    Because the event of someone falling down on the floor affects the edge accessibility of edge_N2_N3 very much then we are giving a high value of 95.
    """,
    },
    {
        "question": "At edge edge_N1_N3 someone dropped their ice cream on the floor on node N2. Please provide a mandatory single value between 0 and 100 for how much the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N1_N3?
    Intermediate answer: Very little, someone dropping their ice cream would barely affect the accessibility of edge edge_N1_N3 because transportation vehicles would not be affected by that.
    So the final answer is: The value for how affected the accessibility of edge_N1_N3 is by the event is 5.
    """,
        "reasoning":"""
    Because the event of someone dropping their ice cream on the floor is not important enough to affect the accessibility of edge edge_N1_N3 a lot so we are we are giving a very low value of 5.
    """,
    },
    {
        "question": "At edge edge_1_2 someone died. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_1_2?
    Intermediate answer: Critticaly, someone dying would make the edge edge_1_2 nearly inaccessible for the transportation vehicles.
    So the final answer is: The value for how affected the accessibility of edge_1_2 is by the event is 100.
    """,
        "reasoning":"""
    Because the event of someone dying is important enough to block access to the edge edge_1_2 we are giving an extremely high value of 100.
    """,
    },
    # {
    #     "question": "The edge list provided is: [('A', 'B'), ('B', 'C'), ('C','A'), ('C', 'D'), ('D', 'E'), ('C','F')]\n Someone is having a heart attack on node C. Please provide the affected edges.",
    #     "answer": """
    # Are follow up questions needed here: Yes.
    # Follow up: Is the event important enough so that node C is not accessible anymore?
    # Intermediate answer: Yes, someone having a heart attack would make node C inaccessible.
    # Follow up: Which edges contain node C?
    # Intermediate answer: Edges ('B', 'C'), ('C','A'), ('C', 'D'), ('C','F') contain node C
    # So the final answer is: List of edges that have to be removed: ('B', 'C'), ('C','A'), ('C', 'D'), ('C','F'). False the edge is not usable.
    # """,
    #     "reasoning":"""
    # Because the event of someone having a heart attack is important enough to block access to the node C so we are removing the edges that contain the node C, those being ('B', 'C'), ('C','A'), ('C', 'D'), ('C','F')
    # """,
    # },
    {
        "question": "At edge edge_A_B someone dropped their papers on the ground. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_A_B?
    Intermediate answer: Very little, someone dropping their papers would barely affect the accessibility of edge edge_A_B because the transportation vehicles would not be affected by that.
    So the final answer is: The value for how affected the accessibility of edge_A_B is by the event is 5.
    """,
        "reasoning":"""
    Because the event of someone dropping their papers is not important enough to disrupt the accessibility of edge_A_B so we are giving a very low value of 5.
    """,
    },
    {
        "question": "At edge edge_A_B the pathway is covered in thick mud due to recent rain. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_A_B?
    Intermediate answer: Seriously, the pathway being covered in thick mud due to recent rain would seriously affect the accessibility of edge edge_A_B because the transportation vehicles would be affected by the event.
    So the final answer is: The value for how affected the accessibility of edge_A_B is by the event is 70.
    """,
        "reasoning":"""
    Because the event of the pathway being covered in thick mud is important enough to disrupt the accessibility of edge_A_B so we are giving a high value of 70.
    """,
    },
    # {
    #     "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),('3','6')]\n Someone is having a seisure on node 2. Please provide the affected edges.",
    #     "answer": """
    # Are follow up questions needed here: Yes.
    # Follow up: Is the event important enough so that node 2 is not accessible anymore?
    # Intermediate answer: Yes, someone dying would make node 2 inaccessible.
    # Follow up: Which edges contain node 2?
    # Intermediate answer: Edges ('1','2'), ('2','3') contain node 2
    # So the final answer is: List of edges that have to be removed: ('1','2'), ('2','3'). False the edge is not usable.
    # """,
    #     "reasoning":"""
    # Because the event of someone having a seisure is important enough to block access to the node 2 so we are removing the edges that contain the node 2, those being ('1','2'), ('2','3')
    # """,
    # },
    # {
    #         "question": "The edge list provided is: [('A', 'B'), ('B', 'C'), ('C','A'), ('C', 'D'), ('D', 'E'), ('C','F')]\n A group of people are chatting on node F. Please provide the affected edges.",
    #         "answer": """
    #     Are follow up questions needed here: Yes.
    #     Follow up: Is the event important enough so that node F is not accessible anymore?
    #     Intermediate answer: No, people chatting would not make node F inaccessible.
    #     So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    #     """,
    #         "reasoning": """
    #     Because the event of people chatting is not important enough to block access to the node F so we are not removing the edges that contain the node F, so NO edges are affected.
    #     """,
    # },
    # {
    #         "question": "The edge list provided is: [('A', 'B'), ('B', 'C'), ('C','A'), ('C', 'D'), ('D', 'E'), ('C','F')]\n A small animal crosses the pathway on node D. Please provide the affected edges.",
    #         "answer": """
    #         Are follow up questions needed here: Yes.
    #         Follow up: Is the event important enough so that node D is not accessible anymore?
    #         Intermediate answer: No, a small animal crossing the pathway would not make node D inaccessible.
    #         So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    #         """,
    #         "reasoning": """
    #         Because the event of a small animal crossing the pathway is not important enough to block access to the node D so we are not removing the edges that contain the node D, so NO edges are affected.
    #         """,
    # },
    # {
    #         "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),('3','6')]\n A vehicle accident has occured on node 3. Please provide the affected edges.",
    #         "answer": """
    #     Are follow up questions needed here: Yes.
    #     Follow up: Is the event important enough so that node 3 is not accessible anymore?
    #     Intermediate answer: Yes, a vehicle accident would make node 3 inaccessible.
    #     Follow up: Which edges contain node 3?
    #     Intermediate answer: Edges ('2','3'), ('3','4'), ('3','6') contain node 3
    #     So the final answer is: List of edges that have to be removed: ('2','3'), ('3','4'), ('3','6'). False the edge is not usable.
    #     """,
    #         "reasoning": """
    #     Because the event of a vehicle accident is important enough to block access to the node 3 so we are removing the edges that contain the node 3, those being ('2','3'), ('3','4'), ('3','6')
    #     """,
    # },
    # {
    #         "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),('3','6')]\n A person is walking with a child holding their hand on node 3. Please provide the affected edges.",
    #         "answer": """
    #         Are follow up questions needed here: Yes.
    #         Follow up: Is the event important enough so that node 3 is not accessible anymore?
    #         Intermediate answer: No, a person walking with a child holding their hand would not make node 3 inaccessible.
    #         So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    #         """,
    #         "reasoning": """
    #         Because the event of a person walking with a child holding their hand is not important enough to block access to the node 3 so we are not removing the edges that contain the node 3, so no edges are affected
    #         """,
    # },
    {
            "question": "At edge edge_N5_N6 a burst fire hydrant floods the path. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: How affected is the accessibility of edge edge_N5_N6?
            Intermediate answer: Seriously, a burst fire hydrant flooding the path would seriously affect the accessibility of edge edge_N5_N6 because the transportation vehicles would be very affected by the event.
            So the final answer is: The value for how affected the accessibility of edge_A_B is by the event is 68.
            """,
            "reasoning": """
            Because the event of a burst fire hydrant flooding the path is important enough to seriously affect the accessibility of edge_N5_N6 so we are giving a high value of 68.
            """,
    },
    {
            "question": "At edge edge_3_4 a fallen tree blocks part of the path. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: How affected is the accessibility of edge edge_3_4?
            Intermediate answer: Moderately, a fallen tree blocking part of the path would moderately affect the accessibility of edge edge_3_4 because the transportation vehicles would have to change their speed or position to be sure that they avoid it.
            So the final answer is: The value for how affected the accessibility of edge_A_B is by the event is 40.
            """,
            "reasoning": """
            Because the event of a fallen tree blocking part of the path is important enough to moderately affect the accessibility of edge_3_4 so we are giving a moderate value of 40.
            """,
    },
    # {
    #         "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n A fallen fence blocks the access near node 6. Please provide the affected edges.",
    #         "answer": """
    #         Are follow up questions needed here: Yes.
    #         Follow up: Is the event important enough so that node 6 is not accessible anymore?
    #         Intermediate answer: Yes, a fallen fence blocking the access would make node 6 inaccessible.
    #         Follow up: Which edges contain node 6?
    #         Intermediate answer: Edges ('2','6') contain node 6
    #         So the final answer is: List of edges that have to be removed: ('2','6'). False the edge is not usable.
    #         """,
    #         "reasoning": """
    #         Because the event of a fallen fence blocking the access is important enough to block access to the node 6 so we are removing the edges that contain the node 6, those being ('2','6')
    #         """,
    # },
    # {
    #         "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n Road repair work is in progress near node 2. Please provide the affected edges.",
    #         "answer": """
    #         Are follow up questions needed here: Yes.
    #         Follow up: Is the event important enough so that node 2 is not accessible anymore?
    #         Intermediate answer: Yes, road repair work would make node 2 inaccessible.
    #         Follow up: Which edges contain node 2?
    #         Intermediate answer: Edges ('1','2'), ('2', '3'), ('2','6') contain node 2
    #         So the final answer is: List of edges that have to be removed: ('1','2'), ('2', '3'), ('2','6'). False the edge is not usable.
    #         """,
    #         "reasoning": """
    #         Because the event of road repair work is important enough to block access to the node 2 so we are removing the edges that contain the node 2, those being ('1','2'), ('2', '3'), ('2','6')
    #         """,
    # },
    # {
    #         "question": "The edge list provided is: [('node1', 'node2'), ('node2', 'node3'), ('node3', 'node4'), ('node4', 'node5'), ('node2','node6'), ('node3','node7')]\n A cyclist is riding along the path near node node3. Please provide the affected edges.",
    #         "answer": """
    #         Are follow up questions needed here: Yes.
    #         Follow up: Is the event important enough so that node node3 is not accessible anymore?
    #         Intermediate answer: No, a cyclist riding along the path would not make node node3 inaccessible.
    #         So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    #         """,
    #         "reasoning": """
    #         Because the event of a cyclist riding along the path is not important enough to block access to the node node3 so we are not removing any edges. No edges are affected.
    #         """,
    # },
    # {
    #         "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n  A delivery person is dropping off a package near node 1. Please provide the affected edges.",
    #         "answer": """
    #         Are follow up questions needed here: Yes.
    #         Follow up: Is the event important enough so that node 1 is not accessible anymore?
    #         Intermediate answer: No, a delivery person dropping off a package would not make node 1 inaccessible.
    #         So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    #         """,
    #         "reasoning": """
    #         Because the event of a delievery person dropping off a package is not important enough to block access to the node 1 so we are not removing the edges that contain the node 1. No edges are affected.
    #         """,
    # },
    # {
    #         "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n A large fallen tree blocks the pathway near node 6. Please provide the affected edges.",
    #         "answer": """
    #         Are follow up questions needed here: Yes.
    #         Follow up: Is the event important enough so that node 6 is not accessible anymore?
    #         Intermediate answer: Yes, a large fallen tree blocking the pathway would make node 6 inaccessible.
    #         Follow up: Which edges contain node 6?
    #         Intermediate answer: Edges ('2','6') contain node 6
    #         So the final answer is: List of edges that have to be removed: ('2','6'). False the edge is not usable.
    #         """,
    #         "reasoning": """
    #         Because the event of a fallen large tree blocking the access is important enough to block access to the node 6 so we are removing the edges that contain the node 6, those being ('2','6')
    #         """,
    # }
    ]
    return examples


def get_model(model_type,approach):
    template = get_prompt_template(approach)

    llm = get_llm(model_type)

    return LLMChain(prompt=template, llm=llm)


def get_prompt_template(approach):
    template: Dict[str, Any] = {
        'fewshot': get_template_fewshot(),
        'zeroshot': get_template_zeroshot(),
        'testing_fewshot': get_template_testing_fewshot(),
    }

    return template[approach]

def get_template_fewshot():
    #Getting the examples for FewShot approach
    examples = get_examples()

    #Create the template and model
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine how much was the provided edge affected based on how important the obstacle given as input is. 
    The values are between 0-100 with 100 being the most affected, values between 0-25 are for events that affect the accessibility of the edge a little bit, 
    values between 25-50 are for events that moderately affect the accessibility of the edge, values between 50-75 are for events that seriously affect the accessibility of the edge 
    and values between 75-100 affect the accessibility of the edge critically. 
    Don't provide the examples in you response but base your answer on them.""",
    suffix="{input} Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge for the transportation vehicles is affected.",
    input_variables=["input"],
    example_separator='\n\n\n')

    return fewshot_template


def get_template_zeroshot():
    context_zero_shot = """As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network.
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is.
    Don't provide the examples in you response but base your answer on them.
    {input} Please provide a True/False value at the end if the edge is usable, exactly like this like this: True the edge is usable or False the edge is not usable."""

    zeroshot_template = PromptTemplate(input_variables=["input"], template=context_zero_shot)

    return zeroshot_template


def get_template_testing_fewshot():
    #Getting the examples for FewShot approach
    examples = get_examples()
    
    #Create the template and model
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    fewshot_testing_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network.
    You're tasked with removing edges from an edge list when something happens that would make the edge impassable. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. Respond
    with True if the edge is available and False if the edge is not available.""",
    #Using a static edge
    suffix="At edge edge_7120224687_7112240050 {input} Please provide the affected edges, and a True/False value if the edge is usable.",
    input_variables=["input"],
    example_separator='\n\n\n')

    return fewshot_testing_template

def get_llm(model_type):
    model: Dict[str, Any] = {
        'openai': get_model_openai(),
        'llama2': get_model_llama2(),
        'llama3': get_model_llama3(),
    }

    return model[model_type]

def get_model_openai():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_model_llama2():
    return LLama(model=llama2)

def get_model_llama3():
    return LLama(model=llama3)


def invoke_llm(prompt, model_type='openai', approach='fewshot'):
    #Load the edges
    G=load_edges()

    #Create the LLM
    new_graph=get_model(model_type,approach)

    #Create and run the prompt
    answer=new_graph.invoke(prompt)

    #Return the output of the LLM
    return answer["text"]



def parsing_llm_result(answer):
    pattern = r"\([`']?\d+[`']?, [`']?\d+[`']?\)"

    removed_edges = re.findall(pattern, answer, re.DOTALL)

    # print("List of removed edges:", removed_edges)

    removed_edges_cleaned = []

    for removed_edge in removed_edges:
        # print(removed_edge)
        cleaned = removed_edge.strip("'´")
        # print(cleaned)
        cleaned = ast.literal_eval(cleaned)
        # print(cleaned)
        removed_edges_cleaned.append(cleaned)

    print("List of edges which weights are changed to infinity:", removed_edges)

    return removed_edges


def load_edges():
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','Playground_Arved','csv','edges_UH_Graph.csv'))
    except:
        df = pd.read_csv(os.path.join('Playground_Arved','csv','edges_UH_Graph.csv'))


    edges_list = [(f'{row[0]}', f'{row[1]}') for _, row in df.iterrows()]
    #print("EDGES" ,edges_list)
    return edges_list


def main(ref_routing):
    # Get node id as input from the command line
    time.sleep(5)
    prompt = input("Enter your prompt: ")

    # Get output of the LLM
    output = invoke_llm(prompt)
    #output=try_llm(prompt)
    print(output)

    # Parse the output
    parsed_res = parsing_llm_result(output)
    # print(parsed_res)

    # Update graph in the routing
    ref_routing.graph = set_weights_to_inf(ref_routing.graph, parsed_res)


if __name__ == "__main__":
    main("")


