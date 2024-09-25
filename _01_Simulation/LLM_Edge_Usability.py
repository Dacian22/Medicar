import sys

sys.path.append('/Users/paulkoenig/WebstormProjects/medicar/_01_Simulation/LLM_Edge_Usability.py')
from typing import Any, Dict
from langchain.chains import LLMChain
from dotenv import load_dotenv
import re
import os
import ast
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langserve import RemoteRunnable
llama2 = RemoteRunnable("http://127.0.0.1:8489/llama2")
llama3 = RemoteRunnable("http://127.0.0.1:8489/llama3")
from LLamaLLMWrapper import LLama

load_dotenv(override=True)


def get_examples():
    """
    Provides a list of example questions and answers related to determining
    affected edges in a transportation network given specific events.

    Each example contains:
    - `question`: A description of an event affecting nodes in a graph.
    - `answer`: The answer detailing which edges are affected by the event.
    - `reasoning`: The explanation for why certain edges are removed based on the event.

    Returns:
        list: A list of dictionaries, each containing a question, answer, and reasoning.
    """

    examples = [
    {
        "question": "The edge list provided is: [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'),('N3','N6')]\n Someone fell on the floor on node N2 blocking it. Please provide the affected edges.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that node N2 is not accessible anymore?
    Intermediate answer: Yes, someone falling down would make node N2 inaccessible.
    Follow up: Which edges contain node N2?
    Intermediate answer: Edges ('N1','N2'), ('N2','N3') contain node N2
    So the final answer is: List of edges that have to be removed: ('N1','N2'), ('N2','N3'). False the edge is not usable.
    """,
        "reasoning":"""
    Because the event of someone falling down on the floor is important enough to block access to the node N2 so we are removing the edges that contain the node N2, those being ('N1','N2'), ('N2','N3')
    """,
    },
    {
        "question": "The edge list provided is: [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'),('N3','N6')]\n Someone dropped their ice cream on the floor on node N2. Please provide the affected edges.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that node N2 is not accessible anymore?
    Intermediate answer: No, someone dropping their ice cream would not make node N2 inaccessible.
    So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    """,
        "reasoning":"""
    Because the event of someone dropping their ice cream on the floor is not important enough to block access to the node N2 so we are not removing the edges that contain the node N2, so NO edges are affected
    """,
    },
    {
        "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),('3','6')]\n Someone died on node 2. Please provide the affected edges.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that node 2 is not accessible anymore?
    Intermediate answer: Yes, someone dying would make node 2 inaccessible.
    Follow up: Which edges contain node 2?
    Intermediate answer: Edges ('1','2'), ('2','3') contain node 2
    So the final answer is: List of edges that have to be removed: ('1','2'), ('2','3'). False the edge is not usable.
    """,
        "reasoning":"""
    Because the event of someone dying is important enough to block access to the node 2 so we are removing the edges that contain the node 2, those being ('1','2'), ('2','3')
    """,
    },
    {
        "question": "The edge list provided is: [('A', 'B'), ('B', 'C'), ('C','A'), ('C', 'D'), ('D', 'E'), ('C','F')]\n Someone is having a heart attack on node C. Please provide the affected edges.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that node C is not accessible anymore?
    Intermediate answer: Yes, someone having a heart attack would make node C inaccessible.
    Follow up: Which edges contain node C?
    Intermediate answer: Edges ('B', 'C'), ('C','A'), ('C', 'D'), ('C','F') contain node C
    So the final answer is: List of edges that have to be removed: ('B', 'C'), ('C','A'), ('C', 'D'), ('C','F'). False the edge is not usable.
    """,
        "reasoning":"""
    Because the event of someone having a heart attack is important enough to block access to the node C so we are removing the edges that contain the node C, those being ('B', 'C'), ('C','A'), ('C', 'D'), ('C','F')
    """,
    },
    {
        "question": "The edge list provided is: [('A', 'B'), ('B', 'C'), ('C','A'), ('C', 'D'), ('D', 'E'), ('C','F')]\n Someone dropped their papers on node C. Please provide the affected edges.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that node C is not accessible anymore?
    Intermediate answer: No, someone dropping their papers would notmake node C inaccessible.
    So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    """,
        "reasoning":"""
    Because the event of someone dropping their papers is not important enough to block access to the node C so we are not removing the edges that contain the node C, so NO edges are affected.
    """,
    },
    {
        "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),('3','6')]\n Someone is having a seisure on node 2. Please provide the affected edges.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that node 2 is not accessible anymore?
    Intermediate answer: Yes, someone dying would make node 2 inaccessible.
    Follow up: Which edges contain node 2?
    Intermediate answer: Edges ('1','2'), ('2','3') contain node 2
    So the final answer is: List of edges that have to be removed: ('1','2'), ('2','3'). False the edge is not usable.
    """,
        "reasoning":"""
    Because the event of someone having a seisure is important enough to block access to the node 2 so we are removing the edges that contain the node 2, those being ('1','2'), ('2','3')
    """,
    },
    {
            "question": "The edge list provided is: [('A', 'B'), ('B', 'C'), ('C','A'), ('C', 'D'), ('D', 'E'), ('C','F')]\n A group of people are chatting on node F. Please provide the affected edges.",
            "answer": """
        Are follow up questions needed here: Yes.
        Follow up: Is the event important enough so that node F is not accessible anymore?
        Intermediate answer: No, people chatting would not make node F inaccessible.
        So the final answer is: List of edges that have to be removed: []. True the edge is usable.
        """,
            "reasoning": """
        Because the event of people chatting is not important enough to block access to the node F so we are not removing the edges that contain the node F, so NO edges are affected.
        """,
    },
    {
            "question": "The edge list provided is: [('A', 'B'), ('B', 'C'), ('C','A'), ('C', 'D'), ('D', 'E'), ('C','F')]\n A small animal crosses the pathway on node D. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node D is not accessible anymore?
            Intermediate answer: No, a small animal crossing the pathway would not make node D inaccessible.
            So the final answer is: List of edges that have to be removed: []. True the edge is usable.
            """,
            "reasoning": """
            Because the event of a small animal crossing the pathway is not important enough to block access to the node D so we are not removing the edges that contain the node D, so NO edges are affected.
            """,
    },
    {
            "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),('3','6')]\n A vehicle accident has occured on node 3. Please provide the affected edges.",
            "answer": """
        Are follow up questions needed here: Yes.
        Follow up: Is the event important enough so that node 3 is not accessible anymore?
        Intermediate answer: Yes, a vehicle accident would make node 3 inaccessible.
        Follow up: Which edges contain node 3?
        Intermediate answer: Edges ('2','3'), ('3','4'), ('3','6') contain node 3
        So the final answer is: List of edges that have to be removed: ('2','3'), ('3','4'), ('3','6'). False the edge is not usable.
        """,
            "reasoning": """
        Because the event of a vehicle accident is important enough to block access to the node 3 so we are removing the edges that contain the node 3, those being ('2','3'), ('3','4'), ('3','6')
        """,
    },
    {
            "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),('3','6')]\n A person is walking with a child holding their hand on node 3. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node 3 is not accessible anymore?
            Intermediate answer: No, a person walking with a child holding their hand would not make node 3 inaccessible.
            So the final answer is: List of edges that have to be removed: []. True the edge is usable.
            """,
            "reasoning": """
            Because the event of a person walking with a child holding their hand is not important enough to block access to the node 3 so we are not removing the edges that contain the node 3, so no edges are affected
            """,
    },
    {
            "question": "The edge list provided is: [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'),('N3','N6')]\n A burst fire hydrant floods the path at node N3. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node N3 is not accessible anymore?
            Intermediate answer: Yes, a burst fire hydrant flooding the path would make node N3 inaccessible.
            Follow up: Which edges contain node N3?
            Intermediate answer: Edges ('N2','N3'), ('N3','N4'), ('N3','N6') contain node N3
            So the final answer is: List of edges that have to be removed: ('N2','N3'), ('N3','N4'), ('N3','N6'). False the edge is not usable.
            """,
            "reasoning": """
            Because the event of a burst fire hydrant flooding the path is important enough to block access to the node N3 so we are removing the edges that contain the node N3, those being ('N2','N3'), ('N3','N4'), ('N3','N6')
            """,
    },
    {
            "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n A fallen fence blocks the access near node 6. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node 6 is not accessible anymore?
            Intermediate answer: Yes, a fallen fence blocking the access would make node 6 inaccessible.
            Follow up: Which edges contain node 6?
            Intermediate answer: Edges ('2','6') contain node 6
            So the final answer is: List of edges that have to be removed: ('2','6'). False the edge is not usable.
            """,
            "reasoning": """
            Because the event of a fallen fence blocking the access is important enough to block access to the node 6 so we are removing the edges that contain the node 6, those being ('2','6')
            """,
    },
    {
            "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n Road repair work is in progress near node 2. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node 2 is not accessible anymore?
            Intermediate answer: Yes, road repair work would make node 2 inaccessible.
            Follow up: Which edges contain node 2?
            Intermediate answer: Edges ('1','2'), ('2', '3'), ('2','6') contain node 2
            So the final answer is: List of edges that have to be removed: ('1','2'), ('2', '3'), ('2','6'). False the edge is not usable.
            """,
            "reasoning": """
            Because the event of road repair work is important enough to block access to the node 2 so we are removing the edges that contain the node 2, those being ('1','2'), ('2', '3'), ('2','6')
            """,
    },
    {
            "question": "The edge list provided is: [('node1', 'node2'), ('node2', 'node3'), ('node3', 'node4'), ('node4', 'node5'), ('node2','node6'), ('node3','node7')]\n A cyclist is riding along the path near node node3. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node node3 is not accessible anymore?
            Intermediate answer: No, a cyclist riding along the path would not make node node3 inaccessible.
            So the final answer is: List of edges that have to be removed: []. True the edge is usable.
            """,
            "reasoning": """
            Because the event of a cyclist riding along the path is not important enough to block access to the node node3 so we are not removing any edges. No edges are affected.
            """,
    },
    {
            "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n  A delivery person is dropping off a package near node 1. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node 1 is not accessible anymore?
            Intermediate answer: No, a delivery person dropping off a package would not make node 1 inaccessible.
            So the final answer is: List of edges that have to be removed: []. True the edge is usable.
            """,
            "reasoning": """
            Because the event of a delievery person dropping off a package is not important enough to block access to the node 1 so we are not removing the edges that contain the node 1. No edges are affected.
            """,
    },
    {
            "question": "The edge list provided is: [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('2','6'), ('3','7')]\n A large fallen tree blocks the pathway near node 6. Please provide the affected edges.",
            "answer": """
            Are follow up questions needed here: Yes.
            Follow up: Is the event important enough so that node 6 is not accessible anymore?
            Intermediate answer: Yes, a large fallen tree blocking the pathway would make node 6 inaccessible.
            Follow up: Which edges contain node 6?
            Intermediate answer: Edges ('2','6') contain node 6
            So the final answer is: List of edges that have to be removed: ('2','6'). False the edge is not usable.
            """,
            "reasoning": """
            Because the event of a fallen large tree blocking the access is important enough to block access to the node 6 so we are removing the edges that contain the node 6, those being ('2','6')
            """,
    }
    ]
    return examples


def get_model(model_type,approach):
    """
    Creates an instance of `LLMChain` using the specified model type and approach.

    Args:
        model_type (str): The type of the language model to be used.
        approach (str): The approach to determine which prompt template to use.

    Returns:
        LLMChain: An instance of the `LLMChain` class, configured with the provided model type and prompt template.
    """

    template = get_prompt_template(approach)
    llm = get_llm(model_type)

    return LLMChain(prompt=template, llm=llm)


def get_prompt_template(approach):
    """
    Retrieves the appropriate prompt template based on the specified approach.

    Args:
        approach (str): The approach to determine which prompt template to use. 
                        Expected values are 'fewshot' or 'zeroshot'.

    Returns:
        PromptTemplate: The selected prompt template corresponding to the provided approach.
    """

    template: Dict[str, Any] = {
        'fewshot': get_template_fewshot(),
        'zeroshot': get_template_zeroshot(),
    }

    return template[approach]

def get_template_fewshot():
    """
    Generates a FewShotPromptTemplate for evaluating edge usability in a transportation network.

    Returns:
        FewShotPromptTemplate: A FewShotPromptTemplate object set up for determining edge usability
                               and affected edges based on input obstacles.
    """
    
    examples = get_examples()
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. 
    Don't provide the examples in you response but base your answer on them.""",
    suffix="{input} Please provide the affected edges, and a mandatory True/False value if the edge is usable.",
    input_variables=["input"],
    example_separator='\n\n\n')

    return fewshot_template


def get_template_zeroshot():
    """
    Generates a zero-shot PromptTemplate for evaluating edge usability in a transportation network.

    Returns:
        PromptTemplate: A PromptTemplate object set up for determining edge usability without examples.
    """

    context_zero_shot = """As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network.
    Your goal is to assess whether each provided edge is passable or not for an electrical 
    vehicle based on the importance of the obstacle at the moment that the input is given. 
    Only use the provided graph and do not make assumptions beyond the given context.
    {input} Please provide a True/False value at the end if the edge is usable, exactly like this like this: True the edge is usable or False the edge is not usable."""

    zeroshot_template = PromptTemplate(input_variables=["input"], template=context_zero_shot)

    return zeroshot_template


def get_llm(model_type):
    """
    Retrieves the appropriate language model based on the specified type.

    Args:
        model_type (str): The type of the model to retrieve. Must be one of 'openai', 'llama2', or 'llama3'.

    Returns:
        Any: The language model object for the specified model type.
    """

    model: Dict[str, Any] = {
        'openai': get_model_openai(),
        'llama2': get_model_llama2(),
        'llama3': get_model_llama3(),
    }

    return model[model_type]

def get_model_openai():
    """
    Initializes and returns an OpenAI model instance.

    Returns:
        OpenAI: An instance of the OpenAI model initialized with the API key from the environment variables.

    """
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_model_llama2():
    """
    Initializes and returns a LLama model instance for Llama2.

    Returns:
        LLama: An instance of the LLama model initialized with the identifier for Llama2.
    """
    return LLama(model=llama2)


def get_model_llama3():
    """
    Initializes and returns a LLama model instance for Llama3.

    Returns:
        LLama: An instance of the LLama model initialized with the identifier for Llama3.
    """

    return LLama(model=llama3)


def load_edges():
    """
    Loads edges from a CSV file into a list of tuples.

    Returns:
        list of tuple: A list where each tuple represents an edge with two elements (node1, node2).
    """

    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','_00_Resources','edges_UH_Graph.csv'))
    except:
        df = pd.read_csv(os.path.join('_00_Resources','edges_UH_Graph.csv'))


    edges_list = [(f'{row[0]}', f'{row[1]}') for _, row in df.iterrows()]

    return edges_list

def parse_response(response):
    """
    If response contains True (edge usable) or False (edge not usable) these values are returned as booleans.
    Otherwise None is returned.

    Args:
        response (str): The response from the language model.

    Returns:
        bool: The boolean value extracted from the response, or None if no boolean value is found.
    """

    if "true" in response.lower():
        return True
    elif "false" in response.lower():
        return False
    else:
        return None


def invoke_llm(prompt, model_type='openai', approach='fewshot'):
    """
    Invokes a language model with a given prompt using a specified model type and approach.

    Args:
        prompt (str): The prompt to be sent to the language model.
        model_type (str): The type of the model to use. Options are 'openai', 'llama2', 'llama3'. Default is 'openai'.
        approach (str): The approach to use with the model. Options are 'fewshot' and 'zeroshot'. Default is 'fewshot'.

    Returns:
        str: The text output from the language model.
    """

    G=load_edges()
    new_graph=get_model(model_type,approach)
    answer=new_graph.invoke(prompt)

    return answer["text"]
