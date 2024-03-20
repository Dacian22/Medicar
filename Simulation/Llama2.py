import time

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
from langchain_openai import OpenAI
from BuildGraph import set_weights_to_inf
from langsmith import Client


def get_examples():
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
    So the final answer is: List of edges that have to be removed: []. True the edge is not usable.
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
    So the final answer is: List of edges that have to be removed: []. True the edge is not usable.
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
    }
    ]
    return examples

def get_model_llama2():
    #Getting the examples for FewShot approach
    examples = get_examples()

    #Create the template and model
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a proffesional graph modeler, you're tasked with removing edges from an edge list when something happens that would make the edge inpassable.""",
    suffix="{input} Please provide a mandatory True/False value if the edge is usable, but only for the prompt given, don't mention the examples!",
    input_variables=["input"],
    example_separator='\n\n\n')

    model_llama= Ollama(model="llama2")

    return LLMChain(prompt=fewshot_template,llm=model_llama)


def get_model_testing_llama2():
    #Getting the examples for FewShot approach
    examples = get_examples()
    
    #Create the template and model
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a proffesional graph modeler, you're tasked with removing edges from an edge list when something happens that would make the edge inpassable.""",
    #Using a static edge for the moment
    suffix="At edge edge_7120224687_7112240050 {input} Please provide a True/False value if the edge is usable.",
    input_variables=["input"],
    example_separator='\n\n\n')

    model_llama= Ollama("llama2")

    return LLMChain(prompt=fewshot_template,llm=model_llama)

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

def invoke_llm_llama2(prompt):
    #Load the edges
    G=load_edges()

    #Create the LLM
    new_graph=get_model_llama2()

    #Create and run the prompt
    answer=new_graph.invoke(prompt)

    #Return the output of the LLM
    return answer["text"]


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
    output = invoke_llm_llama2(prompt)
    print(output)

    # Parse the output
    parsed_res = parsing_llm_result(output)
    # print(parsed_res)

    # Update graph in the routing
    ref_routing.graph = set_weights_to_inf(ref_routing.graph, parsed_res)
