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

from langserve import RemoteRunnable
llama2 = RemoteRunnable("http://127.0.0.1:8489/llama2")
llama3 = RemoteRunnable("http://127.0.0.1:8489/llama3") 
from LLamaLLMWrapper import LLama

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
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. Don't provide the examples in you response but base your answer on them.""",
    suffix="{input} Please provide the affected edges, and a mandatory True/False value if the edge is usable.",
    input_variables=["input"],
    example_separator='\n\n\n')

    llama2_llm = LLama(model=llama2)
    return LLMChain(prompt=fewshot_template,llm=llama2_llm)


def get_model_testing_llama2():
    #Getting the examples for FewShot approach
    examples = get_examples()
    
    #Create the template and model
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. 
    Don't provide the examples in you response but base your answer on them.""",
    #Using a static edge for the moment
    suffix="At edge edge_7120224687_7112240050 {input} Please provide a True/False value if the edge is usable.",
    input_variables=["input"],
    example_separator='\n\n\n')

    llama2_llm = LLama(model=llama2)

    return LLMChain(prompt=fewshot_template,llm=llama2_llm)

def get_model_llama2_zero_shot():
    template_zero_shot = """As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. 
    Don't provide the examples in you response but base your answer on them.
    {input} Please provide a True/False value at the end if the edge is usable, exactly like this like this: True the edge is usable or False the edge is not usable."""

   
    
    prompt_template = PromptTemplate(input_variables=["input"], template=template_zero_shot)
   

    llama2_llm = LLama(model=llama2)

    return LLMChain(prompt=prompt_template,llm=llama2_llm)


def get_model_testing_llama2_zero_shot():
    template_zero_shot = """As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. 
    At edge edge_7120224687_7112240050 {input} Please provide a True/False value at the end if the edge is usable, exactly like this: True the edge is usable or False the edge is not usable."""

   
    
    prompt_template = PromptTemplate(input_variables=["input"], template=template_zero_shot)
   

    llama2_llm = LLama(model=llama2)

    return LLMChain(prompt=prompt_template,llm=llama2_llm)




def get_model_llama3_zero_shot():
    template_zero_shot = """As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. 
    {input} Please provide a True/False value at the end if the edge is usable, exactly like this: True the edge is usable or False the edge is not usable."""

   
    
    prompt_template = PromptTemplate(input_variables=["input"], template=template_zero_shot)
   

    llama3_llm = LLama(model=llama3)

    return LLMChain(prompt=prompt_template,llm=llama3_llm)


def get_model_testing_llama3_zero_shot():
    template_zero_shot = """As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether each provided edge is usable or not based on how important the obstacle given as input is. 
    At edge edge_7120224687_7112240050 {input} Please provide a True/False value at the end if the edge is usable, exactly like this: True the edge is usable or False the edge is not usable."""

   
    
    prompt_template = PromptTemplate(input_variables=["input"], template=template_zero_shot)
   

    llama3_llm = LLama(model=llama3)

    return LLMChain(prompt=prompt_template,llm=llama3_llm)



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

def invoke_llm_llama2_zero_shot(prompt):
    #Load the edges
    G=load_edges()

    #Create the LLM
    new_graph=get_model_llama2_zero_shot()

    #Create and run the prompt
    answer=new_graph.invoke(prompt)

    #Return the output of the LLM
    return answer["text"]


def invoke_llm_llama3_zero_shot(prompt):
    #Load the edges
    G=load_edges()

    #Create the LLM
    new_graph=get_model_llama3_zero_shot()

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
    output = invoke_llm_llama3_zero_shot(prompt)
    print(output)

    # Parse the output
    parsed_res = parsing_llm_result(output)
    # print(parsed_res)

    # Update graph in the routing
    ref_routing.graph = set_weights_to_inf(ref_routing.graph, parsed_res)

if __name__ == "__main__":
    main("")
