import time

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import re
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

import ast

# from Simulation.BuildGraph import set_weights_to_inf
from BuildGraph import set_weights_to_inf

#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def invoke_llm(prompt):
    #Load the edges from the graph, the enviroment and the model
    G=load_edges()
    load_dotenv()
    #model = Ollama(model="llama2")
    #print(G)

    #Create the LLM
    template_new = """context: {G} 
    requirements: You're a graph expert tasked with determining the availability of edges in 
    a transportation network for electrical vehicles. 
    Your goal is to assess whether each provided edge is passable or not for an electrical 
    vehicle at the moment that the input is given. Only use the provided graph and do not 
    make assumptions beyond the given context. Give a True/False answer with a short explanation
    in one sentence.
    \nquestion: {{question}}"""

   
    
    prompt_template = PromptTemplate(input_variables=["question"], template=template_new)
   

    
    llm = OpenAI()

    #new_graph=LLMChain(prompt=prompt_template,llm=model)
    llm_chain = LLMChain(prompt = prompt_template, llm=llm)
    #Create and run the prompt
    #answer=new_graph.invoke(prompt)
    
    answer = llm_chain.invoke(prompt)

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

    #print("List of edges which weights are changed to infinity:", removed_edges)

    return removed_edges


def load_edges():
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','Playground_Arved','csv','edges_UH_Graph.csv'))
    except:
        df = pd.read_csv(os.path.join('Playground_Arved','csv','edges_UH_Graph.csv'))


    edges_list = [(f'{row[0]}', f'{row[1]}') for _, row in df.iterrows()]

    return edges_list


def main(ref_routing):
    # Get node id as input from the command line
    time.sleep(5)
    prompt = input("Enter your prompt: ")

    # Get output of the LLM
    output = invoke_llm(prompt)
    print(output)

    # Parse the output
    parsed_res = parsing_llm_result(output)
    # print(parsed_res)

    # Update graph in the routing
    ref_routing.graph = set_weights_to_inf(ref_routing.graph, parsed_res)
