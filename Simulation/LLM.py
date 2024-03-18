import time

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import re
import os

import ast

from Simulation.BuildGraph import set_weights_to_inf


def invoke_llm(prompt):
    #Load the edges from the graph, the enviroment and the model
    G=load_edges()
    load_dotenv()
    model = Ollama(model="llama2")
    print(G)

    #Create the LLM
    template_new=f"context: {G} \n  requirements: Do not make something up. DO NOT PROVIDE CODE! Please Provide the result!. Only use the provided edges in the context. Please provide the edges as tuples. \n question: {{question}}"
    
    prompt_template = PromptTemplate(input_variables=["question"], template=template_new)

    new_graph=LLMChain(prompt=prompt_template,llm=model)

    #Create and run the prompt
    answer=new_graph.invoke(prompt)

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

    df = pd.read_csv(os.path.join('..','Playground_Arved','edges_UH_Graph.csv'))

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

if __name__ == "__main__":
    main()