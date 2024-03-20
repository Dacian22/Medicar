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
import csv

import ast

import pandas as pd

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
    vehicle based on the importance of the obstacle at the moment that the input is given. 
    Only use the provided graph and do not make assumptions beyond the given context. 
    Respond with True if the edge is available and False when the edge is not available.
    Answer with a short explanation in one sentence.
    \nquestion: {{question}}"""

   
    
    prompt_template = PromptTemplate(input_variables=["question"], template=template_new)
   

    
    llm = OpenAI()

    #new_graph=LLMChain(prompt=prompt_template,llm=model)
    llm_chain = LLMChain(prompt = prompt_template, llm=llm)
    #Create and run the prompt
    #answer=new_graph.invoke(prompt)
    
    answer = llm_chain.invoke(prompt)

    return answer["text"]


def parsing_llm_result(answer, prompt):
    pattern = r"\([`']?\d+[`']?, [`']?\d+[`']?\)"
    decision=""
    # split the answer into words
    split_answer = answer.split()
    # iterate through the split answer
    for word in split_answer:
        # check if the word is "True" or "False"
        if word.rstrip('.,').lower() == "true" or word.rstrip('.,').lower() == "false":
            # store the decision and break the loop
            decision = word.rstrip('.,').lower()
            break

    removed_edges = []
    if decision == "false":
        # check if the word "edge" appears in the prompt in any format
        if "edge" in prompt.lower():  
            # find all occurrences of two numbers close to the word "edge"
            removed_edges = re.findall(r'edge_(\d+)_(\d+)', prompt)

    else:
        removed_edges = []


    # find edges in the answer
    removed_edges_answer = re.findall(pattern, answer, re.DOTALL)

    # clean the found edges
    removed_edges_cleaned = [ast.literal_eval(edge.strip("'´")) for edge in removed_edges_answer]

    # print the list of edges which weights are changed to infinity
    print("List of edges which weights are changed to infinity:", removed_edges)

    return removed_edges, decision


def load_edges():
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','Playground_Arved','csv','edges_UH_Graph.csv'))
    except:
        df = pd.read_csv(os.path.join('Playground_Arved','csv','edges_UH_Graph.csv'))


    edges_list = [(f'{row[0]}', f'{row[1]}') for _, row in df.iterrows()]

    return edges_list

def read_questions_from_csv(filename):
    questions = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            questions.append(row[0])  
    return questions

# Function to write decisions to a CSV file
def write_decisions_to_csv(filename, questions, decisions, parsed_res):
    print(0)
    i=0
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for question, decision,parsed_res_entry in zip(questions, decisions,parsed_res):
            writer.writerow([question, decision, parsed_res_entry])
            i+=1
            print(i)


def main(ref_routing):

    # Read questions from CSV file
    questions = read_questions_from_csv(r'Playground_LLM\questions.csv')

    # Initialize list to store decisions
    decisions = []
    parsed_res=[]
    # Loop over each question
    for question in questions:
        # Invoke LLM for each question
        output = invoke_llm(question)
        # Parse the output
        parsed_res, decision = parsing_llm_result(output, question)
        # Update decisions list
        decisions.append(decision)
        parsed_res.append(parsed_res)
    # Write decisions back to CSV file
    write_decisions_to_csv('decisions.csv', questions,decisions,parsed_res)
    # Get node id as input from the command line
    time.sleep(5)
    print(1)
    #prompt = input("Enter your prompt: ")

    # Get output of the LLM
    #output = invoke_llm(prompt)
    #print(output)

    # Parse the output
    #parsed_res,decision = parsing_llm_result(output, prompt)
    # print(parsed_res)

    # Update graph in the routing
    #ref_routing.graph = set_weights_to_inf(ref_routing.graph, parsed_res)
