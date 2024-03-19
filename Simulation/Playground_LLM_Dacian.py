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

load_dotenv()

client=Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

def invoke_llm(prompt):
    #Load the edges from the graph, the enviroment and the model
    G=load_edges()
    model = Ollama(model="llama2")
    #print(G)

    #Create the LLM
    template_new=f"context: {G} \n  requirements: Do not make something up. DO NOT PROVIDE CODE! Please Provide the result!. Only use the provided edges in the context. Please provide the edges as tuples. \n question: {{question}}"
    template_new_example=f"As a proffesional graph modeler, you're tasked with removing edges from an edge list when something happens. Requirements: Only use the Graph provided in the question. Please list the edges that need to be removed in a comma seperated list of tuples. Respect the structure of the example for your response. THIS IS JUST AN EXAMPLE DO NOT USE THIS Example of Input: The edge list provided is: [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'),('N3','N6')]\n Someone fell on the floor on node N2 blocking it. Please provide the affected edges.\n\nExample of Output:List of edges that have to be removed: ('N1','N2'), ('N2','N3')\n Reasoning: We remove only the edges that contain node N2 because it can't be accessed anymore.END OF EXAMPLE\n\n Answer this Question: The edge list provided is: {G} \n{{question}} Please provide the affected edges."   #prompt_template = PromptTemplate(input_variables=["question"], template=template_new)
    template_new_example2=f"As a proffesional graph modeler, you're tasked with removing edges from an edge list when something happens. Requirements: Only use the Graph provided in the question. Please list the edges that need to be removed in a comma seperated list of tuples. Respect the structure of the example for your response. THIS IS JUST AN EXAMPLE DO NOT USE THIS: USER: The edge list provided is: [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'),('N3','N6')]\n USER: Someone fell on the floor on node N2 blocking it. Please provide the affected edges.\n\n YOU:List of edges that have to be removed: ('N1','N2'), ('N2','N3')\n Reasoning: We remove only the edges that contain node N2 because it can't be accessed anymore.END OF EXAMPLE\n\n Answer this Question: The edge list provided is: {G[:50]} \n{{question}} Please provide the affected edges."   #prompt_template = PromptTemplate(input_variables=["question"], template=template_new)

    prompt_template = PromptTemplate(input_variables=["question"], template=template_new_example)

    new_graph=LLMChain(prompt=prompt_template,llm=model)

    #Create and run the prompt
    print(f"The edge list provided is: {G[:50]} \n" + prompt)
    answer=new_graph.invoke(prompt)

    return answer["text"]

def try_llm(prompt):
    examples = [
    {
        "question": "The edge list provided is: [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'),('N3','N6')]\n Someone fell on the floor on node N2 blocking it. Please provide the affected edges.",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Is the event important enough so that node N2 is not accessible anymore?
Intermediate answer: Yes, someone falling down would make node N2 inaccessible.
Follow up: Which edges contain node N2?
Intermediate answer: Edges ('N1','N2'), ('N2','N3') contain node N2
So the final answer is: List of edges that have to be removed: ('N1','N2'), ('N2','N3')
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
So the final answer is: List of edges that have to be removed: []
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
So the final answer is: List of edges that have to be removed: ('1','2'), ('2','3')
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
So the final answer is: List of edges that have to be removed: ('B', 'C'), ('C','A'), ('C', 'D'), ('C','F')
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
So the final answer is: List of edges that have to be removed: []
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
So the final answer is: List of edges that have to be removed: ('1','2'), ('2','3')
""",
        "reasoning":"""
Because the event of someone having a seizure is important enough to block access to the node 2 so we are removing the edges that contain the node 2, those being ('1','2'), ('2','3')
""",
    }

]
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    #prefix ="""As a proffesional graph modeler, you're tasked with removing edges from an edge list when something important happens. Requirements: MAKE SURE TO ONLY use the Graph provided in the question, DO NOT use ANY other graphs. Please list the edges that need to be removed in a comma seperated list of tuples.""",
    suffix="{input}",
    input_variables=["input"],
    example_separator='\n\n\n')
    G=load_edges()
    #load_dotenv()
    model = Ollama(model="llama2")
    model_mistral = Ollama(model="mistral")
    model_openai= OpenAI()
    
    #print(G)

    #Create the LLM
    template_new=f"context: {G} \n  requirements: Do not make something up. DO NOT PROVIDE CODE! Please Provide the result!. Only use the provided edges in the context. Please provide the edges as tuples. \n question: {{question}}"
    template_new_example=f"As a proffesional graph modeler, you're tasked with removing edges from an edge list when something happens. Requirements: Only use the Graph provided in the question. Please list the edges that need to be removed in a comma seperated list of tuples. Respect the structure of the example for your response. THIS IS JUST AN EXAMPLE DO NOT USE THIS Example of Input: The edge list provided is: [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'),('N3','N6')]\n Someone fell on the floor on node N2 blocking it. Please provide the affected edges.\n\nExample of Output:List of edges that have to be removed: ('N1','N2'), ('N2','N3')\n Reasoning: We remove only the edges that contain node N2 because it can't be accessed anymore.END OF EXAMPLE\n\n Answer this Question: The edge list provided is: {G} \n{{question}} Please provide the affected edges."   #prompt_template = PromptTemplate(input_variables=["question"], template=template_new)
    template_new_example2=f"As a proffesional graph modeler, you're tasked with removing edges from an edge list when something happens. Requirements: Only use the Graph provided in the question. Please list the edges that need to be removed in a comma seperated list of tuples. Answer this Question: \n{{question}} "   #prompt_template = PromptTemplate(input_variables=["question"], template=template_new)

    #prompt_template = PromptTemplate(input_variables=["question"], template=template_new)
    # prompt_template = PromptTemplate(input_variables=["question"], template=template_new_example2)
    # prompt_template = PromptTemplate(input_variables=["question"], template=template_new)

    # new_graph=LLMChain(prompt=prompt_template,llm=model)
    # new_graph=LLMChain(prompt=fewshot_template,llm=model)
    # new_graph=LLMChain(prompt=fewshot_template,llm=model_flan)
    #new_graph=LLMChain(prompt=fewshot_template,llm=model_mistral)
    new_graph=LLMChain(prompt=fewshot_template,llm=model_openai)


    #Create and run the prompt
    #print(f"The edge list provided is: {G[:50]} \n" + prompt)
    answer=new_graph.invoke(f"The edge list provided is: {G} " + prompt + ". Please provide the affected edges.")  

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
    #output = invoke_llm(prompt)
    ####################
    output=try_llm(prompt)
    print(output)

    # Parse the output
    parsed_res = parsing_llm_result(output)
    # print(parsed_res)

    # Update graph in the routing
    ref_routing.graph = set_weights_to_inf(ref_routing.graph, parsed_res)


def test():
    edge_ids,tests=load_tests()
    print(edge_ids,tests)
    try:
        f = open(os.path.join('..','Simulation','LLMFewShot.txt'),'w')
    except:
        f = open(os.path.join('Simulation','LLMFewShot.txt'),'w')

    for test,edge in zip(tests,edge_ids):
       output=try_llm(f'At edge {edge} {test}')
       print(output)
       f.write(output+'\n\n\n')

    f.close()

def load_tests():
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','Playground_Arved','csv','edges_UH_Graph_Ids.csv'))
    except:
        df = pd.read_csv(os.path.join('Playground_Arved','csv','edges_UH_Graph_Ids.csv'))

    try:
        df_test = pd.read_csv(os.path.join('..','Playground_LLM','EvaluationDatabase.csv'),delimiter=';')
    except:
        df_test = pd.read_csv(os.path.join('Playground_LLM','EvaluationDatabase.csv'),delimiter=';')
    #print(df)
    #print(df_test)
    edge_ids = [f'{row[0]}' for _,row in df.iterrows()]
    tests = [f'{test[0]}' for _,test in df_test.iterrows()]
    return (edge_ids,tests)
    
if __name__ == "__main__":
    test()
    #main("")


