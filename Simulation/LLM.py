from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import re

def llm_edge_removing(Node):
    #Load the edges from the graph, the enviroment and the model
    G=load_edges()
    load_dotenv()
    model = Ollama(model="llama2")
    print(','.join(G))
    
    #Create the LLM
    template_new=f"You are a master graph modeler, you have to remove edges from an edge list that contain inaccessible nodes and it is crucial make sure that the other valid edges remain in the list. We have a directed graph. It contains exclusively these edges: {','.join(G)} \n Question: {{question}}"
    
    prompt_template = PromptTemplate(input_variables=["question"], template=template_new)

    new_graph=LLMChain(prompt=prompt_template,llm=model)

    #Create and run the prompt
    question="Node "+"'"+ str(Node) +"'"+" is not accessible anymore. Could you provide me with the edges from the given graph that have to be removed, as a comma separated value list of tuples like (Node1,Node2)?"
    print(question)
    answer=new_graph.invoke(question)

    #Parse the answer gotten from the LLM
    edges_list = parsing_llm_result(answer['text'])
    
    #Return the edge list
    return edges_list


def parsing_llm_result(answer):
    print(answer)
    pattern = r"[R|r]emov.*?:.*?([^\n]+)"

    removed_edges = re.findall(pattern, answer, re.DOTALL)

    print("List of removed edges:", removed_edges)

    return re.split(r",(?![^()]*\))", removed_edges[0])


def load_edges():
    import pandas as pd

    df = pd.read_csv('medicar\Playground_Arved\edges_UH_Graph.csv')

    edges_list = [(f'{row[0]}', f'{row[1]}') for _, row in df.iterrows()]

    return edges_list


def main():
    print("Removed edges: ",llm_edge_removing("31404364"))


if __name__ == "__main__":
    main()