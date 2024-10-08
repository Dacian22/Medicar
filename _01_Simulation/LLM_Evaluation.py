import openai
from dotenv import load_dotenv
from openai import OpenAI
import os
import warnings
import pandas as pd
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv(os.path.join(os.getenv("RESOURCES"),'eval-res-metamodel.csv'), sep=';')
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
scores_length=[]
scores_time=[]
scores_nodes_time=[]
examples_length=[]
examples_time=[]
examples_nodes_time=[]
factors=[]
times=[]
nodes_times=[]


for index, row in df.iterrows():
    if pd.notna(row['output_length']):
        output_length = str(row['output_length'])
        factor = re.findall(r'\d+\.\d+|\d+', output_length)

        prompt = f'''You are a graph expert and you are given a graph structure of a university 
        hospital, where nodes are buildings and edges are routes between the buildings. 
        Autonomous vehicles are transporting goods through the edges but certain events that
        happen on edges can impact the transportation. A numerical factor which expresses how much
        the accessibility of the edge is impacted by the event is assigned to each event.
        This numerical factor can be from 0, where the accessibility of the edge is not impacted 
        at all, to 100, where the accessibility of the edge is fully impacted.
        You need to determine how accurate this factor is for this event based on the type and severity
        of the event: \n
        Event: {row['examples']} \n
        Factor: {factor} \n

        Please provide a mandatory single value between 0 and 1 for how accurate the factor is
        based on the event. Don't replicate the factor. Format it exactly like this: The score is X.'''

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [{'role': 'user', 'content': prompt}],
            max_tokens=300,
            temperature=0,
            )
            
           
        output = response.choices[0].message.content
        score = re.findall(r'\d+\.\d+|\d+', output) 
        if score:
            scores_length.append(float(score[0]))
        else:
            scores_length.append(None)
        
        examples_length.append(row['examples'])
        factors.append(factor)
    
    elif pd.notna(row['output_time']):
        output_time = str(row['output_time'])
        time = re.findall(r'\d+\.\d+|\d+', output_time)

        prompt = f'''You are a graph expert and you are given a graph structure of a university 
        hospital, where nodes are buildings and edges are routes between the buildings. 
        Autonomous vehicles are transporting goods through the edges but certain events that
        happen on edges can impact the transportation. A time penalty which expresses how much
        the vehicle will be delayed if it chooses this edge is assigned to each event.
        The time penalty is given in minutes.
        You need to determine how accurate this time penalty is for this event based on the 
        type and severity of the event: \n
        Event: {row['examples']} \n
        Time Penalty in minutes: {time} \n

        Please provide a mandatory single value between 0 and for how accurate the time penalty is
        based on the event. Format it exactly like this: The score is X.'''

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [{'role': 'user', 'content': prompt}],
            max_tokens=300,
            temperature=0,
            )
            
           
        output = response.choices[0].message.content
        score = re.findall(r'\d+\.\d+|\d+', output) 
        if score:
            scores_time.append(float(score[0]))
        else:
            scores_time.append(None)
        
        examples_time.append(row['examples'])
        times.append(time)
    
    elif pd.notna(row['output_nodes_time']):
        output_nodes_time = str(row['output_nodes_time'])
        time = re.findall(r'\d+\.\d+|\d+', output_nodes_time)

        prompt = f'''You are a graph expert and you are given a graph structure of a university 
        hospital, where nodes are buildings and edges are routes between the buildings. 
        Autonomous vehicles are transporting goods through the edges but certain events that
        happen on edges can impact the transportation. A time penalty which expresses how much
        the vehicle will be delayed if it chooses this edge is assigned to each event.
        The time penalty is given in minutes.
        You need to determine how accurate this time penalty is for this event based on the 
        type and severity of the event: \n
        Event: {row['examples']} \n
        Time Penalty in minutes: {time} \n

        Please provide a mandatory single value between 0 and 1 for how accurate the time penalty is
        based on the event. Format it exactly like this: The score is X.'''

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [{'role': 'user', 'content': prompt}],
            max_tokens=300,
            temperature=0,
            )
            
           
        output = response.choices[0].message.content
        score = re.findall(r'\d+\.\d+|\d+', output) 
        if score:
            scores_nodes_time.append(float(score[0]))
        else:
            scores_nodes_time.append(None)
        
        examples_nodes_time.append(row['examples'])
        nodes_times.append(time)


new_df_length = pd.DataFrame({'Example': examples_length, 'Factor': factors, 'Score': scores_length})
new_df_length.to_csv('evaluation_length.csv', index=False)

new_df_time = pd.DataFrame({'Example': examples_time, 'Time Penalty in minutes': times, 'Score': scores_time})
new_df_time.to_csv('evaluation_time.csv', index=False)

new_df_nodes_time = pd.DataFrame({'Example': examples_nodes_time, 'Time Penalty in minutes': nodes_times, 'Score': scores_nodes_time})
new_df_nodes_time.to_csv('evaluation_nodes_time.csv', index=False)
