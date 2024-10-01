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


def get_examples_factor():
    """
    Provides a list of example scenarios with corresponding questions and answers that evaluate how much 
    the accessibility of a given edge is affected by an event. Each example includes a question, an answer with 
    follow-up reasoning, and the final value assigned to represent the degree of impact on accessibility.

    Returns:
        list: A list of dictionaries where each dictionary contains the following keys:
            - 'question': A string that poses the scenario to evaluate the edge's accessibility.
            - 'answer': A string that contains the follow-up questions, intermediate answers, and the final value.
            - 'reasoning': A string explaining the logic behind the final value assigned to the edge's accessibility.
    """

    examples = [
    {
        "question": "At edge edge_N2_N3 someone fell on the floor blocking it. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N2_N3?
    Intermediate answer: Extremely, someone falling down would seriously affect the accessibility of edge edge_N2_N3 because transportation vehicles would not be able to easily go past.
    So the final answer is: The value is 95.
    """,
        "reasoning":"""
    Because the event of someone falling down on the floor affects the edge accessibility of edge_N2_N3 very much then we are giving a high value of 95.
    """,
    },
    {
        "question": "At edge edge_N1_N3 someone dropped their ice cream on the floor. Please provide a mandatory single value between 0 and 100 for how much the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N1_N3?
    Intermediate answer: Very little, someone dropping their ice cream would barely affect the accessibility of edge edge_N1_N3 because transportation vehicles would not be affected by that.
    So the final answer is: The value is 5.
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
    So the final answer is: The value is 100.
    """,
        "reasoning":"""
    Because the event of someone dying is important enough to block access to the edge edge_1_2 we are giving an extremely high value of 100.
    """,
    },
    {
        "question": "At edge edge_A_B someone dropped their papers on the ground. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_A_B?
    Intermediate answer: Very little, someone dropping their papers would barely affect the accessibility of edge edge_A_B because the transportation vehicles would not be affected by that.
    So the final answer is: The value is 5.
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
    So the final answer is: The value is 70.
    """,
        "reasoning":"""
    Because the event of the pathway being covered in thick mud is important enough to disrupt the accessibility of edge_A_B so we are giving a high value of 70.
    """,
    },
    {
        "question": "At edge edge_C_D a group of people are chatting. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_C_D?
    Intermediate answer: Not at all, people chatting would not affect the accessibility of edge edge_C_D.
    So the final answer is: The value for how affected the accessibility of edge_C_D is by the event is 0.
    """,
        "reasoning": """
    Because the event of people chatting would not affect the accessibility of edge edge_C_D we are giving a very low value of 0.
    """,
    },
    {
        "question": "At edge edge_N1_N4 a medium-sized animal is on the path. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N1_N4?
    Intermediate answer: Moderately, a medium-sized animal being on the path would moderately affect the accessibility of edge edge_N1_N4 because the transportation vehicles would have to change their speed or position a lot to be sure that they avoid it.
    So the final answer is: The value is 50.
    """,
        "reasoning": """
    Because the event of a medium-sized animal being on the path is important enough to moderately affect the accessibility of edge_N1_N4 so we are giving a moderate value of 50.
    """,
    },
    {
        "question": "At edge edge_N5_N6 a burst fire hydrant floods the path. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N5_N6?
    Intermediate answer: Seriously, a burst fire hydrant flooding the path would seriously affect the accessibility of edge edge_N5_N6 because the transportation vehicles would be very affected by the event.
    So the final answer is: The value is 68.
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
    Intermediate answer: Moderately, a fallen tree blocking part of the path would moderately affect the accessibility of edge edge_3_4 because the transportation vehicles would have to change their speed or position a lot to be sure that they avoid it.
    So the final answer is: The value is 40.
    """,
        "reasoning": """
    Because the event of a fallen tree blocking part of the path is important enough to moderately affect the accessibility of edge_3_4 so we are giving a moderate value of 40.
    """,
    },
    {
        "question": "At edge edge_N3_N4 some small branches are on the ground. Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N3_N4?
    Intermediate answer: A little, some small branches are on the ground would affect the accessibility of edge edge_N3_N4 a little because the transportation vehicles would have to slow down a little.
    So the final answer is: The value is 20.
    """,
        "reasoning": """
    Because the event of small branches being on the ground would affect the accessibility of edge_N3_N4 a little so we are giving a low value of 20.
    """,
    },
    ]
    return examples


def examples_length():
    """
    Provides a list of example scenarios with corresponding questions and answers that evaluate whether an event affects 
    the entire length of an edge or just a specific part of it. Each example includes a question, an answer with follow-up reasoning, 
    and the final True/False value indicating if the event affects the whole edge or not.

    Returns:
        list: A list of dictionaries where each dictionary contains the following keys:
            - 'question': A string that poses the scenario to evaluate if the event affects the entire edge.
            - 'answer': A string that contains follow-up questions, intermediate answers, and the final True/False value.
            - 'reasoning': A string explaining the logic behind the final True/False value assigned based on the event's impact.
    """

    examples = [
    {
        "question": "At edge edge_N2_N3 someone fell on the floor blocking it. Please provide a mandatory True/False value if the event affects the whole edge or not.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_N2_N3?
    Intermediate answer: No, because the vehicle would be affected only exactly where the person fell on the ground not for the whole length of the edge.
    So the final answer is: The answer is False, the event doesn't affect the whole edge.
    """,
        "reasoning":"""
    Because the event of someone falling down on the floor affects the accessibility of the edge only in a specific spot we are giving the value False.
    """,
    },
    {
        "question": "At edge edge_N1_N3 someone dropped their ice cream on the floor on node N2. Please provide a mandatory True/False value if the event affects the whole edge or not.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_N1_N3?
    Intermediate answer: No, because the vehicle would be affected only exactly where someon dropped their ice cream and not for the whole length of the edge.
    So the final answer is: The answer is False, the event doesn't affect the whole edge.
    """,
        "reasoning":"""
    Because the event of someone dropping their ice cream on the floor affects the accessibility of the edge only in a specific spot we are giving the value False.
    """,
    },
    {
        "question": "At edge edge_A_B a barrier blocks the entrance. Please provide a mandatory True/False value if the event affects the whole edge or not.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_A_B?
    Intermediate answer: Yes, because the path is blocked at the beggining then the whole edge is inaccessible and so it would seriously affect a vehicle for the whole length of edge edge_A_B.
    So the final answer is: The answer is True, the event affects the whole edge.
    """,
        "reasoning":"""
    Because the event a barrier blocking the entrance affects the accessibility of edge_A_B during its whole length we are giving back the value True.
    """,
    },
    {
        "question": "At edge edge_A_B the pathway is covered in thick mud due to recent rain. Please provide a mandatory True/False value if the event affects the whole edge or not.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_A_B?
    Intermediate answer: Yes, the pathway being covered in thick mud due to recent rain would seriously affect a vehicle for the whole length of edge edge_A_B.
    So the final answer is: The answer is True, the event affects the whole edge.
    """,
        "reasoning":"""
    Because the event of the pathway being covered in thick mud affects the accessibility of edge_A_B during its whole length we are giving back the value True.
    """,
    },
    {
        "question": "At edge edge_A_B someone is having a seisure. Please provide a mandatory True/False value if the event affects the whole edge or not.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_A_B?
    Intermediate answer: No, someone dying would not affect a vehicle for the whole length of the edge only in the specific spot the event happened.
    So the final answer is: The answer is False, the event doens't affect the whole edge.
    """,
        "reasoning":"""
    Because the event of someone having a seisure doesn't affect the accessibility of edge_A_B during its whole length we are giving back the value False.
    """,
    },
    {
        "question": "At edge edge_N5_N6 a burst fire hydrant floods the path. Please provide a mandatory True/False value if the event affects the whole edge or not.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_N5_N6?
    Intermediate answer: Yes, a burst fire hydrant flooding the path would affect the vehicle for the whole length of edge edge_N5_N6.
    So the final answer is: The answer is True, the event affects the whole edge.
    """,
        "reasoning": """
    Because the event of a burst fire hydrant flooding the path affects the accessibility of edge_N5_N6 during its whole length we are giving back the value True.
    """,
    },
    {
        "question": "At edge edge_N5_N6 road repair work is in on the whole path. Please provide a mandatory True/False value if the event affects the whole edge or not.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_N5_N6?
    Intermediate answer: Yes, road repair work on the whole path would affect the vehicle for the whole length of the edge edge_N5_N6.
    So the final answer is: The answer is True, the event affects the whole edge.
    """,
        "reasoning": """
    Because the event of road repair work on the whole path affects the accessibility of edge_N5_N6 during its whole length we are giving back the value True.
    """,
    },
    {
        "question": "At edge edge_N2_N4 a large fallen tree blocks the pathway. Please provide a mandatory True/False value if the event affects the whole edge or not. ",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Would a vehicle be affected by the event for the whole length of edge edge_N2_N4?
    Intermediate answer: No, a large fallen tree blocking the pathway not affect the vehicle for the whole length of the edge, only at the specific point where the tree fell.
    So the final answer is: The answer is False, the event affects the whole edge.
    """,
        "reasoning": """
    Because the event of a fallen large tree blocking the pathway doesn't affect the accessibility of edge_N2_N4 during its whole length we are giving back the value False.
    """,
    },
    ]

    return examples


def examples_time_penalty():
    """
    Provides a list of example scenarios with corresponding questions and answers that evaluate how long the accessibility of an edge 
    for transportation vehicles will be affected by an event, measured in minutes. Each example includes a question, 
    an answer with follow-up reasoning, and the final value assigned to represent the time impact on accessibility.

    Returns:
        list: A list of dictionaries where each dictionary contains the following keys:
            - 'question': A string that poses the scenario to evaluate the time penalty on the edge's accessibility.
            - 'answer': A string that contains the follow-up questions, intermediate answers, and the final value in minutes.
            - 'reasoning': A string explaining the logic behind the final time value assigned to the edge's accessibility.
    """

    examples=[
    {
        "question": "At edge edge_N2_N3 someone fell on the floor blocking it. Please provide a mandatory single value in minutes for how much time will the accessibility of the edge for the transportation vehicles be affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N2_N3?
    Intermediate answer: Moderately, someone falling down would affect the accessibility of edge edge_N2_N3 for a decent amount of time.
    So the final answer is: The value is 15 minutes.
    """,
        "reasoning":"""
    Because the event of someone falling down on the floor affects the edge accessibility of edge_N2_N3 for a decent amount of time then we are giving a medium value of 15 minutes.
    """,
    },
    {
        "question": "At edge edge_N1_N3 someone dropped their ice cream on the floor. Please provide a mandatory single value in minutes for how much time will the accessibility of the edge for the transportation vehicles be affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N1_N3?
    Intermediate answer: Very little, someone dropping their ice cream would barely affect the accessibility of edge edge_N1_N3 because transportation vehicles would not be affected by that.
    So the final answer is: The value is 1 minute.
    """,
        "reasoning":"""
    Because the event of someone dropping their ice cream on the floor is not important enough to affect the accessibility of edge edge_N1_N3 a lot so we are we are giving a very low value of 1 minute.
    """,
    },
    {
        "question": "At edge edge_A_B someone is having a heart attack. Please provide a mandatory single value in minutes for how much time will the accessibility of the edge for the transportation vehicles be affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_A_B?
    Intermediate answer: Extremely, someone having a heart attack would affect the accessibility of edge edge_A_B a lot.
    So the final answer is: The value is 60 minutes.
    """,
        "reasoning":"""
    Because the event of someone having a heart attack is important enough to affect the accessibility of edge edge_A_B a lot we are giving a high value of 60 minutes.
    """,
    },
    {
        "question": "At edge edge_N1_N3 a fallen fence blocks the access. Please provide a mandatory single value in minutes for how much time will the accessibility of the edge for the transportation vehicles be affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N1_N3?
    Intermediate answer: Extremely, a fallen fence blocking the access would affect the accessibility of edge edge_N1_N3 a lot.
    So the final answer is: The value is 50 minutes.
    """,
        "reasoning": """
    Because the event of a fallen fence blocking the access is important enough to to affect the accessibility of edge edge_N1_N3 a lot we are giving a high value of 50 minutes.
    """,
    },
    {
        "question": "At edge edge_N1_N3 some fallen debris from nearby construction on the way. Please provide a mandatory single value in minutes for how much time will the accessibility of the edge for the transportation vehicles be affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_N1_N3?
    Intermediate answer: Moderately, some fallen debris on the way would affect the accessibility of edge edge_N1_N3 because it would cause the vehicle to slow down or change it's course to avoid them.
    So the final answer is: The value is 20 minutes.
    """,
        "reasoning": """
    Because the event of some fallen debris from nearby construction being on the way is important enough to to affect the accessibility of edge edge_N1_N3 a moderate amount we are giving a medium value of 20 minutes.
    """,
    },
    {
        "question": "At edge edge_C_D a some sizeable potholes formed. Please provide a mandatory single value in minutes for how much time will the accessibility of the edge for the transportation vehicles be affected.",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How affected is the accessibility of edge edge_C_D?
    Intermediate answer: Moderately, some sizeable potholes on the way would affect the accessibility of edge edge_C_D because it would cause the vehicle to slow down or change it's course to avoid them.
    So the final answer is: The value is 25 minutes.
    """,
        "reasoning": """
    Because the event of some sizeable potholes being on the way is important enough to to affect the accessibility of edge edge_C_D a moderate amount we are giving a medium value of 25 minutes.
    """,
    },
    ]

    return examples


def get_model(model_type,approach):
    """
    Creates and returns an LLMChain model instance based on the specified model type and approach. 

    Args:
        model_type (str): The type of language model to use, which could specify different architectures or sizes (e.g., 'gpt-4', 'gpt-3.5').
        approach (str): The approach to follow, which determines the prompt template used to interact with the language model.

    Returns:
        LLMChain: An instance of the LLMChain class, configured with the specified prompt template and language model.
    """

    template = get_prompt_template(approach)
    llm = get_llm(model_type)

    return LLMChain(prompt=template, llm=llm)


def get_prompt_template(approach):
    """
    Retrieves the appropriate prompt template based on the specified approach.

    Args:
        approach (str): The approach to use for generating the prompt template. Valid values include:
            - 'fewshot': Retrieves the few-shot learning template.
            - 'zeroshot': Retrieves the zero-shot learning template.

    Returns:
        Dict[str, Any]: The prompt template corresponding to the provided approach.
    """

    template: Dict[str, Any] = {
        'fewshot': get_template_fewshot(),
        'zeroshot': get_template_zeroshot(),
    }

    return template[approach]

def get_template_fewshot():
    """
    Creates and returns a few-shot prompt template for determining the accessibility of edges in a transportation network 
    based on various events. The few-shot approach provides several examples to help guide the model's response.

    The prompt guides the model to assess how much an edge in the network is affected by an event, returning a value 
    between 0-100 based on the severity of the impact:
        - 0-25: Slightly affected
        - 25-50: Moderately affected
        - 50-75: Seriously affected
        - 75-100: Critically affected

    The model bases its answer on provided examples but only outputs a value for the last event.

    Returns:
        FewShotPromptTemplate: The few-shot prompt template that includes:
            - A list of examples (from `get_examples_factor()`),
            - A prefix to explain the task to the model,
            - A suffix prompting the model for the final value based on the input event,
            - Example formatting and input variables to structure the prompt.
    """
   
    examples = get_examples_factor()
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Example Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine how much was the provided edge affected based on how important the event given as input is. 
    The values are between 0-100 with 100 being the most affected, values between 0-25 are for events that affect the accessibility of the edge a little bit, 
    values between 25-50 are for events that moderately affect the accessibility of the edge, values between 50-75 are for events that seriously affect the accessibility of the edge 
    and values between 75-100 affect the accessibility of the edge critically. 
    Don't provide the examples in you response but base your answer on them, provide the value only for the last event.""",
    suffix="{input} Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge for the transportation vehicles is affected. Format it exactly like this: The value is X.",
    input_variables=["input"],
    example_separator='\n\n\n')

    return fewshot_template


def get_template_zeroshot():
    """
    Creates and returns a zero-shot prompt template for determining the accessibility of edges in a transportation network 
    based on the impact of an obstacle provided as input.

    The prompt instructs the model to assess how much an edge is affected by an event, returning a value between 0-100 
    based on the severity of the impact:
        - 0-25: Slightly affected
        - 25-50: Moderately affected
        - 50-75: Seriously affected
        - 75-100: Critically affected

    Unlike few-shot prompting, this template does not provide any examples. The model makes its judgment based purely on the 
    given context and input.

    Returns:
        PromptTemplate: The zero-shot prompt template that includes:
            - A predefined context explaining the task and the value range for determining edge accessibility.
            - Input variables for the template, specifically the input describing the event.
    """

    context_zero_shot = """As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine how much was the provided edge affected based on how important the obstacle given as input is. 
    The values are between 0-100 with 100 being the most affected, values between 0-25 are for events that affect the accessibility of the edge a little bit, 
    values between 25-50 are for events that moderately affect the accessibility of the edge, values between 50-75 are for events that seriously affect the accessibility of the edge 
    and values between 75-100 affect the accessibility of the edge critically. 
    {input} Please provide a mandatory single value between 0 and 100 for how much is the accessibility of the edge for the transportation vehicles is affected. Format it exactly like this: The value is X."""

    zeroshot_template = PromptTemplate(input_variables=["input"], template=context_zero_shot)

    return zeroshot_template



def get_template_length_fewshot():
    """
    Creates and returns a few-shot prompt template for determining if an event affects the entire length of an edge 
    in a transportation network or just a portion of it.

    This template is designed to help the model evaluate whether an event impacts the entire length of an edge 
    or only a specific part based on provided examples. It includes examples that illustrate how to decide and 
    format the impact assessment.

    The prompt instructs the model to determine if the accessibility of the edge is affected for the whole edge or just a part. 
    It provides a True/False answer based on the examples provided.

    The template includes examples for the model to reference and does not include examples in the final response, 
    only basing the answer on them.

    Returns:
        FewShotPromptTemplate: The few-shot prompt template that includes:
            - Examples to guide the model.
            - An example prompt for formatting the provided examples.
            - A prefix with instructions and context for evaluating edge accessibility.
            - A suffix specifying the input format for the edge impact assessment.
    """
    
    examples = examples_length()
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Example Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine whether an event would affect a vehicle for the whole edge or only for a small portion of the edge.
    Don't provide the examples in you response but base your answer on them, provide the value only for the last event.""",
    suffix="{input} Please provide a mandatory True/False value if the event affects the accessibility of the whole edge or not.",
    input_variables=["input"],
    example_separator='\n\n\n')

    return fewshot_template



def get_template_time_penalty_fewshot():
    """
    Creates and returns a few-shot prompt template for determining the time penalty imposed on a vehicle 
    when passing through an edge affected by an event in a transportation network.

    This template helps the model evaluate the time penalty based on the impact of the event on the accessibility of the edge. 
    It includes examples to illustrate how to determine and format the time penalty for the last event.

    The prompt instructs the model to provide a time penalty value in minutes, formatted as "The value is X minutes," 
    based on the examples provided.

    The template includes examples for the model to reference and does not include examples in the final response, 
    only basing the answer on them.

    Returns:
        FewShotPromptTemplate: The few-shot prompt template that includes:
            - Examples to guide the model.
            - An example prompt for formatting the provided examples.
            - A prefix with instructions and context for evaluating the time penalty based on the event.
            - A suffix specifying the input format for the time penalty assessment.
    """
    
    examples = examples_time_penalty()
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Example Question: {question}\n{answer}")

    fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix ="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
    You must determine based on the event given what time penalty should be applied to a vehicle if it passes through it.
    Don't provide the examples in you response but base your answer on them, provide the value only for the last event.""",
    suffix="{input} Please provide a mandatory single value in minutes for how much is the accessibility of the edge for the transportation vehicles is affected. Format it exactly like this: The value is X minutes.",
    input_variables=["input"],
    example_separator='\n\n\n')

    return fewshot_template


def get_llm(model_type):
    """
    Retrieves the language model based on the specified model type.

    Args:
        model_type (str): The type of the language model to retrieve. Should be one of:
            - 'openai': Retrieves the OpenAI language model.
            - 'llama2': Retrieves the Llama2 language model.
            - 'llama3': Retrieves the Llama3 language model.

    Returns:
        LLM: The language model corresponding to the specified `model_type`.
    """

    model: Dict[str, Any] = {
        'openai': get_model_openai(),
        'llama2': get_model_llama2(),
        'llama3': get_model_llama3(),
    }

    return model[model_type]


def get_model_openai():
    """
    Retrieves an OpenAI language model instance.

    Returns:
        OpenAI: An instance of the OpenAI language model initialized with the provided API key.

    """

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_model_llama2():
    """
    Retrieves a Llama2 language model instance.

    Returns:
        LLama: An instance of the Llama2 language model initialized with the specified model configuration.
    """
    return LLama(model=llama2)


def get_model_llama3():
    """
    Retrieves a Llama3 language model instance.

    Returns:
        LLama: An instance of the Llama3 language model initialized with the specified model configuration.
    """

    return LLama(model=llama3)


def invoke_llm(prompt, model_type='llama2', approach='zeroshot'):
    """
    Invokes a language model to generate a response based on the provided prompt.

    Args:
        prompt (str): The input prompt that will be fed to the language model for generating a response.
        model_type (str, optional): The type of the language model to use. Defaults to 'llama2'.
        approach (str, optional): The approach to use with the model. Defaults to 'zeroshot'.

    Returns:
        str: The text response generated by the language model.
    """

    G=load_edges()
    new_graph=get_model(model_type,approach)
    answer=new_graph.invoke(prompt)

    return answer["text"]


def parse_output(output):
    """
    Parses a string output to determine if it contains a boolean value (`True` or `False`).

    Args:
        output (str): The string output to be parsed.

    Returns:
        bool or None: Returns `True` if "True" is found in the output, `False` if "False" is found, or `None` if neither is found.
    """

    pattern = r"[T|t]rue|[F|f]alse"
    result = re.findall(pattern, output)
    if len(result)==0:
        result_bool = None
    elif result[0].lower()=='true':
        result_bool=True
    elif result[0].lower()=='false':
        result_bool=False
    else:
        result_bool = None
    
    return result_bool

#Fucntion for callling the LLM for Dynamic Weights
def invoke_llm_chain(prompt, model_type='openai', approach='fewshot'):
    """
    Invokes a language model chain to assess the impact of an event on the accessibility of an edge.

    This function first determines whether an event affects the entire length of the edge or only a portion of it.
    Based on this assessment, it invokes a second language model chain to evaluate the impact either in terms of time penalty
    or as a factor affecting the accessibility.

    Args:
        prompt (str): The input prompt detailing the event and edge information.
        model_type (str, optional): The type of language model to use ('openai', 'llama2', 'llama3'). Defaults to 'openai'.
        approach (str, optional): The approach for few-shot or zero-shot learning. Defaults to 'fewshot'.

    Returns:
        tuple: A tuple containing three elements:
            - `answer` (str): The final output from the language model chain, which could be a time penalty in minutes or a factor.
            - `output_length` (str): The initial output indicating if the event affects the whole length of the edge.
    """

    G=load_edges()
    template = get_template_length_fewshot()
    llm = get_llm(model_type)
    length_chain = LLMChain(prompt=template, llm=llm)

    output_length = length_chain.invoke(prompt)
    print("Affected for the whole length: ",output_length["text"])

    output = parse_output(output_length["text"])

    if output==False:
        print("Minutes")
        output='minutes'
        template = get_template_time_penalty_fewshot()
    else:
        print("Factor")
        output='factor'
        template = get_template_fewshot()

    llm=get_llm(model_type)
    chain = LLMChain(prompt=template, llm=llm)

    answer=chain.invoke(prompt)

    return answer["text"], output_length["text"], output


def parsing_llm_result(answer):
    """
    Parses the output from a language model to extract and clean information about edges whose weights have changed to infinity.

    Args:
        answer (str): The output text from the language model containing information about edge weights.

    Returns:
        list: A list of tuples representing the edges whose weights have changed to infinity, with each tuple containing
              two integer values.
    """

    pattern = r"\([`']?\d+[`']?, [`']?\d+[`']?\)"
    removed_edges = re.findall(pattern, answer, re.DOTALL)
    
    removed_edges_cleaned = []

    for removed_edge in removed_edges:
        cleaned = removed_edge.strip("'´")
        cleaned = ast.literal_eval(cleaned)
        removed_edges_cleaned.append(cleaned)

    print("List of edges which weights are changed to infinity:", removed_edges)

    return removed_edges



def load_edges():
    """
    Loads edge data from a CSV file and returns a list of edges as tuples.

    Returns:
        list: A list of tuples where each tuple represents an edge with two node identifiers.
    """
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','_00_Resources','edges_UH_Graph.csv'))
    except:
        df = pd.read_csv(os.path.join('_00_Resources','edges_UH_Graph.csv'))


    edges_list = [(f'{row[0]}', f'{row[1]}') for _, row in df.iterrows()]
    return edges_list

def parse_output_weights(output):
    """
    Extracts a numeric value from the output string, typically representing a weight.

    Args:
        output (str): The output string from which to extract the numeric value.

    Returns:
        int or None: The extracted numeric value as an integer, or None if no valid number is found.
    """
    #try a pattern for The value is X
    pattern = r"[V|v]alue[^\d]{0,20}\d{1,3}"
    result = re.findall(pattern, output)

    if len(result)==0:
        pattern = r"[^\d]{2,5}(\d{1,3})(?:[^\d]{2,5}|\.)"
        result = re.findall(pattern, output)

    if len(result)==0:
        return None

    final_pattern = r"\d{1,3}"
    result[0] = re.findall(final_pattern, result[0])[0]
    result_number = int(result[0])

    return result_number


def main(ref_routing):
    """
    Main function to handle user input, invoke the LLM, parse the output, and update the graph.

    Args:
        ref_routing (Routing): An instance of the Routing class which contains the graph to be updated.
    """

    prompt = input("Enter your prompt: ")
    output = invoke_llm(prompt)
    print(output)

    # Parse the output
    parsed_res = parsing_llm_result(output)
    
    print("Value:",parse_output_weights(output))
    print(type(parse_output_weights(output)))

    # Update graph in the routing
    ref_routing.graph = set_weights_to_inf(ref_routing.graph, parsed_res)
