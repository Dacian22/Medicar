import sys

sys.path.append('/Users/paulkoenig/WebstormProjects/medicar/_01_Simulation/LLM_Edge_Usability.py')
from typing import Any, Dict
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
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
            "question": "Someone fell on the floor on edge edge_N1_N2. Please answer with True if the edge is usable and False if the edge is not usable.",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that edge N1_N2 is not accessible anymore?
    Intermediate answer: Yes, someone falling down would make edge N1_N2 inaccessible.
    So the final answer is: List of edges that have to be removed: ('N1','N2'). False the edge is not usable.
    """,
            "reasoning": """
    Because the event of someone falling down on the floor is important enough to block access to the edge N1_N2 so we are removing this edge('N1','N2').
    """,
        },
        {
            "question": "Someone dropped their ice cream on the floor on edge edge_N2_N3. Please answer with True if the edge is usable and False if the edge is not usable.",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that edge edge_N2_N3 is not accessible anymore?
    Intermediate answer: No, someone dropping their ice cream would not make edge edge_N2_N3 inaccessible.
    So the final answer is: List of edges that have to be removed: []. True the edge is usable.
    """,
            "reasoning": """
    Because the event of someone dropping their ice cream on the floor is not important enough to block access to the edge edge_N2_N3 so we are not removing this edge, so NO edges are affected.
    """,
        },
        {
            "question": "Someone is having a heart attack on edge edge_N5_N6. Please answer with True if the edge is usable and False if the edge is not usable.",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that edge edge_N5_N6 is not accessible anymore?
    Intermediate answer: Yes, someone having a heart attack would make edge edge_N5_N6 inaccessible.
    So the final answer is: List of edges that have to be removed: ('N5','N6'). False the edge is not usable.
    """,
            "reasoning": """
    Because the event of someone dying is important enough to block access to the edge edge_N5_N6 so we are removing this edge ('N5','N6').
    """,
        },
        {
            "question": "Someone dropped their papers on edge edge_N1_N6. Please answer with True if the edge is usable and False if the edge is not usable.",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is the event important enough so that edge edge_N1_N6 is not accessible anymore?
    Intermediate answer: No, someone dropping their papers would not make edge edge_N1_N6 inaccessible.
    So the final answer is: List of edges that have to be removed: []. True, the edge is usable.
    """,
            "reasoning": """
    Because the event of someone dropping their papers is not important enough to block access to the edge edge_N5_N6 so we are not removing this edge, so NO edges are affected.
    """,
        },
        {
            "question": "A group of people are chatting on edge edge_N3_N2. Please answer with True if the edge is usable and False if the edge is not usable.",
            "answer": """
        Are follow up questions needed here: Yes.
        Follow up: Is the event important enough so that edge edge_N3_N2 is not accessible anymore?
        Intermediate answer: No, people chatting would not make edge edge_N3_N2 inaccessible.
        So the final answer is: List of edges that have to be removed: []. True the edge is usable.
        """,
            "reasoning": """
        Because the event of people chatting is not important enough to block access to the edge edge_N3_N2 so we are not removing this edge, so NO edges are affected.
        """,
        },
        
    ]
    return examples


def get_model(model_type, approach):
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
        prefix="""As a professional graph modeler, you're tasked with determining the accessibility of edges in a transportation network. 
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
        df = pd.read_csv(os.path.join('..', '_00_Resources', 'edges_UH_Graph.csv'))
    except:
        df = pd.read_csv(os.path.join('_00_Resources', 'edges_UH_Graph.csv'))

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

    G = load_edges()
    new_graph = get_model(model_type, approach)
    answer = new_graph.invoke(prompt)

    return answer["text"]
