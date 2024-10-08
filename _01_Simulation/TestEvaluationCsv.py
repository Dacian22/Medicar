import os
import re

from dotenv import load_dotenv
from langsmith import Client

from LLM_Dynamic_Weights import invoke_llm_chain
from LLM_Edge_Usability import invoke_llm

load_dotenv(override=True)
client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))


def get_output_file_openai_fewshot():
    """Opens and returns the evaluation dataset file for few-shot OpenAI evaluation."""

    f = open(os.path.join(os.getenv("RESOURCES"), "eval-res-edge-usability-openai-fewshot.csv"))
    return f


def get_output_file_llama2_fewshot():
    """Opens and returns the evaluation dataset file for few-shot Llama2 evaluation."""

    f = open(os.path.join(os.getenv("RESOURCES"), "eval-res-edge-usability-llama2-fewshot.csv"))
    return f


def get_output_file_llama2_zero_shot():
    """Opens and returns the evaluation dataset file for zero-shot Llama2 evaluation."""

    f = open(os.path.join(os.getenv("RESOURCES"), 'eval-res-edge-usability-llama2-zeroshot.csv'), 'w')
    return f


def get_output_file_llama3_zero_shot():
    """Opens and returns the evaluation dataset file for zero-shot Llama3 evaluation."""

    f = open(os.path.join(os.getenv("RESOURCES"), 'EvaluationDatasetLLama3ZeroShot.csv'), 'w')
    return f


def get_output_file_llama2_zeroshot_weights():
    """Opens and returns the evaluation dataset file for zero-shot with weights Llama2 evaluation."""

    f = open(os.path.join(os.getenv("RESOURCES"), 'EvaluationDatasetLLama2Weights.csv'), 'w')
    return f


def get_output_file_openai_fewshot_weights():
    """Opens and returns the evaluation dataset file for few-shot with weights OpenAI evaluation."""

    f = open(os.path.join(os.getenv("RESOURCES"), 'eval-res-dynamic-openai-fewshot.csv'), 'w')
    return f


from typing import Any, Dict


def get_output_file(file):
    """
    Returns a file object based on the specified model and evaluation approach.

    Args:
        file (str): A key representing the model type and evaluation approach. 
                    Options include 'openai_fewshot', 'llama2_fewshot', 
                    'llama2_zeroshot', 'llama3_zeroshot', and 
                    'llama2_zeroshot_weights'.

    Returns:
        file object: A file object representing the opened evaluation dataset file.
    """

    function: Dict[str, Any] = {
        'openai_fewshot': get_output_file_openai_fewshot(),
        'llama2_fewshot': get_output_file_llama2_fewshot(),
        'llama2_zeroshot': get_output_file_llama2_zero_shot(),
        'llama3_zeroshot': get_output_file_llama3_zero_shot(),
        'llama2_zeroshot_weights': get_output_file_llama2_zeroshot_weights(),
    }

    return function[file]


def test_llm(file, model, approach, parser):
    """
    Tests the performance of a language model (LLM) by feeding test cases to it and comparing 
    its responses with expected results.

    Args:
        file (str): A key representing the file to write results to. This will determine the output 
                    file based on the test configuration.
        model (str): The type of LLM being tested (e.g., 'openai', 'llama2', etc.).
        approach (str): The approach used by the LLM ('fewshot', 'zeroshot', etc.).
        parser (str): The parser function used to extract relevant information from the LLM output.
    """

    edge_ids, tests = load_tests()

    f = get_output_file(file)
    f.write('action;edgeUsable;LLMAnswer\n')

    none_answers = 0

    parse = get_output_parser(parser)

    for test, edge in zip(tests, edge_ids):
        output = invoke_llm(f'At edge {edge} {test[0]}', model, approach)
        print(output)
        parsed_output = parse(output)
        if parsed_output == None:
            none_answers += 1
        else:
            f.write(f'{test[0]};{test[1]};{parsed_output}\n')
    f.close()

    print(none_answers)


def load_tests():
    """
    Loads test data from CSV files containing edge IDs and corresponding test cases for 
    evaluating the performance of a language model (LLM).

    Returns:
        tuple: A tuple containing:
            - edge_ids (list): A list of edge IDs as strings.
            - tests (list): A list of tuples, where each tuple contains:
                - The test case action (e.g., obstacle description).
                - The expected result (True/False indicating if the edge is usable).
    """
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..', '_00_Resources', 'edges_UH_Graph_Ids.csv'))
    except:
        df = pd.read_csv(os.path.join('_00_Resources', 'csv', 'edges_UH_Graph_Ids.csv'))

    df_test = pd.read_csv(os.path.join(os.getenv("RESOURCES"), 'EvaluationDataset.csv'), delimiter=';')

    print(df_test)
    edge_ids = [f'{row[0]}' for _, row in df.iterrows()]
    tests = [(test[0], test[1]) for _, test in df_test.iterrows()]
    print(type(tests[0][0]))
    print(len(tests[0]))
    return (edge_ids, tests)


def get_output_parser(parser):
    """
    Returns the appropriate parsing function based on the specified parser type.

    Args:
        parser (str): A string specifying the type of parser to use. The available options are:
            - 'bool': This uses the `parse_output` function to interpret the output as a boolean value.
            - 'weight': This uses the `parse_output_weights` function to interpret the output related to weights (e.g., numerical values for edge weights).

    Returns:
        function: The corresponding parsing function based on the `parser` argument.
    """

    function: Dict[str, Any] = {
        'bool': parse_output,
        'weight': parse_output_weights,
    }

    return function[parser]


def parse_output(output):
    """
    Parses the output from a language model and determines if the result is `True` or `False`.

    The function first checks for the presence of 'True' or 'False' in the output using a regular expression. If neither is found, it looks for the phrase "not usable" as an indication that the result is `False`. Otherwise, it assumes the result is `True`.

    Args:
        output (str): The output text from a language model that is being parsed.

    Returns:
        bool or None: Returns `True` if the output indicates a positive or usable result, 
        `False` if the output indicates a negative or unusable result, and `None` if the 
        result cannot be determined.
    """

    pattern = r"[T|t]rue|[F|f]alse"
    result = re.findall(pattern, output)
    if len(result) == 0:
        result_bool = None
    elif result[0].lower() == 'true':
        result_bool = True
    elif result[0].lower() == 'false':
        result_bool = False
    else:
        result_bool = None

    if result_bool == None:
        pattern = r"not usable"  # False
        result = re.findall(pattern, output)
        if len(result) == 0:
            result_bool = True
        else:
            result_bool = False

    return result_bool


def parse_output_weights(output):
    """
    Parses the output from a language model to extract a weight value, typically used to represent 
    how much an edge's accessibility is affected (on a scale of 0-100).

    The function first attempts to find a pattern of the form "The value is X", where X is a number. 
    If this pattern is not found, it falls back to searching for a number in the output using alternative patterns.

    Args:
        output (str): The output text from a language model that is being parsed.

    Returns:
        int or None: Returns the extracted integer value between 0 and 100 if found, or `None` if the result cannot be determined.
    """

    # try a pattern for The value is X
    pattern = r"[V|v]alue[^\d]{0,20}\d{1,3}"

    result = re.findall(pattern, output)
    if len(result) == 0:
        pattern = r"[^\d]{2,5}(\d{1,3})(?:[^\d]{2,5}|\.)"
        result = re.findall(pattern, output)

    if len(result) == 0:
        return None

    final_pattern = r"\d{1,3}"
    result[0] = re.findall(final_pattern, result[0])[0]
    result_number = int(result[0])

    return result_number


def test_llm_weights(model="openai", approach="fewshot"):
    """Tests a language model's ability to evaluate edge weights and length dependency based on a given scenario."""

    edge_ids, tests = load_tests()
    f = get_output_file_openai_fewshot_weights()
    f.write('action;lengthDependency;LLMAnswer;Value\n')

    none_answers = 0

    parse = parse_output_weights

    for test, edge in zip(tests, edge_ids):
        output, output_bool = invoke_llm_chain(f'At edge {edge} {test[0]}', model, approach)
        print(output)
        parsed_output = parse(output)
        f.write(f'{test[0]};{test[1]};{output_bool};{parsed_output}\n')
    f.close()

    print(none_answers)


if __name__ == "__main__":
    test_llm_weights('openai', 'fewshot')
