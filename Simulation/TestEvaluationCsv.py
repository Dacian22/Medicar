from langsmith import Client
from dotenv import load_dotenv
import os
import re
from Playground_LLM_Dacian import invoke_llm


load_dotenv(override=True)
client=Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

def get_output_file_openai_fewshot():
    f = open(os.path.join(os.getenv("RESOURCES"), "EvaluationDatasetFewShot.csv"))
    return f

def get_output_file_llama2_fewshot():
    f = open(os.path.join(os.getenv("RESOURCES"), "EvaluationDatasetLLama2.csv"))
    return f

def get_output_file_llama2_zero_shot():
    f = open(os.path.join(os.getenv("RESOURCES"),'EvaluationDatasetLLama2ZeroShot.csv'),'w')
    return f

def get_output_file_llama3_zero_shot():
    f = open(os.path.join(os.getenv("RESOURCES"),'EvaluationDatasetLLama3ZeroShot.csv'),'w')
    return f


from typing import Any, Dict

def get_output_file(file):
    function: Dict[str, Any] = {
        'openai_fewshot': get_output_file_openai_fewshot(),
        'llama2_fewshot': get_output_file_llama2_fewshot(),
        'llama2_zeroshot': get_output_file_llama2_zero_shot(),
        'llama3_zeroshot': get_output_file_llama3_zero_shot()
    }

    return function[file]

def test_llm(file,model,approach):
    edge_ids,tests=load_tests()
    
    f=get_output_file(file)
    f.write('action;edgeUsable;LLMAnswer\n')

    none_answers=0
    
    invoke = invoke_llm(model,approach)

    for test,edge in zip(tests,edge_ids):
       output=invoke(f'At edge {edge} {test[0]}')
       print(output)
       parsed_output=parse_output(output)
       if parsed_output==None:
           none_answers+=1
       else:
           f.write(f'{test[0]};{test[1]};{parsed_output}\n')
    f.close()

    print(none_answers)




def load_tests():
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','Playground_Arved','csv','edges_UH_Graph_Ids.csv'))
    except:
        df = pd.read_csv(os.path.join('Playground_Arved','csv','edges_UH_Graph_Ids.csv'))

    df_test = pd.read_csv(os.path.join(os.getenv("RESOURCES"),'EvaluationDataset.csv'),delimiter=';')

    print(df_test)
    edge_ids = [f'{row[0]}' for _,row in df.iterrows()]
    tests = [ (test[0],test[1]) for _,test in df_test.iterrows()]
    print(type(tests[0][0]))
    print(len(tests[0]))
    return (edge_ids,tests)


def parse_output(output):
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

    if result_bool == None:
        pattern = r"not usable" # False
        result = re.findall(pattern, output)
        if len(result)==0:
            result_bool = True
        else:
            result_bool = False

    return result_bool


if __name__ == "__main__":
    test_llm('llama2_zeroshot','zeroshot','llama2')