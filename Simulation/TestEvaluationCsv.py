from langsmith import Client
from dotenv import load_dotenv
import os
import re
from Playground_LLM_Dacian import invoke_llm

load_dotenv()
client=Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

def get_output_file_fewshot():
    try:
        f = open(os.path.join('..','Playground_LLM','EvaluationDatasetFewShot.csv'),'w')
    except:
        f = open(os.path.join('Playground_LLM','EvaluationDatasetFewShot.csv'),'w')
    return f

def test_fewshot():
    edge_ids,tests=load_tests()
    
    f=get_output_file_fewshot()
    f.write('action;edgeUsable;LLMAnswer;LLMOutput\n')

    for test,edge in zip(tests,edge_ids):
       output=invoke_llm(f'At edge {edge} {test[0]}')
       print(output)
       parsed_output=parse_output(output)
       f.write(f'{test[0]};{test[1]};{parsed_output};{"".join(output.splitlines())}\n')
    f.close()


def load_tests():
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join('..','Playground_Arved','csv','edges_UH_Graph_Ids.csv'))
    except:
        df = pd.read_csv(os.path.join('Playground_Arved','csv','edges_UH_Graph_Ids.csv'))

    try:
        df_test = pd.read_csv(os.path.join('..','Playground_LLM','EvaluationDataset.csv'),delimiter=';')
    except:
        df_test = pd.read_csv(os.path.join('Playground_LLM','EvaluationDataset.csv'),delimiter=';')
    print(df_test)
    edge_ids = [f'{row[0]}' for _,row in df.iterrows()]
    tests = [ (test[0],test[1]) for _,test in df_test.iterrows()]
    print(type(tests[0][0]))
    print(len(tests[0]))
    return (edge_ids,tests)


def parse_output(output):
    pattern = r"[T|t]rue|[F|f]alse"
    result = re.findall(pattern, output)

    if result[0].lower()=='true':
        result_bool=True
    elif result[0].lower()=='false':
        result_bool=False
    else:
        result_bool=None

    return result_bool


if __name__ == "__main__":
    test_fewshot()