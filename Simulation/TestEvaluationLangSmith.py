from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from dotenv import load_dotenv
import re
import os
from medicar.Simulation.LLM_FewShot import get_model_testing
load_dotenv()
client=Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

@run_evaluator
def regex_evaluator(run: Run, example: Example | None = None):
    print(example)
    model_outputs = run.outputs["text"]
    
    pattern = r"[T|t]rue|[F|f]alse"
    result = re.findall(pattern, model_outputs)

    if result[0].lower()=='true':
        result_bool=True
    elif result[0].lower()=='false':
        result_bool=False
    else:
        result_bool=None
    return EvaluationResult(key="regex_evaluator", score=(example.outputs['edgeUsable']==result_bool and len(result)==1))


def evaluate_fewshot():
    evaluation_config = RunEvalConfig(
    custom_evaluators = [regex_evaluator],
    )


    client.run_on_dataset(
        dataset_name="Evaluation Database",
        llm_or_chain_factory=get_model_testing(),
        evaluation=evaluation_config,
    )

if __name__ == "__main__":
    evaluate_fewshot()