from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langserve import RemoteRunnable




from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

class LLama(LLM):
    model: RemoteRunnable
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        answer=self.model.invoke(prompt)
        return answer.content

    

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "LLama2 Uni Server",
        }

    @property
    def _llm_type(self) -> str:
        return "english"