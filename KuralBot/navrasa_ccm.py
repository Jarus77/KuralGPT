from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)
import re

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel
)
from langchain_core.output_parsers.base import OutputParserLike

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import (
    BaseModel
)
from langchain_core.runnables import Runnable, RunnableMap
from langchain_core.tools import BaseTool

from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from dotenv import load_dotenv
load_dotenv("config.env")


login(token=os.getenv("HF_API_KEY"))

device = "cuda:1" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0",
    load_in_4bit = False,
)

model.to(device)

tokenizer = AutoTokenizer.from_pretrained("Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0")


def convert_messages_to_navarasa_template(messages):
    instruction = messages[0].content

    temp = f"""
### Instruction:
{instruction}
"""

    for i in range(1, len(messages)):
        input_str = ""
        if type(messages[i]).__name__ == "HumanMessage":
            content = messages[i].content

            input_str = f"""
### Input:
{content}
"""     
        if type(messages[i]).__name__ == "AIMessage":
            content = messages[i].content

            input_str = f"""
### Response:
{content}
"""
        temp = temp + input_str

    if type(messages[-1]).__name__ == "HumanMessage":
       input_str = f"""
### Response:
"""    
       temp = temp + input_str
    return temp

def extract_last_response(chat_log):
    # Split the chat log into individual parts
    parts = re.split(r"### (?:Input|Response):", chat_log)

    # The last part should contain the last response
    last_response = parts[-1].strip()

    return last_response

def navarasa(messages):

    input_text = convert_messages_to_navarasa_template(messages)
    # print("****************************************************************************************************")
    # print(input_text)
    # print("****************************************************************************************************")
        
    inputs = tokenizer([input_text], return_tensors = "pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens = 300, use_cache = True)
    response = tokenizer.batch_decode(outputs)[0]
    # print(response)

    # responses = re.findall(r'### Response:\n(.*?)\n###|### Response:\n(.*?)<eos>', response, re.DOTALL)

    # # print(responses)
    # # Extract the last response
    # last_response = responses[-1][0] or responses[-1][1]
    # print(last_response)
    last_response = extract_last_response(response)
    last_response = last_response.replace("<eos>", "")
    return last_response
    

class CCM_navarasa(BaseChatModel):
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Replace this with actual logic to generate a response from a list
        # of messages.
        # print("*************************************************************************")
        # print(messages)
        # print("*************************************************************************")

        
        res = navarasa(messages)
            
        message = AIMessage(
            content=res,
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """

        # print("=========================================================================================================")
        # print(prompt)
        # print("=========================================================================================================")

        res = self._generate(messages).generations[0].text
        # print("****************************************************************************************************")
        # print(res)
        # print("****************************************************************************************************")
        for token in res:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": "llama-3",
        }
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # print(formatted_tools)
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "any", "none")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, dict) and (len(formatted_tools) != 1):
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, dict) and (
                formatted_tools[0]["function"]["name"]
                != tool_choice["function"]["name"]
            ):
                raise ValueError(
                    f"Tool choice {tool_choice} was specified, but the only "
                    f"provided tool was {formatted_tools[0]['function']['name']}."
                )
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        # print(type(super().bind(tools=formatted_tools, **kwargs)))
        # print(super().bind(tools=formatted_tools, **kwargs))
        return super().bind(tools=formatted_tools, **kwargs)

