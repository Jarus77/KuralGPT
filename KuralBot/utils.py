import tiktoken
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    SystemMessage
)

from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """Approximately reproduce https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    For simplicity only supports str Message.contents.
    """
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens

def format_messages(messages):
    temp = """"""
    for i in range(0, len(messages)):
        input_str = ""
        if type(messages[i]).__name__ == "HumanMessage":
            content = messages[i].content

            input_str = f"""User - {content}\n"""   

        if type(messages[i]).__name__ == "AIMessage":
            content = messages[i].content

            input_str = f"""Assistant - {content}\n"""
    
        temp = temp + input_str

    return temp

def make_prompt(system, human, chat_history=False):

    if chat_history == True:
        prompt =  ChatPromptTemplate.from_messages(
                    [
                        ("system",system),    
                        MessagesPlaceholder("chat_history", optional=True),
                        ("human", human)
                    ]
                )
        
        return prompt

    else:

        prompt =  ChatPromptTemplate.from_messages(
                    [
                        ("system",system),    
                        ("human", human)
                    ]
                )
    return prompt 


