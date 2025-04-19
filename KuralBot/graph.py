from dotenv import load_dotenv
import os
import nodes
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from typing import Annotated, TypedDict, List
from langchain.docstore.document import Document
from langgraph.graph.message import add_messages
import warnings

warnings.filterwarnings('ignore')

load_dotenv("config.env")

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        context: context
    """
    messages: Annotated[list, add_messages]
    web_search: bool = False
    docs: List[Document]
    transformed_question: str 
    askUser: bool = False
    translated_generation: str = ""
    input_msg: str = ""
    lang: str = "English"
    Quesloops: int = 0 # Number of times the question has been asked to user must be less or equal to 3.

workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("websearch", nodes.web_search)  # web search
workflow.add_node("retrieve", nodes.retrieve)  # retrieve
workflow.add_node("generate", nodes.generate)  # generatae
workflow.add_node("transform", nodes.transform_query)
workflow.add_node("casual_response", nodes.casual_generate)
workflow.add_node("question_generator", nodes.question_generator)
workflow.add_node("translator-in", nodes.translate)
workflow.add_node("translator-out", nodes.translate)


# Build graph

workflow.add_edge(START, "translator-in")
workflow.add_edge("translator-in", "transform")

workflow.add_conditional_edges(
    "transform",
    nodes.route_question,
    {
        "web_search": "websearch",
        "vectorstore": "retrieve",
        "generator":"casual_response"
    },
)


workflow.add_conditional_edges(
    "retrieve",
    nodes.decide_to_generate,
    {
        "asktoUser": "question_generator",
        "generate": "generate",
    },
)

workflow.add_edge("websearch", "generate")
workflow.add_edge("casual_response", "translator-out")
workflow.add_edge("generate", "translator-out")
workflow.add_edge("question_generator", "translator-out")

workflow.add_edge("translator-out", END)

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)