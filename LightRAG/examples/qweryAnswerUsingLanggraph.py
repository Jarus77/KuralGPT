### Imports ###
import os
import logging
from typing import List
from typing_extensions import TypedDict
import pprint

# LangChain and LightRAG imports
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

### Constants ###
WORKING_DIR = "/nfs/kundeshwar/surajKuralGPT/s2/kuralGPT/dickensKural"

### Logging Configuration ###
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

### Ensure Working Directory Exists ###
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

### LightRAG Setup ###
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2m",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

### Graph State Definition ###
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The input question.
        generation: LLM generation result.
        documents: List of retrieved documents.
    """
    question: str
    generation: str
    documents: List[str]

### Node Functions ###
def retrieve(state):
    """
    Retrieve documents based on the question in the state.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with retrieved documents.
    """
    print("--- RETRIEVE ---")
    question = state["question"]
    answer = rag.query(question)
    print(answer)
    return {"documents": answer, "question": question}

def route_question(state):
    """
    Route the question to the appropriate node.

    Args:
        state (dict): The current graph state.

    Returns:
        str: Next node to call.
    """
    return "vectorstore"

### Workflow Setup ###
workflow = StateGraph(GraphState)

# Add nodes to the workflow
workflow.add_node("retrieve", retrieve)

# Define the conditional edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": "retrieve",
    },
)

# Add edge to end the workflow
workflow.add_edge("retrieve", END)

# Configure memory saver
memory = MemorySaver()

# Compile the workflow
app = workflow.compile(checkpointer=memory)

### Run Workflow ###
config = {"configurable": {"thread_id": "1"}}
inputs = {"question": "Which kural talks about life?"}

for output in app.stream(inputs, config):
    for key, value in output.items():
        # Display node output
        pprint.pprint(f"Node '{key}':")
    pprint.pprint("\n---\n")

