import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    trim_messages,
)
from utils import make_prompt, format_docs, format_messages
from langchain_huggingface import HuggingFaceEmbeddings
from prompts import prompts

from navrasa_ccm import CCM_navarasa
from mistral_ccm import CCM_mistral

from langgraph.graph import END, START
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
import warnings
import re
warnings.filterwarnings('ignore')

llm_nav = CCM_navarasa()
llm = CCM_mistral()

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = FAISS.load_local("new_index", embeddings, allow_dangerous_deserialization=True)

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_relevance_scores(query, k=2))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs

### Translator-----------------------------------------------------------------------------------
translate_prompt = make_prompt(prompts["translate"]["system"], prompts["translate"]["human"])

translator = translate_prompt | llm_nav | StrOutputParser()

def translate(state):
    """
    Translate the query to english/hindi.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates State with a translated message.
    """

    lang = state["lang"]
    
    if lang == "Hindi":
        input_message = state["messages"][-1].content
        translated_msg = translator.invoke({"sentence": input_message, "lang": lang})
        lang = "English"
        return {"translated_generation":translated_msg, "lang": lang}

    else: 
        input_message = state["input_msg"]
        translated_msg = translator.invoke({"sentence": input_message, "lang": "English"})
        lang = "Hindi"
        return {"messages": [HumanMessage(translated_msg)], "lang": lang}

### Transform----------------------------------------------------------------------------------
transform_prompt = make_prompt(prompts["transform"]["system"], prompts["transform"]["human"])

question_rewriter = transform_prompt | llm | StrOutputParser()

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates state with a re-phrased question
    """
    question = state["messages"][-1].content
    messages = state["messages"]
    chat_history = format_messages(messages[:-1])

    if state.get("Quesloops") is None:
        state["Quesloops"] = 0

    # summary = ""
    # if len(messages)  > 1:
    #     summary = summarizer.invoke({"chat_history": format_messages(messages[:-1])})

    transformed_question = question
    if len(messages) > 1:
        transformed_question = question_rewriter.invoke({"question": question, "chat_history":chat_history})

    # transformed_question = translator.invoke({"sentence": transformed_question, "lang": "english"})

    return {"transformed_question": transformed_question, "Quesloops": state["Quesloops"]}

### Question Generator----------------------------------------------------------------------------------------
ques_prompt = make_prompt(prompts["question_generate"]["system"], prompts["question_generate"]["human"])

q_genrator = ques_prompt | llm | StrOutputParser()

def question_generator(state):
    """
    Generate the counter question to user.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    state["Quesloops"] = state["Quesloops"] + 1

    transformed_question = state["transformed_question"]
    # docs = state["docs"]

    # context = format_docs(docs)

    # detail = q_gen.invoke({"question": transformed_question, "context": context})
    new_question = q_genrator.invoke({"question":transformed_question})

    # new_question = translator.invoke({"sentence": new_question, "lang": "hindi"})
    
    return {"Quesloops": state["Quesloops"], "messages": [AIMessage(new_question)]}

### Expert response Generator-------------------------------------------------------------------------------------------
expert_prompt = make_prompt(prompts["expert_generate"]["system"], prompts["expert_generate"]["human"], True)

ans_generator = expert_prompt | llm | StrOutputParser()

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the messages list with new message
    """

    question = state["messages"][-1].content
    docs = state["docs"]
    messages = state["messages"]

    # messages = trim_messages(
    #             messages,
    #             max_tokens = 200,
    #             strategy="last",
    #             token_counter = tiktoken_counter,
    #             end_on = HumanMessage,
    #             start_on = HumanMessage,
    #             include_system = False
    # )

    context = format_docs(docs)
   
    generation = ans_generator.invoke({"chat_history":messages[:-1], "context": context, "question": question})

    generation = re.sub(r"</?answer>", "", generation)
    
    return {"messages":[AIMessage(generation)], "docs": [], "web_search":False, "askUser":False, "Quesloops":0}

### Casual response Generator--------------------------------------------------------------------------------
casual_prompt = make_prompt(prompts["casual_generate"]["system"], prompts["casual_generate"]["human"], True)

cas_ans_generator = casual_prompt | llm | StrOutputParser()

def casual_generate(state):
    """
    Generates LLM response.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the messages list with new message 
    """
    question = state["messages"][-1].content
    messages = state["messages"]

    # messages = trim_messages(
    #             messages,
    #             max_tokens = 200,
    #             strategy="last",
    #             token_counter = tiktoken_counter,
    #             end_on = HumanMessage,
    #             start_on = HumanMessage,
    #             include_system = False
    
    # )

    generation = cas_ans_generator.invoke({"chat_history":messages[:-1], "question": question})

    return {"messages":[AIMessage(generation)], "Quesloops":0}

### Retrieval-------------------------------------------------------------------------------------------------
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    transformed_question = state["transformed_question"]
    askUser = state["askUser"]

    # Retrieval
    docs = retriever.invoke(transformed_question)
    
    filtered_docs = []

    if state["Quesloops"] >= 3:
        askUser = False
        filtered_docs = docs
        print("---QUESTION LIMIT EXCEEDED---")

    elif len(docs) > 0:

        for doc in docs:
            if doc.metadata['score'] > 0.6:
                filtered_docs.append(doc)
            print(doc.metadata['score'])
            

        # Document relevant
        if len(filtered_docs) > 0:
            askUser = False
            print("---GRADE: DOCUMENT RELEVANT---")

        # Not relevant
        else:
            askUser = True
            filtered_docs = docs
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    # Document not retrived
    else:
        askUser = True
        filtered_docs = docs
        print("---DOCUMENT NOT RETRIEVED---")

    return {"docs": filtered_docs, "askUser": askUser, "Quesloops": state["Quesloops"]}



### Web search------------------------------------------------------------------------------------------------
web_search_tool = TavilySearchResults(k=2)

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    # print("---WEB SEARCH---")
    transformed_question = state["transformed_question"]
    
    
    # Web search
    try:
        docs = web_search_tool.invoke({"query": transformed_question})

        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

    except:
        web_results = Document(page_content="")
    
    return {"docs": [web_results]}


### Router and Decider (Function for Conditional Edges)-------------------------------------------------------------

router_prompt = make_prompt(prompts["route"]["system"], prompts["route"]["human"])

ques_router = router_prompt | llm_nav | StrOutputParser()

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    transformed_question = state["transformed_question"]
    
    source = ques_router.invoke({"question": transformed_question})

    if "web_search" in source.lower():
        return "web_search"
    
    elif "vectorstore" in source.lower():
        return "vectorstore"
    
    else:
        return "generator"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    askUser = state["askUser"]

    if askUser == True:
        return "asktoUser"
    else:
        return "generate"

