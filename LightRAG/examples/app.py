import os
import logging
from typing import List, TypedDict
import gradio as gr
import pprint

# LangChain and LightRAG imports
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

# Logging Configuration
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Constants
WORKING_DIR = "/nfs/kundeshwar/surajKuralGPT/kuralLightRAG/LightRAG/dickens"

# Ensure Working Directory Exists
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# LightRAG Setup
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

# Graph State Definition
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Node Functions
def retrieve(state):
    print("--- RETRIEVE ---")
    question = state["question"]
    question=question+".Answer according to the thirukural Dataset."
    answer = rag.query(question)
    print(answer)
    return {"documents": answer, "question": question, "generation": answer}

def route_question(state):
    return "vectorstore"

# Workflow Setup
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)

# Configure memory saver
memory = MemorySaver()
# Compile the workflow
app = workflow.compile(checkpointer=memory)

# Chatbot Class to Manage Conversation
class RAGChatbot:
    def __init__(self):
        self.chat_history = []
    
    def process_query(self, question):
        # Process query using RAG workflow
        config = {"configurable": {"thread_id": "1"}}
        inputs = {"question": question}
        
        # Collect outputs
        outputs = []
        for output in app.stream(inputs, config):
            for key, value in output.items():
                # Extract the 'generation' field for the main response
                if key == "retrieve" and "generation" in value:
                    outputs.append(value["generation"])
        
        # Get the last generation (response content)
        response = outputs[-1] if outputs else "Sorry, I couldn't generate a response."
        
        # Store conversation
        self.chat_history.append((question, response))
        
        return response, self.chat_history


    def clear_history(self):
        self.chat_history = []
        return [], ""

# Create Chatbot Instance
chatbot = RAGChatbot()


# Gradio Interface with Enhanced Design
def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 Multi-turn RAG Chatbot")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat Input
                chatbot_component = gr.Chatbot(
                    height=500, 
                    label="Conversation",
                    show_label=True,
                    bubble_full_width=False
                )
                
                # Input Textbox
                msg = gr.Textbox(
                    label="Enter your query", 
                    placeholder="Ask a question about your documents..."
                )
                
                # Submit and Clear Buttons
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear Chat", variant="stop")
            
            with gr.Column(scale=1):
                # Document Retrieval Information
                retrieval_info = gr.Textbox(
                    label="Retrieval Details", 
                    interactive=False, 
                    height=500
                )
        
        # Event Handlers
        submit.click(
            chatbot.process_query, 
            inputs=[msg], 
            outputs=[retrieval_info, chatbot_component]
        )
        
        msg.submit(
            chatbot.process_query, 
            inputs=[msg], 
            outputs=[retrieval_info, chatbot_component]
        )
        
        clear.click(
            chatbot.clear_history, 
            inputs=None, 
            outputs=[chatbot_component, retrieval_info]
        )
    
    return demo



# Launch the Gradio app
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)