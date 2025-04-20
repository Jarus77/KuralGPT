<center><h2>🚀 LightRAG: For KuralGPT</h2></center>


## Algorithm Flowchart

![LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
*Figure 1: LightRAG Indexing Flowchart*
![LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
*Figure 2: LightRAG Retrieval and Querying Flowchart*

## Install

* Install from source (Recommend)

```bash
cd LightRAG
pip install -e .
```
* Install from PyPI
```bash
pip install lightrag-hku
```

Use the below Python snippet (in a script) to initialize LightRAG and perform queries:

```python
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./dickens"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("./book.txt") as f:
    rag.insert(f.read())

# Perform naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))
```

<details>
### Overview
If you want to use Ollama models, you need to pull model you plan to use and embedding model, for example `nomic-embed-text`.

Then you only need to set LightRAG as follows:

```python
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

# Initialize LightRAG with Ollama model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='your_model_name', # Your model name
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```

### Using Neo4J for Storage

* For production level scenarios you will most likely want to leverage an enterprise solution
* for KG storage. Running Neo4J in Docker is recommended for seamless local testing.
* See: https://hub.docker.com/_/neo4j


```python
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

When you launch the project be sure to override the default KG: NetworkS
by specifying kg="Neo4JStorage".

# Note: Default settings use NetworkX
#Initialize LightRAG with Neo4J implementation.
WORKING_DIR = "./local_neo4jWorkDir"

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    kg="Neo4JStorage", #<-----------override KG default
    log_level="DEBUG"  #<-----------override log_level default
)
```
see test_neo4j.py for a working example.

### Increasing context size
In order for LightRAG to work context should be at least 32k tokens. By default Ollama models have context size of 8k. You can achieve this using one of two ways:

#### Increasing the `num_ctx` parameter in Modelfile.

1. Pull the model:
```bash
ollama pull qwen2
```

2. Display the model file:
```bash
ollama show --modelfile qwen2 > Modelfile
```

3. Edit the Modelfile by adding the following line:
```bash
PARAMETER num_ctx 32768
```

4. Create the modified model:
```bash
ollama create -f Modelfile qwen2m
```

#### Setup `num_ctx` via Ollama API.
Tiy can use `llm_model_kwargs` param to configure ollama:

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='your_model_name', # Your model name
    llm_model_kwargs={"options": {"num_ctx": 32768}},
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```
#### Fully functional example

There fully functional example `examples/app.py` that utilizes `gemma2:2b` model, runs only 4 requests in parallel and set context size to 32k.

