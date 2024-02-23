# LLM-Ecosystem

Code for Embeddings, VectorStore, SemanticSearch, and RAG using Azure OpenAI.

## Installation

Pre-requisites:

- Python 3.10+ and pip

```bash

pip install -r requirements.txt

```

## Running

- Copy `.env.example` to `.env` and fill in the values

- Run the following command to start the server

```bash

python 0_llm.py

python 1_embeddings.py

python 2_semantic_search.py

python 3_vectorstore.py

python 4_llm_app.py

```

---

## Advance

```bash

poetry shell

poetry install

```