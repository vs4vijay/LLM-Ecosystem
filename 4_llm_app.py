#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts import PromptTemplate  # noqa: F401
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureChatOpenAI
# noqa: F401
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,  # noqa: F401
    UnstructuredPowerPointLoader, # noqa: F401
    UnstructuredPDFLoader,  # noqa: F401
    PyPDFLoader,  # noqa: F401
    PyPDFDirectoryLoader,  # noqa: F401
)  
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from sentence_transformers import SentenceTransformer
from langchain.memory import VectorStoreRetrieverMemory

from config import config


def get_embeddings(embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"[+] Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    # embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    # embedding_model = "BAAI/bge-base-en"
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    # embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    embeddings_model = SentenceTransformerEmbeddings(client=model)

    return embeddings_model


def get_llm_model():
    print(f"[+] Loading LLM model: {config.openai_model_name}")
    llm = AzureChatOpenAI(
        openai_api_type=config.openai_api_type,
        openai_api_version=config.azure_openai_api_version,
        azure_endpoint=config.azure_openai_api_base,
        api_key=config.openai_api_key,
        model=config.openai_model_name,
        azure_deployment=config.openai_deployment_name,
        verbose=True,
    )

    return llm


def get_vector_store(namespace, embeddings):
    print("[+] Loading docs using vector store: Qdrant")

    # Loading Text Documents
    # loader = TextLoader("./data/EngSysQueryLang.txt")
    # loader = UnstructuredPowerPointLoader("./data/Prompt_Engineering_DOL.pptx")
    # loader = UnstructuredPDFLoader("./data/Prompt_Engineering_DOL.pdf")
    # loader = PyPDFLoader("./data/Prompt_Engineering_DOL.pdf")
    loader = DirectoryLoader(
        "./data",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()

    # Splitting Text Documents
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Building Vector Store from Text Documents
    vector_store = Qdrant.from_documents(
        texts,
        embeddings,
        location=":memory:",  # For in-memory storage only
        # path="./data/qdrant.db",        # For on-disk storage only
        # url="http://localhost:6333",    # For remote DB
        collection_name=namespace,
        verbose=True,
    )

    return vector_store


def get_chain():
    embeddings = get_embeddings()
    vector_store = get_vector_store("default", embeddings)
    llm = get_llm_model()
    retriever = vector_store.as_retriever()

    # results = vector_store.similarity_search_with_score("EngSysLang")
    # print(f"[+] Similarity Search Results: {results=}")

    # chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="stuff",
    #     verbose=True,
    # )

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
            You're a helpful assistant "DoL-Bot", here to answer the user's qeury. If you don't know the answer, 
            just say "I don't know".

            Relevant pieces of previous conversation:
            {history}
            
            Answer my question: {input}
        """,
    )
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
    )

    return chain


def main():
    print("[+] Running an LLM App")
    chain = get_chain()

    while True:
        query = input("[+] Enter your query (or press enter to exit): ")

        if not query:
            break

        res = chain.invoke(
            f"""
                {query}
            """,
            return_only_outputs=True,
        )
        print(f"[+] Chain Result: {res=}")
        print("---")


if __name__ == "__main__":
    main()
