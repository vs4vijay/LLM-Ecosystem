#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pprint import pprint
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Create in-memory Qdrant instance, for testing, CI/CD
client = QdrantClient(":memory:")
# Persists changes to disk, fast prototyping
# client = QdrantClient(path="path/to/db")


def main():
    print("[+] Running Example of Vector Store")
    # Prepare your documents, metadata, and IDs
    docs = [
        "This framework generates embeddings for each input sentence",
        "Sentences are passed as a list of string.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, Testing from code",
    ]
    metadata = [{"name": f"doc_{i}"} for i in range(len(docs))]
    ids = [42, 1, 2, 3]

    collection_name = "llm_ecosystem"

    # Use the new add method
    client.add(
        collection_name=collection_name, documents=docs, metadata=metadata, ids=ids
    )

    while True:
        query = input("Enter your query (or press enter to exit): ")

        if not query:
            break

        search_results = client.query(collection_name=collection_name, query_text=query)
        results = [search_result.__dict__ for search_result in search_results]
        print("Results:")
        pprint(results)
        print('---')


if __name__ == "__main__":
    main()
