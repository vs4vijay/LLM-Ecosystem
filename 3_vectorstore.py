#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pprint import pprint
from qdrant_client import QdrantClient

# Create in-memory Qdrant instance, for testing, CI/CD
client = QdrantClient(":memory:")
# Persists changes to disk, fast prototyping
# client = QdrantClient(path="path/to/db")


def main():
    print("[+] Running program of Vector Store")

    # Prepare your documents, metadata, and IDs
    docs = [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "id": 42,
            "metadata": {"name": "doc_0"},
        },
        {
            "content": "The cat is sitting on the mat.",
            "id": 1,
            "metadata": {"name": "doc_1"},
        },
        {
            "content": "I love eating pizza on Fridays.",
            "id": 2,
            "metadata": {"name": "doc_2"},
        },
        {
            "content": "The sun sets beautifully over the ocean.",
            "id": 3,
            "metadata": {"name": "doc_3"},
        },
        {
            "content": "Programming is both challenging and rewarding.",
            "id": 4,
            "metadata": {"name": "doc_4"},
        },
        {
            "content": "Music has the power to uplift our spirits.",
            "id": 5,
            "metadata": {"name": "doc_5"},
        },
        {
            "content": "The world is full of wonders waiting to be explored.",
            "id": 6,
            "metadata": {"name": "doc_6"},
        },
        {
            "content": "A cup of coffee in the morning is a great way to start the day.",
            "id": 7,
            "metadata": {"name": "doc_7"},
        },
        {
            "content": "Nature provides us with endless inspiration and tranquility.",
            "id": 8,
            "metadata": {"name": "doc_8"},
        },
        {
            "content": "Learning new things keeps our minds sharp and curious.",
            "id": 9,
            "metadata": {"name": "doc_9"},
        },
    ]

    collection_name = "llm_ecosystem"

    client.add(
        collection_name=collection_name,
        documents=[doc["content"] for doc in docs],
        ids=[doc["id"] for doc in docs],
        metadata=[doc["metadata"] for doc in docs],
    )

    while True:
        query = input("[+] Enter your query (or press enter to exit): ")

        if not query:
            break

        search_results = client.query(collection_name=collection_name, query_text=query)
        # pprint(search_results)
        results = [search_result.__dict__ for search_result in search_results]
        print("[+] Results:")
        pprint(results)
        print("---")


if __name__ == "__main__":
    main()
