#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding  # noqa: F401


def main():
    print("[+] Running program of Embeddings")

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    # embedding_model = "BAAI/bge-base-en"

    print(f'[+] Loading embedding model: {embedding_model}')

    model = SentenceTransformer(embedding_model)
    # model = TextEmbedding(model_name=embedding_model)

    # Our sentences we like to encode
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I love programming and enjoy solving complex problems.",
        "The sun sets in the west, painting the sky with vibrant colors.",
        "Music has the power to uplift our spirits.",
        "The world is full of wonders waiting to be explored.",
        "A cup of coffee in the morning is a great way to start the day.",
        "Nature provides us with endless inspiration and tranquility.",
        "Learning new things keeps our minds sharp and curious.",
    ]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("---")


if __name__ == "__main__":
    main()
