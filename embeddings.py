#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer


def main():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    print(f'[+] Loading embedding model: {embedding_model}')

    model = SentenceTransformer(embedding_model)

    # Our sentences we like to encode
    sentences = [
        "This framework generates embeddings for each input sentence",
        "Sentences are passed as a list of string.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, Testing from code",
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
