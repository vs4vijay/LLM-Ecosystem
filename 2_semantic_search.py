#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("[+] Running Example of Semantic Search")
    
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

    while True:
        query = input("Enter your query (or press enter to exit): ")

        if not query:
            break

        # Encode the query
        query_embedding = model.encode([query])[0]

        # Calculate cosine similarity between query and each sentence embedding
        scores = cosine_similarity([query_embedding], embeddings)[0]

        # Sort sentences by score in descending order
        sorted_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

        # Print the sorted sentences
        for sentence, score in sorted_sentences:
            print("Sentence:", sentence)
            print("Score:", score)
            print("---")
        
        print("===\n")


if __name__ == "__main__":
    main()
