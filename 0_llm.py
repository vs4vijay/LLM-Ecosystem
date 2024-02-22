#!/usr/bin/env python
# -*- coding: utf-8 -*-
from openai import AzureOpenAI

from config import config


def system_prompt(name: str) -> str:
    return f"""
        You're a helpful assistant "{name}", here to answer the user's qeury. If you don't know
        the answer, just say "I don't know".
    """


def main():
    print(f"[+] Starting program with LLM model: {config.openai_model_name}")

    openai_client = AzureOpenAI(
        api_key=config.openai_api_key,
        api_version=config.openai_api_version,
        azure_endpoint=config.azure_openai_api_base,
    )

    
    while True:
        query = input("[+] Enter your query (or press enter to exit): ")

        if not query:
            break

        response = openai_client.chat.completions.create(
            model=config.openai_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt("DoL-Bot")},
                {"role": "user", "content": query},
            ],
            temperature=0.6,
            max_tokens=500,
        )

        print(f"[+] Raw Response: {response}")
        print(f"\n\n[+] Response: {response.choices[0].message.content}")
        print("===\n")
        # print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
