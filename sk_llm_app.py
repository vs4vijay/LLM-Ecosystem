#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from config import config


def get_llm_model():
    print(f"[+] Loading LLM model: {config.openai_model_name}")

    llm = AzureChatCompletion(
        deployment_name=config.openai_deployment_name,
        base_url=config.azure_openai_api_base,
        api_key=config.openai_api_key,
    )

    return llm


async def main():
    print("[+] Running an LLM App")

    llm = get_llm_model()
    kernel = sk.Kernel()
    kernel.add_chat_service(service_id="DoL-Bot", service=llm)

    # kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())

    context = kernel.create_new_context()
    context["history"] = ""

    

    chat_function = kernel.create_semantic_function(
        """
        You're a helpful assistant "DoL-Bot", here to answer the user's qeury. If you don't know the answer, 
        just say "I don't know".

    
        Relevant pieces of previous conversation:
        {{$history}}
        
        Answer my question: {{$user_input}}
    """
    )



    while True:
        query = input("[+] Enter your query (or press enter to exit): ")

        if not query:
            break

        context["user_input"] = query

        chat = await chat_function.invoke(context=context)
        res = chat()
        print(f"[+] Result: {res=}")
        print("---")

        context["history"] += f"\nUser: {context['user_input']}\nChatBot: {res}\n"


if __name__ == "__main__":
    asyncio.run(main())
