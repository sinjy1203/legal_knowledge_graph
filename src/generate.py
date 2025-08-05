import os
from dotenv import load_dotenv
from generate_knowledge_graph.builder import graph
from langfuse.langchain import CallbackHandler


load_dotenv(override=True)


def main():
    langfuse_handler = CallbackHandler()
    input = {}
    context = {"chunking_strategy": "rcts"}
    config = {"callbacks": [langfuse_handler], "metadata": {"langfuse_tags": ["generate"]}}
    _ = graph.invoke(input, context=context, config=config)

if __name__ == "__main__":
    main()