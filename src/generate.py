import os
from dotenv import load_dotenv
from generate_knowledge_graph.builder import graph
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from generate_knowledge_graph.nodes.table_of_contents_extractor import SYSTEM_TEMPLATE as TABLE_OF_CONTENTS_EXTRACTOR_SYSTEM_TEMPLATE, USER_TEMPLATE as TABLE_OF_CONTENTS_EXTRACTOR_USER_TEMPLATE
from generate_knowledge_graph.nodes.intro_body_separator import SYSTEM_TEMPLATE as CONTRACT_START_FINDER_SYSTEM_TEMPLATE, USER_TEMPLATE as CONTRACT_START_FINDER_USER_TEMPLATE
from generate_knowledge_graph.nodes.document_structure_detector import SYSTEM_TEMPLATE as DOCUMENT_STRUCTURE_DETECTOR_SYSTEM_TEMPLATE, USER_TEMPLATE as DOCUMENT_STRUCTURE_DETECTOR_USER_TEMPLATE
from generate_knowledge_graph.nodes.summarizer import SYSTEM_TEMPLATE as SUMMARIZER_SYSTEM_TEMPLATE, USER_TEMPLATE as SUMMARIZER_USER_TEMPLATE



load_dotenv(override=True)

default_prompt = {
    "table-of-contents-extractor": {
        "system": TABLE_OF_CONTENTS_EXTRACTOR_SYSTEM_TEMPLATE,
        "user": TABLE_OF_CONTENTS_EXTRACTOR_USER_TEMPLATE,
    },
    "contract-start-finder": {
        "system": CONTRACT_START_FINDER_SYSTEM_TEMPLATE,
        "user": CONTRACT_START_FINDER_USER_TEMPLATE,
    },
    "document-structure-detector": {
        "system": DOCUMENT_STRUCTURE_DETECTOR_SYSTEM_TEMPLATE,
        "user": DOCUMENT_STRUCTURE_DETECTOR_USER_TEMPLATE,
    },
    "summarizer": {
        "system": SUMMARIZER_SYSTEM_TEMPLATE,
        "user": SUMMARIZER_USER_TEMPLATE,
    },
}

def get_system_prompt(prompt_name: str):
    try:
        langfuse_client = get_client()
        langfuse_prompt = langfuse_client.get_prompt(prompt_name)
        prompt = ChatPromptTemplate.from_messages(langfuse_prompt.get_langchain_prompt())
    except:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", default_prompt[prompt_name]["system"]),
                ("user", default_prompt[prompt_name]["user"]),
            ]
        )
    return prompt


def main():
    langfuse_handler = CallbackHandler()
    input = {}
    context = {
        "chunking_strategy": "page",
        "table_of_contents_extractor_prompt": get_system_prompt("table-of-contents-extractor"),
        "contract_start_finder_prompt": get_system_prompt("contract-start-finder"),
        "document_structure_detector_prompt": get_system_prompt("document-structure-detector"),
        "summarizer_prompt": get_system_prompt("summarizer")
    }
    config = {"callbacks": [langfuse_handler], "metadata": {"langfuse_tags": ["generate"]}}
    _ = graph.invoke(input, context=context, config=config)

if __name__ == "__main__":
    main()