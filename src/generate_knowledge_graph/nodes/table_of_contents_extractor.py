import json
from langchain_core.prompts import ChatPromptTemplate
from logger import setup_logger
from langgraph.types import Command
from langgraph.runtime import Runtime
from generate_knowledge_graph.utils.parser import JsonOutputParser
from generate_knowledge_graph.utils.callback import BatchCallback
from generate_knowledge_graph.state import ContextSchema

logger = setup_logger()


SYSTEM_TEMPLATE = """You are a document analysis expert specializing in legal contracts.
Your task is to extract and organize the table of contents from the document. 
The general structure of a legal contract consists of ARTICLES, each divided into sections, and it may or may not include exhibits and annexes.

Always answer in this format:
```json
{{
  "ARTICLE_I": {{
    "name": "name of article I",
    "sections": {{
        "section_1_1": "name of section 1.1",
        ...
    }},
    ...
  }},
  "Exhibits(optional)": {{
    "Exhibit_A": "name of exhibit A",
    ...
  }},
  "Annexes(optional)": {{
    "Annex_I": "name of annex I",
    ...
  }},
  ...
}}
```
"""

USER_TEMPLATE = """
<Legal_Contract>
{legal_contract}
</Legal_Contract>
"""


class TableOfContentsExtractor:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state, runtime: Runtime[ContextSchema]):
        chain = runtime.context.table_of_contents_extractor_prompt | self.llm | JsonOutputParser()
        queries = [{"legal_contract": document.intro.strip()} for document in state.documents]

        with BatchCallback(total=len(queries), desc="Table of Contents Extractor") as cb:
            responses = chain.batch(queries, config={"callbacks": [cb]})
        
        table_of_contents = {document.file_path: response for document, response in zip(state.documents, responses)}

        return Command(
            update={
                "table_of_contents": table_of_contents
            },
            goto="Chunker"
        )