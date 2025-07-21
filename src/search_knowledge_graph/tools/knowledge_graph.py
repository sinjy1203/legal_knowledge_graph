import os
import json
from typing import Type, Optional
import asyncio
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from generate_knowledge_graph.nodes.entity_relation_extractor import GRAPH_SCHEMA


class KnowledgeGraphSearchInput(BaseModel):
    query: str = Field(
        description="A Cypher query to search the knowledge graph. The query should be in the format of a Cypher query."
    )
    parameters: dict = Field(
        description="A dictionary of parameters to pass to the Cypher query."
    )


class KnowledgeGraphSearchTool(BaseTool):
    name: str = "knowledge_graph_search"
    description: str = (
        "A tool for searching the knowledge graph."
    )
    args_schema: Type[BaseModel] = KnowledgeGraphSearchInput
    return_direct: bool = False

    neo4j_driver: object
    embedding_model: object
    graph_schema: dict

    def __init__(self):
        neo4j_driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        )
        embedding_model = OpenAIEmbeddings(
            base_url=os.getenv("EMBEDDING_BASE_URL"),
            model=os.getenv("EMBEDDING_MODEL"),
            api_key="dummy"
        )

        graph_schema = {"Node": [], "Edge": []}
        graph_schema["Node"].append(
            {
                "type": "Chunk",
                "description": "A chunk of text from a document.",
                "properties": {
                    "id": "chunk id",
                    "content": "chunk content",
                    "file_path": "file path",
                    "span": "span of the chunk",
                    "vector": "chunk embedding vector"
                }
            }
        )
        graph_schema["Edge"].append(
            {
                "type": "MENTIONS",
                "description": "A relationship between a chunk and an entity."
            }
        )
        for entity_info in GRAPH_SCHEMA['entity']:
            graph_schema['Node'].append({
                "type": entity_info['type'],
                "description": entity_info['description'],
                "properties": {
                    "id": "entity name",
                    "vector": "entity name embedding vector"
                }
            })
        for edge_info in GRAPH_SCHEMA['relationship']:
            graph_schema['Edge'].append({
                "type": edge_info['type'],
                "description": edge_info['description'],
                "properties": {
                    "description": "edge description",
                    "vector": "edge description embedding vector"
                }
            })

        super().__init__(
            neo4j_driver=neo4j_driver,
            embedding_model=embedding_model,
            graph_schema=graph_schema
        )

    def get_graph_schema(self):
        return json.dumps(self.graph_schema, ensure_ascii=False, indent=4)

    def _run(
        self,
        query: str,
        parameters: dict,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        for key, value in parameters.items():
            if key.split("_")[-1] == "vector":
                parameters[key] = self.embedding_model.embed_query(value)
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, parameters)
                return result.data()
        except Exception as e:
            return f"Error: {e}"

    async def _arun(
        self,
        query: str,
        parameters: dict,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ):
        return self._run(query, parameters, run_manager)