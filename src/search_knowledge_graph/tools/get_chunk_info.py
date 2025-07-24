import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from generate_knowledge_graph.nodes.entity_relation_extractor import GRAPH_SCHEMA


class GetChunkInfoInput(BaseModel):
    chunk_ids: List[str] = Field(
        description="List of chunk IDs to retrieve file path and span information for"
    )


class GetChunkInfoTool(BaseTool):
    name: str = "get_chunk_info"
    description: str = (
        "This is the final tool for retrieving file path and span information of the most relevant chunks."
        "Purpose: When the agent has identified the most appropriate chunks that can answer the user's question, "
        "this tool provides the source location information (file_path and span) for those chunks. "
        "Using this tool indicates that the search process is complete and the relevant information has been found."
        "return_schema: [{'file_path': 'file_path1', 'span': 'span1'}, {'file_path': 'file_path2', 'span': 'span2'}, ...]"
    )
    args_schema: Type[BaseModel] = GetChunkInfoInput
    return_direct: bool = True

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    graph_schema: dict = None

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (c:Chunk)
    WHERE c.id IN $chunk_ids
    RETURN c.file_path AS file_path, c.span AS span
    ORDER BY c.id
    """

    def __init__(self, neo4j_driver: Any):
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
            graph_schema=graph_schema
        )

    def _run(
        self, 
        chunk_ids: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        with self.neo4j_driver.session() as session:
            params = {
                "chunk_ids": chunk_ids
            }
            
            results = session.run(self.CYPHER_QUERY, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self, 
        chunk_ids: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(chunk_ids) 