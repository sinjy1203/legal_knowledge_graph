import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class GetMentionChunkInput(BaseModel):
    entity_id: str = Field(
        description="Entity id to get mention chunk"
    )
    entity_type: str = Field(
        description="entity id's type"
    )


class GetMentionChunkTool(BaseTool):
    name: str = "get_mention_chunk"
    description: str = (
        "This is a tool for getting mention chunk from a given entity id."
        "Purpose: To get the mention chunk from the entity id"
        "return_schema: [{'id': 'chunk_id1', 'content': 'chunk_content1'}, {'id': 'chunk_id2', 'content': 'chunk_content2'}, ...]"
    )
    args_schema: Type[BaseModel] = GetMentionChunkInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    top_k: int = 5

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:{entity_type})
    WHERE e.id = $entity_id
    WITH c
    MATCH (c)-[:MENTIONS]->()
    WITH c, count(*) as mention_count
    RETURN c.id AS id, c.content AS content
    ORDER BY mention_count DESC
    LIMIT $top_k
    """

    def __init__(self, neo4j_driver: Any, top_k: int = 5):
        super().__init__(
            neo4j_driver=neo4j_driver,
            top_k=top_k
        )

    def _run(
        self, 
        entity_id: str, 
        entity_type: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        with self.neo4j_driver.session() as session:
            cypher_query = self.CYPHER_QUERY.format(entity_type=entity_type)
            params = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "top_k": self.top_k
            }
            
            results = session.run(cypher_query, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self, 
        entity_id: str, 
        entity_type: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(entity_id, entity_type)
