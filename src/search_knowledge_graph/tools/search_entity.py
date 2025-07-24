import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class SearchEntityInput(BaseModel):
    entity_type: str = Field(
        description="Entity type to search - must be one of Node types in the graph database schema"
    )
    entity_name: str = Field(
        description="Entity name to search"
    )


class SearchEntityTool(BaseTool):
    name: str = "search_entity"
    description: str = (
        "This is a tool for searching entities in a graph database that are similar to a given entity name."
        "Purpose: To find the actual entities stored in the graph database based on the entity name mentioned in the user's question."
        "return_schema: [{'id': 'entity_id1'}, {'id': 'entity_id2'}, ...]"
    )
    args_schema: Type[BaseModel] = SearchEntityInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (e:{entity_type})
    WHERE e.vector IS NOT NULL
    WITH e, gds.similarity.cosine(e.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN e.id AS id
    ORDER BY score DESC
    LIMIT $top_k
    """

    def __init__(self, neo4j_driver: Any, embedding_model: Any, top_k: int = 5, similarity_threshold: float = 0.0):
        super().__init__(
            neo4j_driver=neo4j_driver,
            embedding_model=embedding_model,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

    def _run(
        self, 
        entity_type: str, 
        entity_name: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        with self.neo4j_driver.session() as session:
            cypher_query = self.CYPHER_QUERY.format(entity_type=entity_type)
            params = {
                "query_vector": self.embedding_model.embed_query(entity_name),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }
            
            results = session.run(cypher_query, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self, 
        entity_type: str, 
        entity_name: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(entity_type, entity_name)
