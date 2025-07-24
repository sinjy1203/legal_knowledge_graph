import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class SearchConnectedEntityInput(BaseModel):
    entity_id: str = Field(
        description="ID of the entity to start the search from"
    )
    entity_type: str = Field(
        description="Type of the entity to start the search from - must be one of Node types in the graph database schema"
    )
    relationship_type: str = Field(
        description="Relationship type to search - must be one of Edge types in the graph database schema"
    )
    relationship_direction: str = Field(
        description="Direction of the relationship: 'outgoing', 'incoming', or 'both'"
    )
    query: str = Field(
        description="Query text to find relationships with similar descriptions"
    )


class SearchConnectedEntityTool(BaseTool):
    name: str = "search_connected_entity"
    description: str = (
        "This is a tool for searching entities connected to a given entity through relationships with similar descriptions to a query."
        "Purpose: To find entities connected to a specific entity through relationships that match the query description."
        "return_schema: [{'id': 'entity_id1', 'type': 'entity_type1', 'description': 'relationship_description1', 'chunk_id': 'chunk_id1'}, ...]"
    )
    args_schema: Type[BaseModel] = SearchConnectedEntityInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY_OUTGOING: ClassVar[str] = """
    MATCH (start:{entity_type})-[r:{relationship_type}]->(target)
    WHERE start.id = $entity_id AND r.vector IS NOT NULL
    WITH r, target, gds.similarity.cosine(r.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN 
        target.id AS id,
        head(labels(target)) AS type,
        r.description AS description,
        r.chunk_id AS chunk_id
    ORDER BY score DESC
    LIMIT $top_k
    """

    CYPHER_QUERY_INCOMING: ClassVar[str] = """
    MATCH (start:{entity_type})<-[r:{relationship_type}]-(target)
    WHERE start.id = $entity_id AND r.vector IS NOT NULL
    WITH r, target, gds.similarity.cosine(r.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN 
        target.id AS id,
        head(labels(target)) AS type,
        r.description AS description,
        r.chunk_id AS chunk_id
    ORDER BY score DESC
    LIMIT $top_k
    """

    CYPHER_QUERY_BOTH: ClassVar[str] = """
    MATCH (start:{entity_type})-[r:{relationship_type}]-(target)
    WHERE start.id = $entity_id AND r.vector IS NOT NULL
    WITH r, target, gds.similarity.cosine(r.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN 
        target.id AS id,
        head(labels(target)) AS type,
        r.description AS description,
        r.chunk_id AS chunk_id
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
        entity_id: str,
        entity_type: str,
        relationship_type: str, 
        relationship_direction: str,
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # 방향에 따라 적절한 쿼리 선택
        if relationship_direction.lower() == "outgoing":
            cypher_query = self.CYPHER_QUERY_OUTGOING.format(entity_type=entity_type, relationship_type=relationship_type)
        elif relationship_direction.lower() == "incoming":
            cypher_query = self.CYPHER_QUERY_INCOMING.format(entity_type=entity_type, relationship_type=relationship_type)
        elif relationship_direction.lower() == "both":
            cypher_query = self.CYPHER_QUERY_BOTH.format(entity_type=entity_type, relationship_type=relationship_type)
        else:
            return f"Error: Invalid relationship_direction. Must be 'outgoing', 'incoming', or 'both'"
        
        with self.neo4j_driver.session() as session:
            params = {
                "entity_id": entity_id,
                "query_vector": self.embedding_model.embed_query(query),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }
            
            results = session.run(cypher_query, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self, 
        entity_id: str,
        entity_type: str,
        relationship_type: str, 
        relationship_direction: str,
        query: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(entity_id, entity_type, relationship_type, relationship_direction, query) 