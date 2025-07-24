import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class SearchRelationshipInput(BaseModel):
    relationship_type: str = Field(
        description="Relationship type to search - must be one of Edge types in the graph database schema"
    )
    query: str = Field(
        description="Query text to find relationships with similar descriptions"
    )


class SearchRelationshipTool(BaseTool):
    name: str = "search_relationship"
    description: str = (
        "This is a tool for searching relationships in a graph database that have similar descriptions to a given query."
        "Purpose: To find the actual relationships stored in the graph database "
        "return_schema: [{'relationship_type': 'relationship_type1', 'relationship_description': 'description1', 'source_id': 'source_id1', 'source_type': 'source_type1', 'target_id': 'target_id1', 'target_type': 'target_type1'}, ...]"
    )
    args_schema: Type[BaseModel] = SearchRelationshipInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (source)-[r:{relationship_type}]->(target)
    WHERE r.vector IS NOT NULL
    WITH r, source, target, gds.similarity.cosine(r.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN 
        type(r) AS relationship_type,
        r.description AS relationship_description,
        source.id AS source_id,
        head(labels(source)) AS source_type,
        target.id AS target_id,
        head(labels(target)) AS target_type
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
        relationship_type: str, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        with self.neo4j_driver.session() as session:
            cypher_query = self.CYPHER_QUERY.format(relationship_type=relationship_type)
            params = {
                "query_vector": self.embedding_model.embed_query(query),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }
            
            results = session.run(cypher_query, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self, 
        relationship_type: str, 
        query: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(relationship_type, query)
