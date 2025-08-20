import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class SearchChunkInput(BaseModel):
    section_id: str = Field(
        description="Section ID to search within"
    )
    query: str = Field(
        description="Query to search"
    )


class SearchChunkTool(BaseTool):
    name: str = "SearchChunkTool"
    description: str = (
        "This tool searches Chunk nodes within a specific Section by semantic similarity to the query. "
        "Purpose: Given a section_id and a natural-language query, return the most relevant Chunks. "
        "return_schema: [{‘id’: chunk_id, ‘content’: chunk_content}, …]"
    )
    args_schema: Type[BaseModel] = SearchChunkInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (s:Section {id: $section_id})-[:CHILD]->(c:Chunk)
    WHERE c.vector IS NOT NULL
    WITH c, gds.similarity.cosine(c.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN c.id AS id, c.content AS content
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
        section_id: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        with self.neo4j_driver.session() as session:
            params = {
                "section_id": section_id,
                "query_vector": self.embedding_model.embed_query(query),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }

            results = session.run(self.CYPHER_QUERY, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self,
        section_id: str,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(section_id, query)
