import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class SearchSectionInput(BaseModel):
    article_id: str = Field(
        description="Article ID to search within"
    )
    query: str = Field(
        description="Query to search"
    )


class SearchSectionTool(BaseTool):
    name: str = "SearchSectionTool"
    description: str = (
        "This tool searches Section nodes within a specific Article by semantic similarity to the query. "
        "Purpose: Given an article_id and a natural-language query, return the most relevant Sections. "
        "return_schema: [{‘id’: section_id, ‘name’: section_name, ‘summary’: section_summary}, …]"
    )
    args_schema: Type[BaseModel] = SearchSectionInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (a:Article {id: $article_id})-[:CHILD]->(s:Section)
    WHERE s.vector IS NOT NULL
    WITH s, gds.similarity.cosine(s.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN s.id AS id, s.name AS name, s.summary AS summary
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
        article_id: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        with self.neo4j_driver.session() as session:
            params = {
                "article_id": article_id,
                "query_vector": self.embedding_model.embed_query(query),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }

            results = session.run(self.CYPHER_QUERY, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self,
        article_id: str,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(article_id, query)


