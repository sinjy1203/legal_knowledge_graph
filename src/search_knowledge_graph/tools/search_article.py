import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class SearchArticleInput(BaseModel):
    corpus_id: str = Field(
        description="Corpus ID to search"
    )
    query: str = Field(
        description="Query to search"
    )


class SearchArticleTool(BaseTool):
    name: str = "SearchArticleTool"
    description: str = (
        "This tool searches Article nodes within a specific Corpus by semantic similarity to the query. "
        "Purpose: Given a corpus_id and a natural-language query, return the most relevant Articles. "
        "return_schema: [{‘id’: article_id, ‘name’: article_name, ‘summary’: article_summary}, …]"
    )
    args_schema: Type[BaseModel] = SearchArticleInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (co:Corpus {id: $corpus_id})-[:CHILD]->(a:Article)
    WHERE a.vector IS NOT NULL
    WITH a, gds.similarity.cosine(a.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN a.id AS id, a.name AS name, a.summary AS summary
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
        corpus_id: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        with self.neo4j_driver.session() as session:
            params = {
                "corpus_id": corpus_id,
                "query_vector": self.embedding_model.embed_query(query),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }
            
            results = session.run(self.CYPHER_QUERY, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self, 
        corpus_id: str,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(corpus_id, query)
