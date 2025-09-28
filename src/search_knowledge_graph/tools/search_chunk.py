import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class SearchSubComponentInput(BaseModel):
    id: str = Field(
        description="A component_id in UUID format"
    )
    query: str = Field(
        description="needed content"
    )


class SearchSubComponentTool(BaseTool):
    name: str = "SearchSubComponentTool"
    description: str = (
        "This tool looks through the tree-structured table of contents of a contract and finds sub components that are most relevant to the needed content. "
        "return_schema: [{‘sub_component_id’: sub_component_id, 'sub_component_name': sub_component_name, ‘sub_component_summary’: summary, 'sub_component_leaf': Existence of lower-level subcomponents (True or False)}, …]"
    )
    args_schema: Type[BaseModel] = SearchSubComponentInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (n)-[:CHILD]->(c:Chunk)
    WHERE (n:Corpus OR n:Chunk) AND n.id = $id AND c.vector IS NOT NULL
    WITH c, gds.similarity.cosine(c.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN c.id AS sub_component_id,
           c.name AS sub_component_name,
           CASE WHEN c.summary IS NULL OR c.summary = '' THEN c.content ELSE c.summary END AS sub_component_summary,
           c.leaf AS sub_component_leaf
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
        id: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        with self.neo4j_driver.session() as session:
            params = {
                "id": id,
                "query_vector": self.embedding_model.embed_query(query),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }

            results = session.run(self.CYPHER_QUERY, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self,
        id: str,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(id, query)



class SearchComponentInput(BaseModel):
    id: str = Field(
        description="A contract_id in UUID format"
    )
    query: str = Field(
        description="needed content"
    )


class SearchComponentTool(BaseTool):
    name: str = "SearchComponentTool"
    description: str = (
        "This tool looks through the tree-structured table of contents of a contract and finds components that are most relevant to the needed content. "
        "return_schema: [{‘component_id’: component_id, 'component_name': component_name, ‘component_summary’: summary}, …]"
    )
    args_schema: Type[BaseModel] = SearchComponentInput
    return_direct: bool = False

    # 필요한 의존성 주입
    neo4j_driver: Any = None
    embedding_model: Any = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (n)-[:CHILD]->(c:Chunk)
    WHERE (n:Corpus OR n:Chunk) AND n.id = $id AND c.vector IS NOT NULL
    WITH c, gds.similarity.cosine(c.vector, $query_vector) AS score
    WHERE score > $similarity_threshold
    RETURN c.id AS component_id,
           c.name AS component_name,
           CASE WHEN c.summary IS NULL OR c.summary = '' THEN c.content ELSE c.summary END AS component_summary
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
        id: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        with self.neo4j_driver.session() as session:
            params = {
                "id": id,
                "query_vector": self.embedding_model.embed_query(query),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k
            }

            results = session.run(self.CYPHER_QUERY, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self,
        id: str,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(id, query)
