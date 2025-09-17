import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class SearchNeighborChunkInput(BaseModel):
    section_id: str = Field(
        description="section_id"
    )


class SearchNeighborChunkTool(BaseTool):
    name: str = "SearchNeighborSectionTool"
    description: str = (
        "Given a section_id, return sections connected via NEXT and PREV edges. "
        "Purpose: To find the neighboring sections of a given section. "
        "Return format: [{'section_id': id, 'section_name': section_name, 'summary': summary_or_content_if_empty}]"
    )
    args_schema: Type[BaseModel] = SearchNeighborChunkInput
    return_direct: bool = False

    # Dependency injection
    neo4j_driver: Any = None

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (c:Chunk {id: $chunk_id})-[:NEXT]->(nb:Chunk)
    RETURN DISTINCT nb.id AS chunk_id,
           nb.name AS section_name,
           CASE WHEN nb.summary IS NULL OR nb.summary = '' THEN nb.content ELSE nb.summary END AS summary
    UNION
    MATCH (nb:Chunk)-[:NEXT]->(c:Chunk {id: $chunk_id})
    RETURN DISTINCT nb.id AS chunk_id,
           nb.name AS section_name,
           CASE WHEN nb.summary IS NULL OR nb.summary = '' THEN nb.content ELSE nb.summary END AS summary
    """

    def __init__(self, neo4j_driver: Any):
        super().__init__(neo4j_driver=neo4j_driver)

    def _run(
        self,
        section_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        with self.neo4j_driver.session() as session:
            params = {"section_id": section_id}
            results = session.run(self.CYPHER_QUERY, params)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self,
        section_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(section_id)


