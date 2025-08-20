import json
from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class ResponseInput(BaseModel):
    chunk_ids: List[str] = Field(
        description="List of Chunk IDs to fetch final response information for"
    )


class ResponseTool(BaseTool):
    name: str = "ResponseTool"
    description: str = (
        "Given a list of Chunk IDs, return each chunk's file_path and span. "
        "Purpose: This is the final tool used by the agent to assemble the answer context. "
        "return_schema: [{‘file_path’: file_path, ‘span’: span}, …]"
    )
    args_schema: Type[BaseModel] = ResponseInput
    return_direct: bool = True

    # Dependency injection
    neo4j_driver: Any = None

    CYPHER_QUERY: str = (
        """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        RETURN c.file_path AS file_path, c.span AS span
        """
    )

    def __init__(self, neo4j_driver: Any):
        super().__init__(neo4j_driver=neo4j_driver)

    def _run(
        self,
        chunk_ids: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Query and then reorder results to match input order
        with self.neo4j_driver.session() as session:
            results = session.run(self.CYPHER_QUERY, {"chunk_ids": chunk_ids})
            records = [record.data() for record in results]

        return records

    async def _arun(
        self,
        chunk_ids: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(chunk_ids)


