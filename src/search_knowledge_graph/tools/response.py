import json
from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class ResponseInput(BaseModel):
    component_ids: List[str] = Field(
        description="The component IDs of the lowest-level components in the contract’s table of contents"
    )


class ResponseTool(BaseTool):
    name: str = "ResponseTool"
    description: str = (
        "This tool retrieves the spans of each component ID at the lowest level of a contract’s table of contents. "
        "return_schema: [{‘file_path’: file_path, ‘span’: span}, …]"
    )
    args_schema: Type[BaseModel] = ResponseInput
    return_direct: bool = True

    # Dependency injection
    neo4j_driver: Any = None

    CYPHER_QUERY: str = (
        """
        MATCH (c:Chunk)
        WHERE c.id IN $component_ids
        RETURN c.file_path AS file_path, c.span AS span, c.name AS name, c.content AS content
        """
    )

    def __init__(self, neo4j_driver: Any):
        super().__init__(neo4j_driver=neo4j_driver)

    def _run(
        self,
        component_ids: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Query and then reorder results to match input order
        with self.neo4j_driver.session() as session:
            results = session.run(self.CYPHER_QUERY, {"component_ids": component_ids})
            records = [record.data() for record in results]

        final_records = []

        for record in records:
            with open(f"./data/corpus/{record['file_path']}", "r") as f:
                contract_content = f.read()
            start_index = contract_content.find(record['content'])
            end_index = start_index + len(record['content'])
            if start_index == -1 or end_index == -1:
                continue
            record['span'] = [start_index, end_index]
            final_records.append(record)

        return final_records

    async def _arun(
        self,
        component_ids: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(component_ids)


