import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class SearchCorpusTool(BaseTool):
    name: str = "SearchCorpusTool"
    description: str = (
        "This tool searches for the names of contracts stored in the database. "
        "Purpose: To determine which legal contract(s) should be reviewed in order to answer the user’s question. "
        "return_schema: [{‘id’: corpus1_id, ‘name’: corpus1_name}, {‘id’: corpus2_id, ‘name’: corpus2_name}, …]"
    )
    args_schema: Type[BaseModel] = None
    return_direct: bool = True

    # 필요한 의존성 주입
    neo4j_driver: Any = None

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (c:Corpus)
    RETURN c.id AS id, c.name AS name
    ORDER BY c.name
    """

    def __init__(self, neo4j_driver: Any):
        super().__init__(
            neo4j_driver=neo4j_driver
        )

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> str:
        with self.neo4j_driver.session() as session:
            results = session.run(self.CYPHER_QUERY)
            dict_results = [record.data() for record in results]
            return dict_results

    async def _arun(
        self, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> str:
        return self._run(**kwargs) 