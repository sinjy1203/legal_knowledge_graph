import json
from typing import Type, Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class GetCorpusTOCInput(BaseModel):
    contract_id: str = Field(
        description="Contract ID"
    )


class GetCorpusTOCTool(BaseTool):
    name: str = "GetContractTOCTool"
    description: str = (
        "This tool retrieves the table of contents of a contract. "
        "return_schema: { 'contract_id': str, 'table_of_contents': dict }"
    )
    args_schema: Type[BaseModel] = GetCorpusTOCInput
    return_direct: bool = True

    # Dependencies
    neo4j_driver: Any = None

    CYPHER_QUERY: ClassVar[str] = """
    MATCH (c:Corpus {id: $corpus_id})
    RETURN c.table_of_contents AS table_of_contents
    """

    def __init__(self, neo4j_driver: Any):
        super().__init__(neo4j_driver=neo4j_driver)

    def _convert_toc_to_components(self, toc: Any) -> List[Dict[str, Any]]:
        # If already in target schema, return as is
        if isinstance(toc, list) and all(isinstance(x, dict) and "component_name" in x for x in toc):
            return toc

        components: List[Dict[str, Any]] = []
        if not isinstance(toc, dict):
            return components

        for top_key, top_val in toc.items():
            key_str = top_key if isinstance(top_key, str) else str(top_key)
            is_article = key_str.lower().startswith("article_")

            if is_article:
                name = None
                sections = {}
                if isinstance(top_val, dict):
                    name = top_val.get("name")
                    sections = top_val.get("sections", {})
                    if not isinstance(sections, dict):
                        sections = {}

                childs: List[Dict[str, Any]] = []
                for sec_key, sec_val in sections.items():
                    childs.append({
                        "component_name": sec_key,
                        "component_description": sec_val if isinstance(sec_val, str) else None,
                        "childs": []
                    })

                components.append({
                    "component_name": top_key,
                    "component_description": name,
                    "childs": childs
                })
            else:
                comp_desc = None
                childs: List[Dict[str, Any]] = []

                if isinstance(top_val, dict):
                    comp_desc = top_val.get("name")
                    for child_key, child_val in top_val.items():
                        if child_key == "name":
                            continue
                        if isinstance(child_val, str):
                            child_desc = child_val
                        elif isinstance(child_val, dict):
                            child_desc = child_val.get("name")
                        else:
                            child_desc = None
                        childs.append({
                            "component_name": child_key,
                            "component_description": child_desc,
                            "childs": []
                        })
                elif isinstance(top_val, str):
                    comp_desc = top_val

                components.append({
                    "component_name": top_key,
                    "component_description": comp_desc,
                    "childs": childs
                })

        return components

    def _run(
        self,
        contract_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        with self.neo4j_driver.session() as session:
            result = session.run(self.CYPHER_QUERY, {"corpus_id": contract_id})
            record = result.single()
            toc_raw = record["table_of_contents"] if record else None
            toc_parsed = None
            if isinstance(toc_raw, str):
                try:
                    toc_parsed = json.loads(toc_raw)
                except Exception:
                    toc_parsed = toc_raw
            else:
                toc_parsed = toc_raw
            converted_toc = self._convert_toc_to_components(toc_parsed)
            return {"contract_id": contract_id, "table_of_contents": converted_toc}

    async def _arun(
        self,
        contract_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(contract_id)


