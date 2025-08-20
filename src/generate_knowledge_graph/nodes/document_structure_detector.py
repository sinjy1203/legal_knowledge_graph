import json
from langchain_core.prompts import ChatPromptTemplate
from generate_knowledge_graph.utils import *
from logger import setup_logger
from langgraph.types import Command
from langgraph.runtime import Runtime

from generate_knowledge_graph.utils.callback import BatchCallback
from generate_knowledge_graph.state import ContextSchema

logger = setup_logger()


SYSTEM_TEMPLATE = """You are a document analysis expert specializing in legal contracts.  
Your task is to determine which numbered section(s) a given contract chunk belongs to, based on:
- the chunk content
- the previous chunk
- the section numbers of the previous chunk

You must also provide a concise summary of the chunk.  

## Contract Structure
- The first-level hierarchy is **ARTICLE** (e.g., "ARTICLE I", "ARTICLE II", ...).
- The second-level hierarchy is **Section** within an ARTICLE (e.g., "Section 1.1", "Section 1.2", ...).
- **Do not go deeper than the second level**. Ignore subsections like "Section 1.1.1" and treat their content as part of their parent Section.

## Input Information
- Full text of the current chunk
- Full text of the previous chunk
- The list of section numbers that the previous chunk belongs to (from parent to child), e.g.:  
  `[["ARTICLE I", "Section 1.1"], ["ARTICLE I", "Section 1.2"]]`

## Analysis Rules
1. **Numbered Section Detection**
   - Identify any ARTICLE or Section numbers in the current chunk.
   - If an ARTICLE is found, it becomes the new top-level section, and all subsequent Sections belong under it until a new ARTICLE is encountered.
   - If a Section is found, its parent is the current ARTICLE.
   - If multiple Sections of the same level are present in a single chunk, record each as a separate path.  
     Example: `"Section 1.1"` and `"Section 1.2"` in the same chunk →  
     `[["ARTICLE I", "Section 1.1"], ["ARTICLE I", "Section 1.2"]]`

2. **No Section Number**
   - If the chunk does not begin with a section number, assume it belongs to the **last (deepest) section** from the previous chunk.  
     Example: If the previous chunk belongs to `[["ARTICLE I", "Section 1.1"]]`, then the current chunk belongs to `["ARTICLE I", "Section 1.1"]`.

3. **Summary Creation**
   - Summarize the current chunk in **1–2 concise sentences**.

## Output Format (JSON)
Always return the output strictly in the following JSON format:
```json
{{
  "section_numbers": [
    ["parent_number", "child_number"],
    ["parent_number", "another_child_number"]
  ],
  "summary": "1–2 sentence concise summary of the current chunk"
}}
```
"""

USER_TEMPLATE = """
<current_chunk>
{current_chunk}
</current_chunk>

<previous_chunk>
{previous_chunk}
</previous_chunk>

<previous_chunk_section_numbers>
{previous_chunk_section_numbers}
</previous_chunk_section_numbers>
"""


class DocumentStructureDetector:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state, runtime: Runtime[ContextSchema]):
        chain = runtime.context.document_structure_detector_prompt | self.llm | JsonOutputParser()
        previous_chunk_clause_list = []
        previous_chunk_file_path = None
        
        # table_of_contents_flag = False
        # merger_flag = False
        hierarchical_chunk_ids = {}

        for chunk_idx, chunk in enumerate(state.chunks):
            chunk.content = chunk.content.strip()
            if chunk.content == "":
                continue

            if previous_chunk_file_path != chunk.file_path:
                previous_chunk_clause_list = []
                previous_chunk_file_path = chunk.file_path
            
            # if not table_of_contents_flag and "TABLE OF CONTENTS" in chunk.content.upper():
            #     table_of_contents_flag = True
            
            # if not merger_flag and table_of_contents_flag and "AGREEMENT AND PLAN OF MERGER" in chunk.content.upper():
            #     merger_flag = True
            
            # if not (table_of_contents_flag and merger_flag):
            #     continue
                    
            response = chain.invoke(
                {
                    "file_path": chunk.file_path,
                    "table_of_contents": state.table_of_contents[chunk.file_path],
                    "identified_clauses": previous_chunk_clause_list,
                    "latest_identified_clause": previous_chunk_clause_list[-1] if len(previous_chunk_clause_list) > 0 else "",
                    "current_chunk_content": chunk.content,
                }
            )

            chunk_summary = response["chunk_summary"]
            if len(response["chunk_clause_list"]) > 0:
                if len(previous_chunk_clause_list) == 0 and response["chunk_clause_list"][0] != "1.1":
                    pass
                else:
                    chunk_clause_list = response["chunk_clause_list"]
            else:
                chunk_clause_list = previous_chunk_clause_list[-1:]

            if len(chunk_clause_list) != 0:
                chunk.summary = chunk_summary
                # Update hierarchical_chunk_ids with current chunk index for each detected path under file_path
                current_file_path = getattr(chunk, "file_path", None)
                if not current_file_path:
                    # Skip if file_path is unavailable
                    pass
                else:
                    if current_file_path not in hierarchical_chunk_ids:
                        hierarchical_chunk_ids[current_file_path] = {}

                    for chunk_clause in chunk_clause_list:
                        parent_number, child_number = chunk_clause.split(".")
                        parent_number = f"Article {parent_number}"
                        child_number = f"Section {child_number}"

                        # Initialize nested structure if missing
                        if parent_number not in hierarchical_chunk_ids[current_file_path]:
                            hierarchical_chunk_ids[current_file_path][parent_number] = {}
                        if child_number not in hierarchical_chunk_ids[current_file_path][parent_number]:
                            hierarchical_chunk_ids[current_file_path][parent_number][child_number] = []

                        # Append the current chunk index
                        hierarchical_chunk_ids[current_file_path][parent_number][child_number].append(chunk_idx)

                        if chunk_clause not in previous_chunk_clause_list:
                            previous_chunk_clause_list.append(chunk_clause)

        
        return Command(
            update={
                "chunks": state.chunks,
                "hierarchical_chunk_ids": hierarchical_chunk_ids
            },
            goto="Summarizer"
        )