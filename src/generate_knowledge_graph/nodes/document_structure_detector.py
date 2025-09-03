import re
import json
import difflib
from langchain_core.prompts import ChatPromptTemplate
from generate_knowledge_graph.utils import *
from logger import setup_logger
from langgraph.types import Command
from langgraph.runtime import Runtime

from generate_knowledge_graph.utils.callback import BatchCallback
from generate_knowledge_graph.state import ContextSchema

logger = setup_logger()


SYSTEM_TEMPLATE = """You must organize the contents of the legal contract into a hierarchical structure according to the given table of contents. The legal contract is being read sequentially, page by page. At this stage, you need to identify which parts of the current page’s content correspond to the items in the table of contents.

The provided data will be as follows:
	•	<Table_of_Contents>: The table of contents of the legal contract. You must organize the contract contents according to this structure.
	•	<Entries_identified_so_far>: The items from the table of contents that have been identified up to the previous page.
	•	<Previous_Page>: The content of the previous page. Use this to determine whether the current page’s content continues from it.
	•	<Current_Page>: The content of the current page. You must identify which parts of this correspond to the items in the table of contents.

Your answer must be in the following format:
```json
{{
“Article_I”: {{
“section_1_1”: {{
“start_sentence”: “The first sentence corresponding to section_1_1 under Article_I. This must be copied verbatim from <Current_Page>.”,
“end_sentence”: “The last sentence corresponding to section_1_1 under Article_I. This must be copied verbatim from <Current_Page>.”
}},
…
}},
…
}}
```
"""

USER_TEMPLATE = """<Table_of_Contents>
{table_of_contents}
</Table_of_Contents>

<Entries_identified_so_far>
{entries_identified_so_far}
</Entries_identified_so_far>

<Previous_Page>
{previous_page}
</Previous_Page>

<Current_Page>
{current_page}
</Current_Page>
"""

def _best_window_by_words(current_page: str, target_sentence: str):
    token_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', current_page)]
    words_in_target = re.findall(r'\S+', target_sentence)
    window_words = len(words_in_target)
    if window_words <= 0 or not token_spans:
        return 0, 0, 0.0

    best_score = -1.0
    best_start_char, best_end_char = 0, 0
    cp_lower = current_page.lower()
    target_lower = target_sentence.lower()

    max_start = len(token_spans) - window_words
    for i in range(max_start + 1):
        w_start = token_spans[i][0]
        w_end = token_spans[i + window_words - 1][1]
        window_text = cp_lower[w_start:w_end]
        score = difflib.SequenceMatcher(None, target_lower, window_text).ratio()
        if score > best_score:
            best_score = score
            best_start_char, best_end_char = w_start, w_end

    return best_start_char, best_end_char, best_score

def find_sentence_range(current_page: str, start_sentence: str, end_sentence: str):
    s_start, s_end, _ = _best_window_by_words(current_page, start_sentence)
    e_start, e_end, _ = _best_window_by_words(current_page, end_sentence)
    return s_start, e_end


class DocumentStructureDetector:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state, runtime: Runtime[ContextSchema]):
        chain = runtime.context.document_structure_detector_prompt | self.llm | JsonOutputParser()
        # 각 path마다 하나의 Chunk를 만들기 위한 버킷
        path_level_chunks = []
        previous_chunk = None
        entries_identified_so_far = []

        # file_path → 원문 Document content 매핑 구성
        structured_chunks = {}

        for chunk in state.chunks:
            if previous_chunk and previous_chunk.file_path != chunk.file_path:
                previous_chunk = None
                entries_identified_so_far = []

            
                    
            response = chain.invoke(
                {
                    "file_path": chunk.file_path,
                    "table_of_contents": state.table_of_contents[chunk.file_path],
                    "latest_entry": entries_identified_so_far[-1] if entries_identified_so_far else None,
                    # "previous_page": previous_chunk.content if previous_chunk else "",
                    "current_page": chunk.content,
                }
            )

            def find_key_paths(response_data, structured_chunks, path=[]):
                for key, value in response_data.items():
                    if "start_sentence" in value and "end_sentence" in value:
                        local_start, local_end = find_sentence_range(
                            chunk.content, value["start_sentence"], value["end_sentence"]
                        )
                        abs_start = chunk.span[0] + local_start
                        abs_end = chunk.span[0] + local_end

                        if key not in structured_chunks:
                            structured_chunks[key] = []

                        structured_chunks[key].append(
                            Chunk(
                                file_path=chunk.file_path,
                                span=(abs_start, abs_end),
                                content=chunk.content[local_start:local_end],
                            )
                        )
                        entries_identified_so_far.append(path + [key])
                    else:
                        if key not in structured_chunks:
                            structured_chunks[key] = {}
                        
                        find_key_paths(value, structured_chunks[key], path + [key])

            if chunk.file_path not in structured_chunks:
                structured_chunks[chunk.file_path] = {}

            find_key_paths(response, structured_chunks[chunk.file_path])
        
        # path 단위로 생성한 Chunk 목록을 다음 단계로 전달
        return Command(
            update={
                "structured_chunks": structured_chunks,
            },
            goto="Summarizer"
        )