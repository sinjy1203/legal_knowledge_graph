import os
import re
from logger import setup_logger
import difflib
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain_core.prompts import ChatPromptTemplate
import pickle

from generate_knowledge_graph.utils.model import Chunk
from generate_knowledge_graph.utils.parser import JsonOutputParser
from generate_knowledge_graph.utils.callback import BatchCallback
from generate_knowledge_graph.state import ContextSchema


logger = setup_logger()

SYSTEM_TEMPLATE = """You are a legal contract document analysis assistant. Your task is to split the content of the legal contract according to the Table of Contents.

Inputs:  
- Table_of_Contents  
- Legal_Contract

Output Format (must follow exactly):  
{{  
  "table_of_contents_key_1": {{  
    "table_of_contents_key_1_1": {{  
      "start_sentence": "First sentence for table_of_contents_key_1_1 (copied verbatim from Legal_Contract, must contain at least 4 words).",  
      "end_sentence": "Last sentence for table_of_contents_key_1_1 (copied verbatim from Legal_Contract, must contain at least 4 words)."  
    }},  
    "table_of_contents_key_1_2": {{  
      "start_sentence": "First sentence for table_of_contents_key_1_2 (copied verbatim, must contain at least 4 words).",  
      "end_sentence": "Last sentence for table_of_contents_key_1_2 (copied verbatim, must contain at least 4 words)."  
    }}  
  }}  
}}"""

USER_TEMPLATE = """<Table_of_Contents>
{table_of_contents}
</Table_of_Contents>

<Legal_Contract>
{legal_contract}
</Legal_Contract>
"""


# SYSTEM_TEMPLATE = """You are a legal contract document analysis assistant. Your task is to split the content of the legal contract according to the Table of Contents. 

# CRITICAL REQUIREMENTS:
# - You MUST process EVERY single item listed in the Table of Contents
# - Do NOT skip any sections, articles, or subsections 
# - Before finalizing your response, verify that your output contains ALL items from the Table of Contents
# - If the legal contract is long, work systematically through each section in order

# Inputs: 
# - Table_of_Contents 
# - Legal_Contract 

# Output Format (must follow exactly): 
# {{ 
#     "table_of_contents_key_1": {{ 
#         "table_of_contents_key_1_1": {{ 
#             "start_sentence": "First sentence for table_of_contents_key_1_1 (copied verbatim from Legal_Contract, must contain at least 4 words).", 
#             "end_sentence": "Last sentence for table_of_contents_key_1_1 (copied verbatim from Legal_Contract, must contain at least 4 words)." 
#         }}, 
#         "table_of_contents_key_1_2": {{ 
#             "start_sentence": "First sentence for table_of_contents_key_1_2 (copied verbatim, must contain at least 4 words).", 
#             "end_sentence": "Last sentence for table_of_contents_key_1_2 (copied verbatim, must contain at least 4 words)." 
#         }} 
#     }} 
# }}

# VALIDATION CHECKLIST:
# - Count the items in Table of Contents: [X] items
# - Count the items in your output: [X] items  
# - These numbers MUST match exactly
# - If they don't match, you have missed sections and must include them"""

# USER_TEMPLATE = """<Table_of_Contents>
# {table_of_contents}
# </Table_of_Contents>

# <Legal_Contract>
# {legal_contract}
# </Legal_Contract>

# REMINDER: Process ALL sections listed in the Table of Contents above. Do not stop at early sections - continue through the entire document to capture every article, section, and subsection listed."""


def _best_window_by_words(content: str, target_sentence: str):
    token_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', content)]
    words_in_target = re.findall(r'\S+', target_sentence)
    window_words = len(words_in_target)
    if window_words <= 0 or not token_spans:
        return 0, 0, 0.0

    best_score = -1.0
    best_start_char, best_end_char = 0, 0
    cp_lower = content.lower()
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


def find_sentence_range(content: str, start_sentence: str, end_sentence: str):
    s_start, s_end, _ = _best_window_by_words(content, start_sentence)
    e_start, e_end, _ = _best_window_by_words(content, end_sentence)
    return s_start, e_end


class Chunker:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state, runtime: Runtime[ContextSchema]):
        # 캐시 로드 시도
        cache_path = os.path.join("./data/cache", "chunker_documents.pkl")
        if getattr(runtime.context, "use_cache", False) and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    documents = pickle.load(f)
                logger.info(f"Loaded documents from cache: {cache_path}")
                return Command(update={"documents": documents}, goto="Summarizer")
            except Exception as e:
                logger.error(f"Failed to load Chunker cache. Fallback to generation. err={e}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("user", USER_TEMPLATE),
        ])

        chain = prompt | self.llm | JsonOutputParser()
        queries = [{"table_of_contents": doc.table_of_contents, "legal_contract": doc.content} for doc in state.documents]

        with BatchCallback(total=len(queries), desc="Chunker") as cb:
            responses = chain.batch(queries, config={"callbacks": [cb], "max_concurrency": 4})

        def transform_tree(node, content: str, name: str = ""):
            # 리프 판단: start_sentence/end_sentence를 가진 dict
            if isinstance(node, dict) and "start_sentence" in node and "end_sentence" in node \
               and isinstance(node["start_sentence"], str) and isinstance(node["end_sentence"], str):
                s, e = find_sentence_range(content, node["start_sentence"], node["end_sentence"])
                if e < s:
                    s, e = e, s
                s = max(0, min(s, len(content)))
                e = max(0, min(e, len(content)))
                text = content[s:e] if s < e else ""
                return Chunk(name=name, span=(s, e), content=text, children=[])

            # 내부 노드: 하위 key들로 children을 생성하고 content/span을 집계
            if isinstance(node, dict):
                children = []
                for k, v in node.items():
                    child_chunk = transform_tree(v, content, name=k)
                    if child_chunk is None:
                        continue
                    children.append(child_chunk)
                # 집계
                if children:
                    agg_start = min(c.span[0] for c in children if isinstance(c.span[0], int))
                    agg_end = max(c.span[1] for c in children if isinstance(c.span[1], int))
                    agg_content = "".join(c.content for c in children)
                else:
                    agg_start, agg_end, agg_content = 0, 0, ""
                return Chunk(name=name, span=(agg_start, agg_end), content=agg_content, children=children)

            # list는 무시(예상치 않음)
            return None

        documents = []
        for doc, resp in zip(state.documents, responses):
            try:
                transformed_root = {}
                if isinstance(resp, dict):
                    for top_key, subtree in resp.items():
                        ch = transform_tree(subtree, doc.content, name=top_key)
                        if ch is not None:
                            transformed_root[top_key] = ch
                else:
                    transformed_root = {}
            except Exception:
                transformed_root = {}
            doc.children = transformed_root
            documents.append(doc)

        # 캐시 저장 시도 (반환 직전)
        try:
            os.makedirs("./data/cache", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(documents, f)
            logger.info(f"Saved documents to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save Chunker cache: {e}")

        return Command(update={"documents": documents}, goto="Summarizer")
