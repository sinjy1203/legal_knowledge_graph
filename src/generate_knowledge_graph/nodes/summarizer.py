from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command

from generate_knowledge_graph.utils.parser import JsonOutputParser
from generate_knowledge_graph.utils.callback import BatchCallback
from generate_knowledge_graph.state import ContextSchema
from langgraph.runtime import Runtime
from logger import setup_logger

logger = setup_logger()

SYSTEM_TEMPLATE = """You are a legal contract analysis expert. Your task is to summarize the given contents in 2–3 sentences.
"""

USER_TEMPLATE = """
<Contents>
{contents}
</Contents>

summary:"""


class Summarizer:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state, runtime: Runtime[ContextSchema]):
        chain = runtime.context.summarizer_prompt | self.llm | StrOutputParser()

        structured_chunks = state.structured_chunks or {}

        def summarize_leaf(chunks_leaf):
            contents = "\n\n".join([(getattr(c, "summary", None) or c.content or "").strip() for c in chunks_leaf]).strip()
            if not contents:
                return ""
            return chain.invoke({"contents": contents})

        def summarize_node(node):
            # leaf: list[Chunk]
            if isinstance(node, list):
                summary = summarize_leaf(node)
                return {"chunks": node, "summary": summary}
            # dict: summarize children then self
            if isinstance(node, dict):
                child_summaries = []
                updated = {}
                for key, child in node.items():
                    summarized_child = summarize_node(child)
                    updated[key] = summarized_child
                    # 자식 summary 수집
                    child_summary = summarized_child.get("summary") if isinstance(summarized_child, dict) else None
                    if child_summary:
                        child_summaries.append(child_summary)
                # 부모 요약 생성
                parent_contents = "\n\n".join(child_summaries).strip()
                parent_summary = chain.invoke({"contents": parent_contents}) if parent_contents else ""
                updated["summary"] = parent_summary
                return updated
            return node

        # 파일 경로 루트별로 처리
        summarized_structured = {}
        for file_path, tree in structured_chunks.items():
            summarized_structured[file_path] = summarize_node(tree)

        return Command(
            update={
                "structured_chunks": summarized_structured,
            },
            goto="GraphDBWriter",
        )
