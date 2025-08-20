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

        hierarchical_chunk_ids = state.hierarchical_chunk_ids or {}
        chunks = state.chunks or []

        # Bottom-up summarization using batch with progress bars
        for file_path, article_map in hierarchical_chunk_ids.items():
            # 1) Section summaries from chunk summaries (batched)
            section_tasks = []  # list[(article_name, section_name)]
            section_inputs = []  # list[{"contents": str}]
            for article_name, sections_map in list(article_map.items()):
                for section_name, section_value in list(sections_map.items()):
                    if section_name == "summary":
                        continue
                    # Support both legacy list format and new dict format
                    if isinstance(section_value, dict):
                        chunk_idx_list = section_value.get("chunk_ids", [])
                    else:
                        chunk_idx_list = section_value

                    # Build contents from chunk summaries (fallback to content)
                    contents_parts = []
                    for idx in chunk_idx_list:
                        if not (isinstance(idx, int) and 0 <= idx < len(chunks)):
                            continue
                        chunk_obj = chunks[idx]
                        chunk_summary = (getattr(chunk_obj, "summary", None) or "").strip()
                        if chunk_summary:
                            contents_parts.append(chunk_summary)
                        else:
                            chunk_content = (getattr(chunk_obj, "content", None) or "").strip()
                            if chunk_content:
                                contents_parts.append(chunk_content)

                    contents = "\n\n".join(contents_parts).strip()

                    # Ensure dict structure exists for section
                    if not isinstance(section_value, dict):
                        sections_map[section_name] = {
                            "chunk_ids": list(chunk_idx_list) if isinstance(chunk_idx_list, (list, tuple)) else [],
                        }
                    else:
                        if "chunk_ids" not in sections_map[section_name]:
                            sections_map[section_name]["chunk_ids"] = list(chunk_idx_list) if isinstance(chunk_idx_list, (list, tuple)) else []

                    if contents:
                        section_tasks.append((article_name, section_name))
                        section_inputs.append({"contents": contents})
                    else:
                        sections_map[section_name]["summary"] = ""

            if section_inputs:
                with BatchCallback(total=len(section_inputs)) as cb:
                    section_summaries = chain.batch(section_inputs, config={"callbacks": [cb]})
                for (article_name, section_name), summary in zip(section_tasks, section_summaries):
                    hierarchical_chunk_ids[file_path][article_name][section_name]["summary"] = summary

            # 2) Article summary from section summaries (batched)
            article_tasks = []  # list[article_name]
            article_inputs = []  # list[{"contents": str}]
            for article_name, sections_map in list(article_map.items()):
                section_summaries_for_article = []
                for section_name, section_value in sections_map.items():
                    if section_name == "summary":
                        continue
                    if isinstance(section_value, dict):
                        s = section_value.get("summary")
                        if s:
                            section_summaries_for_article.append(s)

                article_contents = "\n\n".join(section_summaries_for_article).strip()
                if article_contents:
                    article_tasks.append(article_name)
                    article_inputs.append({"contents": article_contents})
                else:
                    sections_map["summary"] = ""

            if article_inputs:
                with BatchCallback(total=len(article_inputs)) as cb:
                    article_summaries = chain.batch(article_inputs, config={"callbacks": [cb]})
                for article_name, summary in zip(article_tasks, article_summaries):
                    hierarchical_chunk_ids[file_path][article_name]["summary"] = summary

        return Command(
            update={
                "hierarchical_chunk_ids": hierarchical_chunk_ids
            },
            goto="GraphDBWriter",
        )
