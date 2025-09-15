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
        documents = getattr(state, "documents", []) or []

        def walk_chunks(tree):
            # dict 트리: 값으로 저장된 최상위 Chunk들을 순회
            if isinstance(tree, dict):
                for v in (tree or {}).values():
                    yield from walk_chunks(v)
                return
            # Chunk 노드
            if hasattr(tree, "children") and isinstance(getattr(tree, "children"), list):
                yield tree
                for child in tree.children or []:
                    yield from walk_chunks(child)

        def collect_leaves(tree):
            leaves = []
            for node in walk_chunks(tree):
                if not getattr(node, "children", None):
                    leaves.append(node)
            return leaves

        def collect_parents_by_depth(tree):
            # 깊이 계산: 루트 0부터, 자식은 +1. 리프 제외
            depth_map = {}
            def dfs(node, depth):
                if not getattr(node, "children", None):
                    return
                depth_map.setdefault(depth, []).append(node)
                for ch in node.children or []:
                    dfs(ch, depth + 1)
            if isinstance(tree, dict):
                for v in (tree or {}).values():
                    dfs(v, 0)
            else:
                dfs(tree, 0)
            return depth_map

        def gather_nodes_at_level(tree, level):
            return [ch for ch in iter_chunks_from_tree(tree or []) if int(getattr(ch, "level", 0) or 0) == level]

        def batch_run(inputs, batch_size=16, desc="Summarizer"):
            outputs = []
            if not inputs:
                return outputs
            with BatchCallback(total=len(inputs), desc=desc) as cb:
                outputs = chain.batch(inputs, config={"callbacks": [cb], "max_concurrency": batch_size})
            return outputs

        def compute_max_depth(tree):
            max_depth = 0
            def dfs(node, depth):
                nonlocal max_depth
                if isinstance(node, dict):
                    for v in (node or {}).values():
                        dfs(v, depth)
                    return
                # node is a Chunk
                if depth > max_depth:
                    max_depth = depth
                for ch in (getattr(node, "children", []) or []):
                    dfs(ch, depth + 1)
            if isinstance(tree, dict):
                for v in (tree or {}).values():
                    dfs(v, 0)
            else:
                dfs(tree, 0)
            return max_depth

        # 레벨(층) 정보 계산: 루트=0, 리프=최대 깊이
        global_max_depth = 0
        for d in documents:
            global_max_depth = max(global_max_depth, compute_max_depth(getattr(d, "children", {}) or {}))
        total_layers = global_max_depth + 1

        # 1) 모든 문서의 리프를 요약(배치)
        leaf_inputs = []
        leaf_nodes = []
        for d in documents:
            for leaf in collect_leaves(getattr(d, "children", {}) or {}):
                leaf_inputs.append({"contents": leaf.content})
                leaf_nodes.append(leaf)
        if leaf_inputs:
            # 리프는 최하단 레벨로 표시
            leaf_summaries = batch_run(
                leaf_inputs,
                desc=f"Summarizer L{global_max_depth + 1}/{total_layers}"
            )
            for node, summary in zip(leaf_nodes, leaf_summaries):
                node.summary = summary

        # 2) 부모들을 바텀업으로 요약(깊은 레벨부터)
        # 문서마다 깊이 맵 생성 후, 가장 깊은 레벨부터 위로 올라가며 요약
        for d in documents:
            depth_map = collect_parents_by_depth(getattr(d, "children", {}) or {})
            for depth in sorted(depth_map.keys(), reverse=True):
                parents = depth_map[depth]
                if not parents:
                    continue
                inputs = []
                for node in parents:
                    child_text = "\n\n".join(
                        [
                            (getattr(c, "summary", None) or getattr(c, "content", "") or "").strip()
                            for c in (getattr(node, "children", []) or [])
                        ]
                    )
                    inputs.append({"contents": child_text})
                # 부모는 해당 깊이(0부터 시작)를 1-based로 표기
                summaries = batch_run(
                    inputs,
                    desc=f"Summarizer L{depth + 1}/{total_layers}"
                )
                for node, summary in zip(parents, summaries):
                    node.summary = summary

        # 3) 문서 레벨 요약(배치 처리)
        doc_inputs = []
        doc_refs = []
        for d in documents:
            top_children = getattr(d, "children", {}) or {}
            top_chunks = list(top_children.values())
            contents = "\n\n".join([ (getattr(c, "summary", None) or getattr(c, "content", "") or "").strip() for c in top_chunks ])
            doc_inputs.append({"contents": contents})
            doc_refs.append(d)
        if doc_inputs:
            doc_summaries = batch_run(doc_inputs)
            for d, s in zip(doc_refs, doc_summaries):
                d.summary = s

        return Command(update={"documents": documents}, goto="GraphDBWriter")
