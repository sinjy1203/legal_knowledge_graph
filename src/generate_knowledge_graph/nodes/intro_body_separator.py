import json
from langchain_core.prompts import ChatPromptTemplate
from logger import setup_logger
from langgraph.types import Command
from langgraph.runtime import Runtime

from generate_knowledge_graph.utils.callback import BatchCallback
from generate_knowledge_graph.state import ContextSchema
from generate_knowledge_graph.utils import JsonOutputParser

logger = setup_logger()


SYSTEM_TEMPLATE = """
You are a document analysis expert specializing in legal contracts.
Your task is to locate the sentence that appears just before the start of the main text of ARTICLE I, following the table of contents and basic contract information. Typically, the sentence ‘NOW, THEREFORE, in consideration of the…’ appears immediately before ARTICLE I begins. Therefore, you need to identify a sentence matching this pattern.
# Output Format
{{
"contract_start_sentence": "The exact sentence from the contract that appears just before the start of the main text of ARTICLE I"
}}

"""

USER_TEMPLATE = """
<Legal_Contract>
{legal_contract}
</Legal_Contract>
"""


class IntroBodySeparator:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state, runtime: Runtime[ContextSchema]):
        # chain = runtime.context.contract_start_finder_prompt | self.llm | JsonOutputParser()
        # queries = [{"legal_contract": document.content.strip()[:10000]} for document in state.documents]

        # with BatchCallback(total=len(queries), desc="Contract Start Finder") as cb:
        #     responses = chain.batch(queries, config={"callbacks": [cb]})
        
        # contract_start_sentence = {document.file_path: response["contract_start_sentence"] for document, response in zip(state.documents, responses)}
        
        # Filter text after 'follows:' for each document
        for document in state.documents:
            content = (document.content or "")
            lower_content = content.lower()
            marker = "follows:"
            idx = lower_content.find(marker)
            if idx != -1:
                # 본문(body): 'follows:' 이후
                start = idx + len(marker)
                end = len(content)
                filtered = content[start:]
                document.body = filtered
                # 인트로(intro): 'follows:' 이전
                document.intro = content[:idx]
                # 본문 구간 span 저장 (원본 텍스트 기준 인덱스)
                document.body_span = (start, end)
            else:
                # 구분자가 없으면 본문 전체로 간주하고 인트로는 빈 문자열
                filtered = content
                document.body = content
                document.intro = ""
                document.body_span = (0, len(content))

        return Command(
            update={
                "documents": state.documents,
            },
            goto="TableOfContentsExtractor"
        )