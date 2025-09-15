import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph
from generate_knowledge_graph.state import State, ContextSchema
from generate_knowledge_graph.nodes import *
from generate_knowledge_graph.utils.database import Neo4jConnection

load_dotenv(override=True)

# LLM 설정
llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    # temperature=0.0,
    api_key=os.getenv("LLM_API_KEY"),
    max_tokens=32768
)

# reasoning_llm = ChatOpenAI(
#     base_url=os.getenv("REASONING_LLM_BASE_URL"),
#     model=os.getenv("REASONING_LLM_MODEL"),
#     api_key=os.getenv("REASONING_LLM_API_KEY"),
#     # reasoning={
#     #     "effort": "high"
#     # }
#     # temperature=0.6,
#     # top_p=0.95,
#     # extra_body={
#     #     "top_k": 20,
#     #     "min_p": 0
#     # }
# )

embedding_model = OpenAIEmbeddings(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("EMBEDDING_API_KEY")
)

# Neo4j 설정 (임베딩 모델 포함)
neo4j_client = Neo4jConnection(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASSWORD"),
    embedding_model=embedding_model
)

# 워크플로우 생성
workflow = StateGraph(State, context_schema=ContextSchema)

# 노드 추가
workflow.add_node("DataLoader", DataLoader())
workflow.add_node("IntroBodySeparator", IntroBodySeparator(llm))
workflow.add_node("TableOfContentsExtractor", TableOfContentsExtractor(llm))
workflow.add_node("Chunker", Chunker(llm))
workflow.add_node("Summarizer", Summarizer(llm))
workflow.add_node("GraphDBWriter", GraphDBWriter(neo4j_client))


# 엣지 추가
workflow.add_edge("__start__", "DataLoader")

# 워크플로우 컴파일
graph = workflow.compile()
graph.name = "generate_knowledge_graph"
