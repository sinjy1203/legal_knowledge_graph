import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from search_knowledge_graph.tools import *
from neo4j import GraphDatabase


load_dotenv(override=True)

embedding_model = OpenAIEmbeddings(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    model=os.getenv("EMBEDDING_MODEL"),
    api_key="dummy"
)
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# tool = SearchCorpusTool(neo4j_driver)
# result = tool.invoke({})

# tool = SearchArticleTool(neo4j_driver, embedding_model)
# result = tool.invoke({"corpus_id": "a4346665-4dce-4aab-97f8-e1dbe089f10e", "query": "term description"})

# tool = SearchSectionTool(neo4j_driver, embedding_model)
# result = tool.invoke({"article_id": "62f11699-1452-44a1-8b33-1c18951e5007", "query": "Affiliate description"})

# tool = SearchChunkTool(neo4j_driver, embedding_model)
# result = tool.invoke({"section_id": "df745afb-79d3-42d4-8b93-fabe4a888b7d", "query": "Affiliate description"})

tool = ResponseTool(neo4j_driver)
result = tool.invoke({"chunk_ids": ["a088e71d-6fcd-42a1-b281-35075fc44c00", "4281ea0d-4e84-4da8-aeef-0c714596f0d7"]})

print(json.dumps(result, indent=4))