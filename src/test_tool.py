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

# tool = SearchEntityTool(neo4j_driver, embedding_model)
# result = tool.invoke({"entity_type": "MaterialAdverseEffectDefinition", "entity_name": "parent material"})

# tool = GetMentionChunkTool(neo4j_driver)
# result = tool.invoke({"entity_id": "Company", "entity_type": "TargetCompany"})

# tool = SearchRelationshipTool(neo4j_driver, embedding_model)
# result = tool.invoke({"relationship_type": "Acquires", "query": "Merger Consideration"})

# tool = SearchConnectedEntityTool(neo4j_driver, embedding_model)
# result = tool.invoke({"entity_id": "Parent", "entity_type": "Acquirer", "relationship_type": "Pays", "relationship_direction": "outgoing", "query": "Consideration"})

tool = GetChunkInfoTool(neo4j_driver)
result = tool.invoke({"chunk_ids": ["6be78fb3-a50d-476d-b4c0-784138dff93e", "bb047214-fa64-4de0-b4fe-0fb68ed95ced"]})

print(json.dumps(result, indent=4))