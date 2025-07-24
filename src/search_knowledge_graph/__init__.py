import os
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from .agent import ReactAgent
from .tools import *


embedding_model = OpenAIEmbeddings(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    model=os.getenv("EMBEDDING_MODEL"),
    api_key="dummy"
)
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

final_tool = GetChunkInfoTool(neo4j_driver)

agent = ReactAgent(
    model_kwargs={
        "base_url": os.getenv("LLM_BASE_URL"),
        "model": os.getenv("LLM_MODEL"),
        "temperature": 0.1,
        "api_key": "dummy"
    },
    tools=[
        SearchEntityTool(neo4j_driver, embedding_model),
        GetMentionChunkTool(neo4j_driver),
        SearchRelationshipTool(neo4j_driver, embedding_model),
        SearchConnectedEntityTool(neo4j_driver, embedding_model),
        final_tool
    ],
    graph_schema=final_tool.graph_schema
)