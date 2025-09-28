import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from .agent import ReactAgent
from .tools import *

load_dotenv(override=True)


embedding_model = OpenAIEmbeddings(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("EMBEDDING_API_KEY")
)
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

agent = ReactAgent(
    model_kwargs={
        "base_url": os.getenv("LLM_BASE_URL"),
        "model": os.getenv("LLM_MODEL"),
        # "model": os.getenv("REASONING_LLM_MODEL"),
        # "temperature": 0.1,
        "api_key": os.getenv("LLM_API_KEY")
    },
    tools=[
        SearchCorpusTool(neo4j_driver),
        GetCorpusTOCTool(neo4j_driver),
        SearchComponentTool(neo4j_driver, embedding_model),
        SearchSubComponentTool(neo4j_driver, embedding_model),
        # SearchNeighborChunkTool(neo4j_driver),
        ResponseTool(neo4j_driver)
    ]
)
