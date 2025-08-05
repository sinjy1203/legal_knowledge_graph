import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from generate_knowledge_graph.utils.callback import BatchCallback
from logger import setup_logger
from langgraph.types import Command
from langgraph.runtime import Runtime

from generate_knowledge_graph.state import ContextSchema


logger = setup_logger()


class GraphDBWriter:
    def __init__(self, neo4j_client):
        self.neo4j_client = neo4j_client

    def __call__(self, state, runtime: Runtime[ContextSchema]):
        # 데이터베이스 초기화 여부 확인
        if runtime.context.clear_database:
            logger.info("Neo4j 데이터베이스 초기화 중...")
            self.neo4j_client.clear_database()
            
            # 벡터 인덱스 및 제약 조건 재설정
            logger.info("Neo4j 데이터베이스 인덱스 및 제약 조건 설정 중...")
            self.neo4j_client.setup_constraints()
            self.neo4j_client.setup_vector_indexes()
        
        logger.info("Neo4j 그래프 데이터베이스에 데이터 저장 중...")
    
        self.neo4j_client.create_nodes_and_relationships(state.chunks)      
        self.neo4j_client.close()
        
        logger.info("Neo4j 그래프 데이터베이스에 데이터 저장 완료")
        return Command(
            update={},
            goto="EntityResolver"
        )

