from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from generate_knowledge_graph.utils.cluster import cluster_entities
from generate_knowledge_graph.utils.parser import JsonOutputParser
from generate_knowledge_graph.utils.callback import BatchCallback
from logger import setup_logger
from .entity_relation_extractor import ENTITY_TYPES

logger = setup_logger()


SYSTEM_TEMPLATE = """
You are an expert in Entity Resolution.
Please identify entities that refer to the same real-world object but are expressed differently, and unify them to resolve entity consistency.

Please respond in the given <Json_Response_Format>

<JSON_Response_Format>
{{
  "resolved_entities": [
    {{
      "original_name": Name of the entity that needs to be corrected
      "resolved_name": The name of the resolved entity
    }},
    ...
  ]
}}
</Json_Response_Format>
"""

USER_TEMPLATE = """
<Entity_Type>
{entity_type}
</Entity_Type>

<Entities>
{entities}
</Entities>
"""



class EntityResolver:
    def __init__(self, neo4j_conn, llm):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_TEMPLATE),
                ("user", USER_TEMPLATE)
            ]
        )

        self.chain = prompt | llm | JsonOutputParser()
        self.neo4j_conn = neo4j_conn
        

    def __call__(self, state):
        logger.info("resolving entities...")
        for node_type in ENTITY_TYPES:
            entity_vectors = self.neo4j_conn.get_node_vectors_by_type(node_type)
            if len(entity_vectors) <= 1:
                continue

            entities = list(entity_vectors.keys())
            entity_vectors = list(entity_vectors.values())

            clusters = cluster_entities(entities, entity_vectors)

            queries = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                queries.append(
                    {
                        "entity_type": node_type,
                        "entities": cluster
                    }
                )

            with BatchCallback(len(queries)) as cb:
                responses = self.chain.batch(queries, config={"max_concurrency": 8, "callbacks": [cb]})
                
            all_resolved_entities = []
            for cluser_idx, response in enumerate(responses):
                if not response or "resolved_entities" not in response:
                    continue

                all_resolved_entities.extend(response["resolved_entities"])

            update_count = self.neo4j_conn.update_entity_names(node_type, all_resolved_entities)
            logger.info(f"updated {update_count} entities for {node_type}")
        
        return Command(
            update={},
            goto="__end__"
        )
