from tqdm.auto import tqdm
from uuid import uuid4
from neo4j import GraphDatabase
from generate_knowledge_graph.nodes.entity_relation_extractor import ENTITY_TYPES, RELATIONSHIP_TYPES


class Neo4jConnection:
    def __init__(self, uri, user, password, embedding_model):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = embedding_model
       
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
            # 문서 노드 제약조건 삭제
            node_types = ["Chunk"] + ENTITY_TYPES
            for node_type in node_types:
                session.run(f"DROP CONSTRAINT {node_type.lower()}_id_unique IF EXISTS")
            
            for node_type in node_types:
                session.run(f"DROP INDEX {node_type.lower()}_vector_index IF EXISTS")
            
            relationship_types = RELATIONSHIP_TYPES
            for rel_type in relationship_types:
                session.run(f"DROP INDEX {rel_type.lower()}_relationship_vector_index IF EXISTS")
            
            print("✅ 데이터베이스 초기화 완료")
    
    def batch_embed(self, texts, batch_size=4):
        embedding_vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"임베딩 벡터 생성 중..."):
            batch = texts[i:i + batch_size]
            embedding_vectors += self.embedding_model.embed_documents(batch)
        
        return embedding_vectors
    
    def setup_vector_indexes(self):
        """벡터 인덱스를 생성합니다."""
        print("🔧 문서 벡터 인덱스 설정 중...")
        with self.driver.session() as session:
            # 문서 노드 타입에 대한 벡터 인덱스
            node_types = ["Chunk"] + ENTITY_TYPES
            
            for node_type in node_types:
                session.run(f"""
                CREATE VECTOR INDEX {node_type.lower()}_vector_index IF NOT EXISTS
                FOR (n:{node_type})
                ON (n.vector)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: "cosine"
                    }}
                }}
                """)
                print(f"✅ {node_type} 노드 벡터 인덱스 생성 완료")
            
            # 관계 벡터 인덱스
            for rel_type in RELATIONSHIP_TYPES:
                session.run(f"""
                CREATE VECTOR INDEX {rel_type.lower()}_relationship_vector_index IF NOT EXISTS
                FOR ()-[r:{rel_type}]-()
                ON (r.vector)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: "cosine"
                    }}
                }}
                """)
                print(f"✅ {rel_type} 관계 벡터 인덱스 생성 완료")
            
        print("✅ 문서, 엔티티 및 관계 벡터 인덱스 설정 완료")
    
    def setup_constraints(self):
        """데이터베이스 제약 조건 설정"""
        with self.driver.session() as session:
            # 문서 노드 제약조건 생성
            node_types = ["Chunk"] + ENTITY_TYPES
            for node_type in node_types:
                session.run(f"""
                    CREATE CONSTRAINT {node_type.lower()}_id_unique IF NOT EXISTS
                    FOR (n:{node_type})
                    REQUIRE n.id IS UNIQUE
                """)
                print(f"✅ {node_type} ID 제약조건 생성 완료")
    
    def get_node_names_by_type(self, node_type):
        with self.driver.session() as session:
            query = f"""
                MATCH (n:{node_type})
                RETURN n.id AS id
                ORDER BY n.id
            """
            
            result = session.run(query)
            node_names = [record["id"] for record in result]
            
            return node_names
            
    def get_node_vectors_by_type(self, node_type):    
        with self.driver.session() as session:
            query = f"""
                MATCH (n:{node_type})
                RETURN n.id AS id, n.vector AS vector
                ORDER BY n.id
            """
            
            result = session.run(query)
            node_vectors = {record["id"]: record["vector"] for record in result}
            
            return node_vectors

    def create_nodes_and_relationships(self, chunks):       
        with self.driver.session() as session:         
            for chunk in tqdm(chunks, desc="Writing Nodes & Relationships", total=len(chunks)):
                if chunk.entities == [] and chunk.relationships == []:
                    continue
                
                chunk_vector = self.embedding_model.embed_query(chunk.content)
                chunk_id = str(uuid4())
                # 1. 문서 노드 생성 (벡터 포함)
                query = f"""
                    CREATE (n:Chunk {{
                        id: $id,
                        file_path: $file_path,
                        span: $span,
                        content: $content,
                        vector: $vector
                    }})
                """
                params = {
                    'id': chunk_id,
                    'file_path': chunk.file_path,
                    'span': chunk.span,
                    'content': chunk.content,
                    'vector': chunk_vector
                }
                
                session.run(query, params)
                
                # 2. 엔티티 노드 생성 & 엔티티 노드와 문서 노드 연결
                for entity in chunk.entities:
                    entity_type = entity['type']
                    entity_name = entity['name']
                    entity_vector = self.embedding_model.embed_query(entity_name)
                        
                    # 엔티티 노드 생성 (벡터 포함)
                    entity_query = f"""
                        MERGE (e:{entity_type} {{id: $id, vector: $vector}})
                    """
                    
                    entity_params = {
                        'id': entity_name,
                        'vector': entity_vector
                    }
                    
                    session.run(entity_query, entity_params)

                    # 문서와 엔티티 연결
                    relationship_query = f"""
                        MATCH (n1:Chunk {{id: $chunk_id}})
                        MATCH (n2:{entity_type} {{id: $entity_id}})
                        MERGE (n1)-[:MENTIONS]->(n2)
                    """
                    session.run(relationship_query, {
                        'chunk_id': chunk_id,
                        'entity_id': entity_name
                    })
                
                # 3. 엔티티 노드 간의 관계 생성
                for rel in chunk.relationships:
                    rel_type = rel['type']
                    source_entity = rel['source_entity']
                    source_type = rel['source_type']
                    target_entity = rel['target_entity']
                    target_type = rel['target_type']
                    description = rel['description']
                    vector = self.embedding_model.embed_query(description)
                    
                    # 관계 생성 (벡터 포함)
                    relationship_query = f"""
                    MATCH (e1:{source_type}) WHERE e1.id = $source_entity
                    MATCH (e2:{target_type}) WHERE e2.id = $target_entity
                    
                    // 동일한 문서에서 추출한 같은 타입의 관계가 이미 존재하는지 확인
                    OPTIONAL MATCH (e1)-[existing:{rel_type}]->(e2)
                    WHERE existing.chunk_id = $chunk_id
                    
                    // 존재하지 않는 경우에만 새 관계 생성
                    WITH e1, e2, existing
                    WHERE existing IS NULL
                    CREATE (e1)-[r:{rel_type}]->(e2)
                    SET r.description = $description,
                        r.chunk_id = $chunk_id,
                        r.vector = $vector
                    """
                    
                    rel_params = {
                        'source_entity': source_entity,
                        'target_entity': target_entity,
                        'description': description,
                        'chunk_id': chunk_id,
                        'vector': vector
                    }
                    
                    session.run(relationship_query, rel_params)
    
    def update_entity_names(self, node_type, resolved_entities):
        updates_count = 0
        with self.driver.session() as session:
            for entity in resolved_entities:
                original_name = entity.get("original_name")
                resolved_name = entity.get("resolved_name")
                
                # 원본 이름과 표준화된 이름이 같거나 둘 중 하나라도 없으면 처리하지 않음
                if not original_name or not resolved_name or original_name == resolved_name:
                    continue
                    
                # 1. 표준화된 이름의 노드가 이미 존재하는지 확인
                query = f"""
                    MATCH (target:{node_type} {{id: $resolved_name}})
                    RETURN target
                """
                result = session.run(query, {"resolved_name": resolved_name})
                target_exists = result.single() is not None
                
                if target_exists:
                    query = f"""
                        MATCH (source)-[r]->(original:{node_type} {{id: $original_name}})
                        WHERE NOT source:Entity OR labels(source)[0] <> '{node_type}' OR source.id <> $resolved_name
                        MATCH (target:{node_type} {{id: $resolved_name}})
                        WITH source, target, type(r) AS rel_type, properties(r) AS props
                        CALL apoc.create.relationship(source, rel_type, props, target) YIELD rel AS new_rel
                        RETURN count(new_rel)
                    """
                    session.run(query, {"original_name": original_name, "resolved_name": resolved_name})
                    
                    # 나가는 관계(outgoing) 이전
                    query = f"""
                        MATCH (original:{node_type} {{id: $original_name}})-[r]->(target)
                        WHERE NOT target:Entity OR labels(target)[0] <> '{node_type}' OR target.id <> $resolved_name
                        MATCH (source:{node_type} {{id: $resolved_name}})
                        WITH source, target, type(r) AS rel_type, properties(r) AS props
                        CALL apoc.create.relationship(source, rel_type, props, target) YIELD rel AS new_rel
                        RETURN count(new_rel)
                    """
                    session.run(query, {"original_name": original_name, "resolved_name": resolved_name})
                    
                    # 원본 노드 삭제
                    query = f"""
                        MATCH (original:{node_type} {{id: $original_name}})
                        DETACH DELETE original
                    """
                    session.run(query, {"original_name": original_name})
                    
                    updates_count += 1
                else:
                    # 3. 표준화된 이름의 노드가 없는 경우
                    # 기존 노드의 이름만 변경
                    query = f"""
                        MATCH (node:{node_type} {{id: $original_name}})
                        SET node.id = $resolved_name
                        RETURN node
                    """
                    session.run(query, {"original_name": original_name, "resolved_name": resolved_name})
                    updates_count += 1
                    
        return updates_count

                    
