from tqdm.auto import tqdm
import json
from uuid import uuid4
from neo4j import GraphDatabase


NODE_TYPES = ["Corpus", "Article", "Section", "Chunk"]
RELATIONSHIP_TYPES = ["CHILD", "NEXT", "PREV"]


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
            for node_type in NODE_TYPES:
                session.run(f"DROP CONSTRAINT {node_type.lower()}_id_unique IF EXISTS")
            
            for node_type in NODE_TYPES:
                if node_type == "Corpus":
                    continue
                session.run(f"DROP INDEX {node_type.lower()}_vector_index IF EXISTS")
    
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
            node_types = NODE_TYPES
            
            for node_type in NODE_TYPES:
                if node_type == "Corpus":
                    continue
                session.run(f"""
                CREATE VECTOR INDEX {node_type.lower()}_vector_index IF NOT EXISTS
                FOR (n:{node_type})
                ON (n.vector)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 3072,
                        `vector.similarity_function`: "cosine"
                    }}
                }}
                """)
                print(f"✅ {node_type} 노드 벡터 인덱스 생성 완료")
            
        # print("✅ 문서, 엔티티 및 관계 벡터 인덱스 설정 완료")
    
    def setup_constraints(self):
        """데이터베이스 제약 조건 설정"""
        with self.driver.session() as session:
            # 문서 노드 제약조건 생성
            node_types = NODE_TYPES
            for node_type in node_types:
                session.run(f"""
                    CREATE CONSTRAINT {node_type.lower()}_id_unique IF NOT EXISTS
                    FOR (n:{node_type})
                    REQUIRE n.id IS UNIQUE
                """)
                print(f"✅ {node_type} ID 제약조건 생성 완료")

    def create_nodes_and_relationships(self, documents):
        with self.driver.session() as session:
            # 임베딩 대상 수집용 버퍼
            node_ids_for_embedding = []
            texts_for_embedding = []

            def ensure_corpus(file_path: str, table_of_contents: dict | None = None):
                corpus_id = str(uuid4())
                corpus_name = file_path.split("/")[-1] if file_path else ""
                toc_str = json.dumps(table_of_contents or {}, ensure_ascii=False)
                session.run(
                    """
                    MERGE (co:Corpus {id: $corpus_id})
                    SET co.name = $corpus_name,
                        co.file_path = $file_path,
                        co.table_of_contents = $table_of_contents
                    """,
                    {
                        "corpus_id": corpus_id,
                        "corpus_name": corpus_name,
                        "file_path": file_path,
                        "table_of_contents": toc_str,
                    },
                )
                return corpus_id

            def create_chunks_recursive(parent_label: str, parent_id: str, chunk_obj, order_idx: int = 0, file_path: str = ""):
                chunk_id = str(uuid4())
                # Chunk 객체 처리 (dict는 들어오지 않음)
                span_value = list(getattr(chunk_obj, "span", (0, 0)))
                content_value = getattr(chunk_obj, "content", "")
                summary_value = getattr(chunk_obj, "summary", "")
                name_value = getattr(chunk_obj, "name", "")
                session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id})
                    ON CREATE SET c.span = $span,
                                  c.content = $content,
                                  c.summary = $summary,
                                  c.order = $order,
                                  c.name = $name,
                                  c.file_path = $file_path
                    """,
                    {
                        "chunk_id": chunk_id,
                        "span": span_value,
                        "content": content_value,
                        "summary": summary_value,
                        "order": order_idx,
                        "name": name_value,
                        "file_path": file_path,
                    },
                )
                session.run(
                    f"""
                    MATCH (p:{parent_label} {{id: $parent_id}}), (c:Chunk {{id: $chunk_id}})
                    MERGE (p)-[:CHILD]->(c)
                    """,
                    {"parent_id": parent_id, "chunk_id": chunk_id},
                )
                # 자식 재귀 및 NEXT/PREV 연결
                prev_child_id = None
                for child_idx, child in enumerate(getattr(chunk_obj, "children", []) or []):
                    child_chunk_id = create_chunks_recursive("Chunk", chunk_id, child, child_idx, file_path)
                    if prev_child_id is not None:
                        session.run(
                            """
                            MATCH (p:Chunk {id: $prev}), (c:Chunk {id: $cur})
                            MERGE (p)-[:NEXT]->(c)
                            MERGE (c)-[:PREV]->(p)
                            """,
                            {"prev": prev_child_id, "cur": child_chunk_id},
                        )
                    prev_child_id = child_chunk_id

                # 임베딩 대상 텍스트 선택: summary가 비었으면 content 사용
                text_for_vec = summary_value.strip() if summary_value and summary_value.strip() else content_value
                if text_for_vec.strip():
                    node_ids_for_embedding.append(("Chunk", chunk_id))
                    texts_for_embedding.append(text_for_vec)
                return chunk_id

            for doc in documents:
                file_path = getattr(doc, "file_path", "")
                toc = getattr(doc, "table_of_contents", {}) or {}
                corpus_id = ensure_corpus(file_path, toc)
                # 문서 루트는 Article로 간주하지 않고, Corpus -> Chunk 트리로 적재
                prev_top_id = None
                # dict 트리를 순회
                top_children = getattr(doc, "children", {}) or {}
                prev_top_id = None
                for idx, top_chunk in enumerate(list(top_children.values())):
                    top_id = create_chunks_recursive("Corpus", corpus_id, top_chunk, idx, file_path)
                    if prev_top_id is not None:
                        session.run(
                            """
                            MATCH (p:Chunk {id: $prev}), (c:Chunk {id: $cur})
                            MERGE (p)-[:NEXT]->(c)
                            MERGE (c)-[:PREV]->(p)
                            """,
                            {"prev": prev_top_id, "cur": top_id},
                        )
                    prev_top_id = top_id

                # 문서 요약도 저장 및 벡터화 대상
                doc_summary = getattr(doc, "summary", "")
                doc_content = getattr(doc, "content", "")
                text_for_vec = doc_summary.strip() if doc_summary and doc_summary.strip() else doc_content
                session.run(
                    """
                    MERGE (doc:Corpus {id: $corpus_id})
                    SET doc.summary = $summary,
                        doc.content = $content
                    """,
                    {"corpus_id": corpus_id, "summary": doc_summary, "content": doc_content},
                )
                if text_for_vec.strip():
                    node_ids_for_embedding.append(("Corpus", corpus_id))
                    texts_for_embedding.append(text_for_vec)

            # 임베딩 생성 후 각 노드에 저장
            if texts_for_embedding:
                vectors = self.batch_embed(texts_for_embedding)
                for (label, node_id), vec in zip(node_ids_for_embedding, vectors):
                    session.run(
                        f"""
                        MATCH (n:{label} {{id: $id}})
                        SET n.vector = $vector
                        """,
                        {"id": node_id, "vector": vec},
                    )
