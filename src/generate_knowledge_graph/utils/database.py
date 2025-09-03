from tqdm.auto import tqdm
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
                        `vector.dimensions`: 1024,
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

    def create_nodes_and_relationships(self, structured_chunks):
        with self.driver.session() as session:
            # 임베딩 대상 수집용 버퍼
            article_ids_for_embedding = []
            article_summaries_for_embedding = []
            section_ids_for_embedding = []
            section_summaries_for_embedding = []
            chunk_ids_for_embedding = []
            chunk_summaries_for_embedding = []

            def collect_all_chunk_contents(subtree):
                contents = []
                if isinstance(subtree, dict):
                    if isinstance(subtree.get("chunks"), list):
                        contents += [getattr(ch, "content", "") for ch in subtree["chunks"]]
                    for k, v in subtree.items():
                        if k in ("summary", "chunks"):
                            continue
                        if isinstance(v, dict):
                            contents += collect_all_chunk_contents(v)
                return contents

            for file_path, article_tree in structured_chunks.items():
                corpus_name = (file_path.split("/")[-1] if file_path else "")
                corpus_id = str(uuid4())

                # Corpus 노드 생성
                session.run(
                    """
                    MERGE (co:Corpus {id: $corpus_id})
                    SET co.name = $corpus_name,
                        co.file_path = $file_path
                    """,
                    {"corpus_id": corpus_id, "corpus_name": corpus_name, "file_path": file_path},
                )

                corpus_contents = []

                def create_article(name: str, subtree: dict):
                    article_id = str(uuid4())
                    article_summary = subtree.get("summary", "") if isinstance(subtree, dict) else ""
                    session.run(
                        """
                        MERGE (a:Article {id: $article_id})
                        SET a.name = $name,
                            a.summary = $summary
                        """,
                        {"article_id": article_id, "name": name, "summary": article_summary},
                    )
                    session.run(
                        """
                        MATCH (co:Corpus {id: $corpus_id}), (a:Article {id: $article_id})
                        MERGE (co)-[:CHILD]->(a)
                        """,
                        {"corpus_id": corpus_id, "article_id": article_id},
                    )
                    if article_summary.strip():
                        article_ids_for_embedding.append(article_id)
                        article_summaries_for_embedding.append(article_summary)

                    # Article content는 하위 섹션 전체 chunk content를 연결
                    article_contents = []

                    def create_section(parent_label: str, parent_id: str, name: str, subtree: dict):
                        section_id = str(uuid4())
                        section_summary = subtree.get("summary", "") if isinstance(subtree, dict) else ""
                        session.run(
                            """
                            MERGE (s:Section {id: $section_id})
                            SET s.name = $name,
                                s.summary = $summary
                            """,
                            {"section_id": section_id, "name": name, "summary": section_summary},
                        )
                        session.run(
                            f"""
                            MATCH (p:{parent_label} {{id: $parent_id}}), (s:Section {{id: $section_id}})
                            MERGE (p)-[:CHILD]->(s)
                            """,
                            {"parent_id": parent_id, "section_id": section_id},
                        )
                        if section_summary.strip():
                            section_ids_for_embedding.append(section_id)
                            section_summaries_for_embedding.append(section_summary)

                        # Leaf chunks 생성
                        if isinstance(subtree, dict) and isinstance(subtree.get("chunks"), list):
                            prev_chunk_id = None
                            for order_idx, ch in enumerate(subtree["chunks"]):
                                chunk_id = str(uuid4())
                                span_value = list(getattr(ch, "span", (0, 0)))
                                content_value = getattr(ch, "content", "")
                                summary_value = getattr(ch, "summary", "")
                                file_path_value = getattr(ch, "file_path", file_path)
                                session.run(
                                    """
                                    MERGE (c:Chunk {id: $chunk_id})
                                    ON CREATE SET c.span = $span,
                                                  c.content = $content,
                                                  c.summary = $summary,
                                                  c.order = $order,
                                                  c.file_path = $file_path
                                    """,
                                    {
                                        "chunk_id": chunk_id,
                                        "span": span_value,
                                        "content": content_value,
                                        "summary": summary_value,
                                        "order": order_idx,
                                        "file_path": file_path_value,
                                    },
                                )
                                session.run(
                                    """
                                    MATCH (s:Section {id: $section_id}), (c:Chunk {id: $chunk_id})
                                    MERGE (s)-[:CHILD]->(c)
                                    """,
                                    {"section_id": section_id, "chunk_id": chunk_id},
                                )
                                if summary_value.strip():
                                    chunk_ids_for_embedding.append(chunk_id)
                                    chunk_summaries_for_embedding.append(summary_value)
                                if prev_chunk_id is not None:
                                    session.run(
                                        """
                                        MATCH (p:Chunk {id: $prev}), (c:Chunk {id: $cur})
                                        MERGE (p)-[:NEXT]->(c)
                                        MERGE (c)-[:PREV]->(p)
                                        """,
                                        {"prev": prev_chunk_id, "cur": chunk_id},
                                    )
                                prev_chunk_id = chunk_id

                        # Section content: 하위 전체 chunk content 연결
                        section_contents = collect_all_chunk_contents(subtree)
                        section_content_str = "\n\n".join([t for t in section_contents if t])
                        session.run(
                            """
                            MATCH (s:Section {id: $section_id})
                            SET s.content = $content
                            """,
                            {"section_id": section_id, "content": section_content_str},
                        )

                        # 하위 섹션 처리
                        if isinstance(subtree, dict):
                            for child_name, child_val in subtree.items():
                                if child_name in ("summary", "chunks"):
                                    continue
                                if isinstance(child_val, dict):
                                    create_section("Section", section_id, child_name, child_val)

                        # Article content 누적
                        article_contents.append(section_content_str)

                    # 최상위 섹션들 생성
                    for section_name, section_subtree in subtree.items():
                        if section_name == "summary":
                            continue
                        if isinstance(section_subtree, dict):
                            create_section("Article", article_id, section_name, section_subtree)

                    # Article content 저장
                    article_content_str = "\n\n".join([t for t in article_contents if t])
                    session.run(
                        """
                        MATCH (a:Article {id: $article_id})
                        SET a.content = $content
                        """,
                        {"article_id": article_id, "content": article_content_str},
                    )

                    # Corpus content 누적
                    corpus_contents.append(article_content_str)

                # 각 Article 처리
                for article_name, article_subtree in article_tree.items():
                    if article_name == "summary":
                        continue
                    if isinstance(article_subtree, dict):
                        article_id = create_article(article_name, article_subtree)

                # Corpus content 저장
                corpus_content_str = "\n\n".join([t for t in corpus_contents if t])
                session.run(
                    """
                    MATCH (co:Corpus {id: $corpus_id})
                    SET co.content = $content
                    """,
                    {"corpus_id": corpus_id, "content": corpus_content_str},
                )

            # 임베딩 벡터 저장 (요약이 있는 것만)
            if article_summaries_for_embedding:
                article_vectors = self.batch_embed(article_summaries_for_embedding)
                for node_id, vector in zip(article_ids_for_embedding, article_vectors):
                    session.run(
                        """
                        MATCH (a:Article {id: $id})
                        SET a.vector = $vector
                        """,
                        {"id": node_id, "vector": vector},
                    )

            if section_summaries_for_embedding:
                section_vectors = self.batch_embed(section_summaries_for_embedding)
                for node_id, vector in zip(section_ids_for_embedding, section_vectors):
                    session.run(
                        """
                        MATCH (s:Section {id: $id})
                        SET s.vector = $vector
                        """,
                        {"id": node_id, "vector": vector},
                    )

            if chunk_summaries_for_embedding:
                chunk_vectors = self.batch_embed(chunk_summaries_for_embedding)
                for node_id, vector in zip(chunk_ids_for_embedding, chunk_vectors):
                    session.run(
                        """
                        MATCH (c:Chunk {id: $id})
                        SET c.vector = $vector
                        """,
                        {"id": node_id, "vector": vector},
                    )
