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

    def create_nodes_and_relationships(self, chunks, hierarchical_chunk_ids):       
        with self.driver.session() as session:         
            # 1) 사전 준비: 사용될 모든 chunk 인덱스 수집 후 UUID 매핑 생성
            all_chunk_indices = set()
            for file_path, articles in hierarchical_chunk_ids.items():
                for article_name, sections in articles.items():
                    if not isinstance(sections, dict):
                        continue
                    for section_name, section_value in sections.items():
                        # Skip article-level summary key
                        if section_name == "summary":
                            continue
                        if isinstance(section_value, dict):
                            chunk_idx_list = section_value.get("chunk_ids", [])
                        else:
                            chunk_idx_list = section_value
                        if not isinstance(chunk_idx_list, (list, tuple)):
                            continue
                        for idx in chunk_idx_list:
                            if isinstance(idx, int):
                                all_chunk_indices.add(idx)

            chunk_idx_to_uuid = {idx: str(uuid4()) for idx in all_chunk_indices}

            # 임베딩 대상 수집용 버퍼
            article_ids_for_embedding = []
            article_summaries_for_embedding = []
            section_ids_for_embedding = []
            section_summaries_for_embedding = []
            chunk_ids_for_embedding = []
            chunk_summaries_for_embedding = []

            # 2) 그래프 생성
            for file_path, articles in hierarchical_chunk_ids.items():
                corpus_name = (file_path.split("/")[-1] if file_path else "")
                corpus_id = str(uuid4())
                corpus_contents: list[str] = []

                # Corpus 노드 생성 및 속성 설정
                session.run(
                    """
                    MERGE (co:Corpus {id: $corpus_id})
                    SET co.name = $corpus_name
                    SET co.file_path = $file_path
                    """,
                    {
                        "corpus_id": corpus_id,
                        "corpus_name": corpus_name,
                        "file_path": file_path,
                    },
                )

                for article_name, sections in articles.items():
                    article_id = str(uuid4())
                    article_summary = None
                    if isinstance(sections, dict):
                        article_summary = sections.get("summary")
                    session.run(
                        """
                        MERGE (a:Article {id: $article_id})
                        SET a.name = $article_name
                        SET a.summary = $article_summary
                        """,
                        {
                            "article_id": article_id,
                            "article_name": article_name,
                            "article_summary": article_summary,
                        },
                    )

                    # 임베딩 대상 수집 (Article)
                    if isinstance(article_summary, str) and article_summary.strip() != "":
                        article_ids_for_embedding.append(article_id)
                        article_summaries_for_embedding.append(article_summary)

                    # Corpus -> Article CHILD
                    session.run(
                        """
                        MATCH (co:Corpus {id: $corpus_id}), (a:Article {id: $article_id})
                        MERGE (co)-[:CHILD]->(a)
                        """,
                        {
                            "article_id": article_id,
                            "corpus_id": corpus_id,
                        },
                    )

                    # 누적할 아티클 콘텐츠 버퍼
                    article_contents: list[str] = []
                    for section_name, section_value in sections.items():
                        # Skip article-level summary key
                        if section_name == "summary":
                            continue
                        section_id = str(uuid4())
                        # Support both dict {chunk_ids, summary} and legacy list format
                        if isinstance(section_value, dict):
                            chunk_idx_list = section_value.get("chunk_ids", [])
                            section_summary = section_value.get("summary")
                        else:
                            chunk_idx_list = section_value
                            section_summary = None
                        # 섹션 콘텐츠 생성: 청크들의 content를 "\n\n"으로 연결
                        section_chunk_texts: list[str] = []
                        if isinstance(chunk_idx_list, (list, tuple)):
                            for idx in chunk_idx_list:
                                if isinstance(idx, int) and 0 <= idx < len(chunks):
                                    section_chunk_texts.append(chunks[idx].content)
                        section_content_str = "\n\n".join(section_chunk_texts)
                        session.run(
                            """
                            MERGE (s:Section {id: $section_id})
                            SET s.name = $section_name
                            SET s.summary = $section_summary
                            SET s.content = $section_content
                            """,
                            {
                                "section_id": section_id,
                                "section_name": section_name,
                                "section_summary": section_summary,
                                "section_content": section_content_str,
                            },
                        )

                        # 임베딩 대상 수집 (Section)
                        if isinstance(section_summary, str) and section_summary.strip() != "":
                            section_ids_for_embedding.append(section_id)
                            section_summaries_for_embedding.append(section_summary)

                        # Article -> Section CHILD
                        session.run(
                            """
                            MATCH (a:Article {id: $article_id}), (s:Section {id: $section_id})
                            MERGE (a)-[:CHILD]->(s)
                            """,
                            {"section_id": section_id, "article_id": article_id},
                        )

                        # 각 섹션의 청크들 생성 및 연결
                        for order_idx, chunk_idx in enumerate(chunk_idx_list if isinstance(chunk_idx_list, (list, tuple)) else []):
                            # 유효성 체크
                            if not (0 <= chunk_idx < len(chunks)):
                                continue
                            chunk_obj = chunks[chunk_idx]

                            chunk_id = chunk_idx_to_uuid[chunk_idx]
                            span_value = list(chunk_obj.span) if isinstance(chunk_obj.span, (list, tuple)) else []
                            content_value = chunk_obj.content
                            summary_value = getattr(chunk_obj, "summary", "")

                            # Chunk 노드 생성 (최초 생성시에만 속성 설정)
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
                                    "file_path": file_path,
                                },
                            )

                            # 임베딩 대상 수집 (Chunk)
                            if isinstance(summary_value, str) and summary_value.strip() != "":
                                chunk_ids_for_embedding.append(chunk_id)
                                chunk_summaries_for_embedding.append(summary_value)

                            # Section -> Chunk CHILD
                            session.run(
                                """
                                MATCH (s:Section {id: $section_id}), (c:Chunk {id: $chunk_id})
                                MERGE (s)-[:CHILD]->(c)
                                """,
                                {"chunk_id": chunk_id, "section_id": section_id},
                            )

                            # 이전 청크와의 NEXT/PREV 관계 설정
                            if order_idx > 0:
                                prev_chunk_idx = chunk_idx_list[order_idx - 1]
                                if 0 <= prev_chunk_idx < len(chunks):
                                    prev_chunk_id = chunk_idx_to_uuid[prev_chunk_idx]
                                    session.run(
                                        """
                                        MATCH (p:Chunk {id: $prev_id}), (c:Chunk {id: $cur_id})
                                        MERGE (p)-[:NEXT]->(c)
                                        MERGE (c)-[:PREV]->(p)
                                        """,
                                        {"prev_id": prev_chunk_id, "cur_id": chunk_id},
                                    )

                        # 아티클 콘텐츠에 섹션 콘텐츠 추가
                        article_contents.append(section_content_str)

                    # 아티클 콘텐츠 저장 (섹션 콘텐츠들을 "\n\n"으로 연결)
                    article_content_str = "\n\n".join([t for t in article_contents if t])
                    session.run(
                        """
                        MATCH (a:Article {id: $article_id})
                        SET a.content = $article_content
                        """,
                        {"article_id": article_id, "article_content": article_content_str},
                    )

                    # 코퍼스 콘텐츠 버퍼에 아티클 콘텐츠 추가
                    corpus_contents.append(article_content_str)

                # 코퍼스 콘텐츠 저장 (아티클 콘텐츠들을 "\n\n"으로 연결)
                corpus_content_str = "\n\n".join([t for t in corpus_contents if t])
                session.run(
                    """
                    MATCH (co:Corpus {id: $corpus_id})
                    SET co.content = $corpus_content
                    """,
                    {"corpus_id": corpus_id, "corpus_content": corpus_content_str},
                )

            # 3) 요약 임베딩 벡터 생성 및 노드에 저장 (Corpus 제외)
            # Article vectors
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

            # Section vectors
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

            # Chunk vectors
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
