# Legal Knowledge Graph 프로젝트 구조 및 설명

```
legal_knowledge_graph/
├── pyproject.toml                 # Python 프로젝트 및 의존성 관리 파일
├── uv.lock                        # 의존성 잠금 파일
├── docker-compose.yml             # Neo4j 등 서비스 도커 컴포즈 설정
├── neo4j.conf                     # Neo4j 데이터베이스 보안 및 프로시저 설정
├── apoc.conf                      # Neo4j APOC 플러그인 관련 설정
├── README.md                      # 프로젝트 설명 및 구조(본 문서)
├── .python-version                # Python 버전 지정 파일 (3.12)
├── .gitignore                     # Git 무시 파일 설정
├── logs/                          # 데이터 처리 및 실행 로그 저장 폴더
│   └── *.log                      # 데이터 로딩 등 주요 작업 로그 파일
├── data/                          # 데이터 및 벤치마크 폴더
│   ├── corpus/                    # 원본 텍스트 및 PDF 데이터 저장 폴더
│   │   └── maud/                  # 실제 계약서 등 대용량 텍스트/문서 파일
│   └── benchmarks/                # 벤치마크용 정답 데이터
│       └── maud.json              # QA 벤치마크용 정답 JSON
├── neo4j/                         # Neo4j 데이터베이스 저장소
├── src/                           # 소스 코드 폴더
│   ├── main.py                    # 프로젝트 진입점. 전체 파이프라인 실행
│   ├── logger.py                  # 로그 설정 및 파일/콘솔 출력 함수
│   └── generate_knowledge_graph/  # 지식 그래프 생성 관련 모듈
│       ├── state.py               # 파이프라인 상태 및 설정 데이터 클래스
│       ├── builder.py             # 전체 워크플로우(그래프) 빌더 및 LLM/Neo4j 초기화
│       ├── prompt.py              # 엔티티/관계 추출 프롬프트 등 LLM용 프롬프트 정의
│       ├── utils/                 # 유틸리티 모듈
│       │   ├── __init__.py        # 유틸리티 모듈 임포트 관리
│       │   ├── model.py           # Document, Chunk 등 데이터 모델 정의
│       │   ├── parser.py          # LLM 출력 파싱용 JSON 파서
│       │   ├── database.py        # Neo4j 데이터베이스 연결 및 벡터 인덱스 관리
│       │   ├── callback.py        # LLM 진행상황 표시용 BatchCallback 클래스
│       │   └── cluster.py         # 엔티티 클러스터링 기능 (sklearn 기반)
│       └── nodes/                 # 파이프라인 각 단계별 노드 구현
│           ├── __init__.py        # 노드 모듈 임포트 관리
│           ├── data_loader.py     # 벤치마크 및 원본 데이터 로딩, Document 생성
│           ├── chunker.py         # 문서 분할(Chunk) 처리
│           ├── entity_relation_extractor.py # 엔티티/관계 추출 및 검증
│           ├── graph_db_writer.py # 추출된 정보 Neo4j 그래프DB에 저장
│           └── entity_resolver.py # 엔티티 정규화 및 중복 해결 기능
```

## 주요 폴더 및 파일 설명

- **pyproject.toml / uv.lock**: Python 패키지 및 의존성 관리 파일입니다. 주요 의존성: langchain-openai, langgraph, neo4j, python-dotenv, scikit-learn
- **docker-compose.yml**: Neo4j 데이터베이스 및 플러그인(GDS, APOC) 환경을 도커로 손쉽게 실행할 수 있도록 설정합니다.
- **neo4j.conf / apoc.conf**: Neo4j 및 APOC 관련 보안, 프로시저, 파일 입출력 권한을 설정합니다.
- **.python-version**: Python 3.12 버전을 명시합니다.
- **logs/**: 데이터 로딩, 파이프라인 실행 등 주요 작업의 로그가 저장됩니다.
- **data/corpus/**: 실제 계약서 등 대용량 원본 텍스트/문서 파일이 저장됩니다.
- **data/benchmarks/**: 벤치마크용 정답 데이터(JSON)가 저장됩니다.
- **neo4j/**: Neo4j 데이터베이스의 실제 데이터가 저장되는 폴더입니다.
- **src/main.py**: 프로젝트의 메인 실행 파일로, State 객체를 생성해 graph.invoke로 전체 파이프라인을 실행합니다.
- **src/logger.py**: 로그 파일 및 콘솔 출력을 위한 로거 설정 함수가 정의되어 있습니다.
- **src/generate_knowledge_graph/**: 지식 그래프 생성의 핵심 로직이 구현된 폴더입니다.
  - **state.py**: 파이프라인의 상태(State)와 설정(Config) 데이터 클래스를 정의합니다.
  - **builder.py**: LLM, 임베딩, Neo4j 연결 등 전체 파이프라인 워크플로우를 빌드합니다.
  - **prompt.py**: 엔티티/관계 추출을 위한 LLM 프롬프트가 정의되어 있습니다.
  - **utils/**: 데이터 모델, 파서, DB연결, 콜백 등 유틸리티 모듈 모음입니다.
    - **model.py**: Document, Chunk 등 데이터 구조를 정의합니다.
    - **parser.py**: LLM 출력 결과를 JSON으로 파싱하는 파서입니다.
    - **database.py**: Neo4j 연결, 벡터 인덱스/제약조건 관리, 노드/관계 생성 등 DB 관련 기능을 제공합니다.
    - **callback.py**: LLM 실행 시 진행상황을 tqdm으로 표시하는 BatchCallback 클래스입니다.
    - **cluster.py**: sklearn의 AgglomerativeClustering을 사용하여 유사한 엔티티들을 클러스터링하는 기능을 제공합니다.
  - **nodes/**: 파이프라인 각 단계별 노드가 구현되어 있습니다.
    - **data_loader.py**: 벤치마크 및 원본 데이터를 로딩하여 Document 객체로 변환합니다.
    - **chunker.py**: 문서를 일정 크기로 분할(Chunk)합니다.
    - **entity_relation_extractor.py**: LLM을 이용해 엔티티 및 관계를 추출하고 검증합니다.
    - **graph_db_writer.py**: 추출된 엔티티/관계를 Neo4j 그래프DB에 저장합니다.
    - **entity_resolver.py**: 클러스터링된 유사 엔티티들을 분석하여 중복을 해결하고 정규화하는 기능을 제공합니다.

## 실행 및 개발 참고사항
- Python 3.12 이상이 필요합니다.
- Neo4j, GDS, APOC 플러그인이 필요하며, docker-compose로 손쉽게 실행할 수 있습니다.
- 전체 파이프라인 실행은 `src/main.py`를 실행하면 됩니다.
- 각 모듈/노드별로 확장 및 커스터마이징이 용이하도록 설계되어 있습니다.
- 엔티티 해결 기능을 통해 유사한 엔티티들을 자동으로 클러스터링하고 정규화할 수 있습니다.
