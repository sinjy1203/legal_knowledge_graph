# Legal Knowledge Graph RAG 시스템

</br>

## 프로젝트 개요

본 프로젝트는 **법률 문서 검색을 위한 지식 그래프 기반 RAG(Retrieval-Augmented Generation) 시스템**입니다. 

### 주요 목적
- **입력**: 법률 관련 질문
- **출력**: 질문에 답변하기 위해 필요한 문서의 `file_path`와 해당 텍스트 청크의 `span` 정보
- **목표**: 복잡한 법률 질의에 대해 정확하고 관련성 높은 문서 부분을 효율적으로 검색

### 시스템 구조
1. **지식 그래프 구축**: 법률 문서를 불러오고(Intro/Body 분리 포함) Neo4j에 Corpus→Article→Section→Chunk 계층으로 저장
2. **검색 Agent**: LangGraph 기반 ReAct 패턴으로 Corpus→Article→Section→Chunk 순서로 탐색하고 최종적으로 `file_path`와 `span` 반환
3. **벤치마크 평가**: LegalBenchRAG 라이브러리를 사용한 정량적 성능 평가

</br>

## 프로젝트 구조

```
legal_knowledge_graph/
├── pyproject.toml                 # Python 프로젝트 및 의존성 관리 파일
├── uv.lock                        # 의존성 잠금 파일
├── docker-compose.yml             # Neo4j, Langfuse 등 서비스 도커 컴포즈 설정
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
├── benchmark_results/             # 벤치마크 실행 결과 저장 폴더
├── neo4j/                         # Neo4j 데이터베이스 저장소
├── src/                           # 소스 코드 폴더
│   ├── generate.py                # 지식 그래프 생성 파이프라인 실행
│   ├── search.py                  # 지식 그래프 검색 agent 실행
│   ├── run_benchmark.py           # LegalBenchRAG 벤치마크 실행 및 평가
│   ├── logger.py                  # 로그 설정 및 파일/콘솔 출력 함수
│   ├── legalbenchrag/             # 벤치마크 평가용 외부 라이브러리 (git submodule)
│   ├── generate_knowledge_graph/  # 지식 그래프 생성 관련 모듈
│   │   ├── state.py               # 파이프라인 상태 및 설정 데이터 클래스
│   │   ├── builder.py             # 전체 워크플로우(그래프) 빌더 및 LLM/Neo4j 초기화
│   │   ├── prompt.py              # 엔티티/관계 추출 프롬프트 등 LLM용 프롬프트 정의
│   │   ├── utils/                 # 유틸리티 모듈
│   │   │   ├── __init__.py        # 유틸리티 모듈 임포트 관리
│   │   │   ├── model.py           # Document, Chunk 등 데이터 모델 정의
│   │   │   ├── parser.py          # LLM 출력 파싱용 JSON 파서
│   │   │   ├── database.py        # Neo4j 데이터베이스 연결 및 벡터 인덱스 관리
│   │   │   ├── callback.py        # LLM 진행상황 표시용 BatchCallback 클래스
│   │   │   └── cluster.py         # 엔티티 클러스터링 기능 (sklearn 기반)
│   │   └── nodes/                 # 파이프라인 각 단계별 노드 구현
│   │       ├── __init__.py        # 노드 모듈 임포트 관리
│   │       ├── data_loader.py     # 벤치마크 및 원본 데이터 로딩, Document 생성
│   │       ├── intro_body_separator.py  # "follows:" 기준으로 intro/body 분리 및 body_span 저장
│   │       ├── table_of_contents_extractor.py # 목차(Article/Section) 텍스트 추출
│   │       ├── chunker.py         # body 기준 청크 생성, span은 body_span 기반 절대 인덱스
│   │       ├── document_structure_detector.py # 섹션/아티클 인식, hierarchical_chunk_ids 누적
│   │       ├── summarizer.py      # 섹션/아티클 요약 배치 생성 및 저장
│   │       ├── graph_db_writer.py # Neo4j 그래프 DB 저장 (벡터 포함)
│   │       └── entity_resolver.py # 엔티티 정규화 및 중복 해결 기능 (선택)
│   └── search_knowledge_graph/    # 지식 그래프 검색 관련 모듈
│       ├── state.py               # 검색 agent 상태 및 설정 데이터 클래스
│       ├── agent.py               # ReAct agent 구현 (LangGraph 기반)
│       ├── prompt.py              # 지식 그래프 검색용 프롬프트 정의
│       └── tools/                 # 검색용 도구 모음
│           ├── __init__.py        # 도구 모듈 임포트 관리
│           ├── search_corpus.py   # 코퍼스 리스트 검색
│           ├── search_article.py  # 특정 Corpus 내 Article 검색
│           ├── search_section.py  # 특정 Article 내 Section 검색
│           ├── search_chunk.py    # 특정 Section 내 Chunk 검색
│           └── response.py        # 최종 반환용 file_path/span 조회 도구 (Return Direct)
```

</br>

## 주요 컴포넌트 설명

### 1. 지식 그래프 생성 (Generate Knowledge Graph)
- **목적**: 법률 문서를 계층 구조로 정리하고 벡터 기반 검색이 가능한 그래프를 구성
- **핵심 파일(주요 단계 순서)**:
  - `data_loader.py` → `intro_body_separator.py` → `table_of_contents_extractor.py` → `chunker.py` → `document_structure_detector.py` → `summarizer.py` → `graph_db_writer.py`
  - Intro/Body 분리: "follows:"를 기준으로 `Document.intro`/`Document.body`를 저장, `Document.body_span`에 본문 절대 인덱스 저장
  - Chunking: `document.body`를 기준으로 분할하며 각 청크 `span`은 `body_span` 기반 절대 좌표로 저장
  - 요약/임베딩: 섹션/아티클 요약을 배치로 생성(`chain.batch` + `BatchCallback`) 후 Neo4j 노드의 `summary`와 `vector`로 저장

### 2. 검색 Agent (Search Knowledge Graph)
- **목적**: 사용자 질문에 대해 그래프를 탐색하여 최종적으로 `file_path`/`span` 반환
- **핵심 파일**:
  - `src/search.py`: 검색 agent 실행 엔트리포인트
  - `src/search_knowledge_graph/agent.py`: ReAct 패턴 기반 검색 로직
  - `src/search_knowledge_graph/tools/`: 도구 모음
    - `search_corpus.py`: 코퍼스 후보 조회
    - `search_article.py`: 코퍼스 내 아티클 의미 유사도 검색
    - `search_section.py`: 아티클 내 섹션 의미 유사도 검색
    - `search_chunk.py`: 섹션 내 청크 의미 유사도 검색
    - `response.py`: 선택된 청크들의 `file_path`와 `span`을 최종 반환 (Return Direct)

### 3. 벤치마크 평가 (Benchmark Evaluation)
- **목적**: LegalBenchRAG를 사용한 정량적 성능 평가
- **핵심 파일**:
  - `src/run_benchmark.py`: 벤치마크 실행 및 precision/recall 계산
  - `src/legalbenchrag/`: 외부 벤치마크 라이브러리 (git submodule)
  - `benchmark_results/`: 평가 결과 저장 디렉터리

### 4. 핵심 인프라
- **pyproject.toml / uv.lock**: Python 의존성 관리 (langchain-openai, langgraph, neo4j 등)
- **docker-compose.yml**: Neo4j(GDS, APOC), Langfuse, Redis, MinIO 등 서비스 컨테이너
- **neo4j.conf / apoc.conf**: Neo4j 및 APOC 플러그인 보안 설정
- **logs/**: 파이프라인 실행 로그
- **data/**: 원본 문서(corpus)와 벤치마크 정답 데이터

</br>

## 실행 방법

### 1. 환경 설정
```bash
# 리포지토리 클론 (서브모듈 포함)
git clone --recursive <repository-url>
cd legal_knowledge_graph

# 또는 이미 클론했다면 서브모듈만 별도로 가져오기
git submodule update --init --recursive

# Python 3.12 가상환경 생성
uv venv --python 3.12 .venv

# 가상환경 활성화
source .venv/bin/activate

# uv를 사용한 의존성 설치 (LegalBenchRAG 포함)
uv sync
```

### 2. 인프라 실행
```bash
# Neo4j, Langfuse 등 서비스 실행
docker compose up -d
```

### 3. 시스템 실행 워크플로우
```bash
# 1. 지식 그래프 생성
python src/generate.py

# 2. 검색 Agent 테스트
python src/search.py

# 3. 벤치마크 평가 실행
python src/run_benchmark.py
```

### 4. 성능 측정
- **Input**: 사용자의 법률 관련 질문
- **Output**: 답변에 필요한 문서의 `file_path`와 텍스트 청크의 `span`
- **평가 지표**: LegalBenchRAG를 통한 Precision과 Recall 계산

</br>

## 그래프 데이터베이스 스키마

본 시스템은 다음과 같은 최소 스키마를 사용합니다:

```json
{
    "Node": [
        {
      "type": "Corpus",
      "properties": { "id": "uuid", "name": "file basename", "file_path": "original file path" }
    },
    {
      "type": "Article",
      "properties": { "id": "uuid", "name": "article title", "summary": "text", "vector": "embedding" }
    },
    {
      "type": "Section",
      "properties": { "id": "uuid", "name": "section title", "summary": "text", "vector": "embedding" }
    },
    {
      "type": "Chunk",
      "properties": { "id": "uuid", "content": "text", "summary": "text", "order": "int", "file_path": "string", "span": "[start,end]", "vector": "embedding" }
        }
    ],
    "Edge": [
    { "type": "CHILD", "direction": "Corpus→Article→Section→Chunk" },
    { "type": "NEXT",  "direction": "Chunk→Chunk (within same Section)" },
    { "type": "PREV",  "direction": "Chunk→Chunk (within same Section)" }
    ]
}
```

### 스키마 특징
- **Intro/Body 분리**: "follows:" 기준으로 본문만을 대상으로 청크 생성하며, `body_span`으로 원본 인덱스 보존
- **청크 span**: `document.body_span`을 기준으로 계산된 절대 인덱스 `[start,end]`
- **요약/벡터**: Summarizer가 섹션/아티클 요약을 배치로 생성, DB 저장 시 Article/Section/Chunk의 `vector`에 임베딩 저장
- **관계 방향**: 상위→하위(`CHILD`), 섹션 내 인접 청크 간 `NEXT`/`PREV`
