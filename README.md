# Legal Knowledge Graph RAG 시스템

</br>

## 프로젝트 개요

본 프로젝트는 **법률 문서 검색을 위한 지식 그래프 기반 RAG(Retrieval-Augmented Generation) 시스템**입니다. 

### 주요 목적
- **입력**: 법률 관련 질문
- **출력**: 질문에 답변하기 위해 필요한 문서의 `file_path`와 해당 텍스트 청크의 `span` 정보
- **목표**: 복잡한 법률 질의에 대해 정확하고 관련성 높은 문서 부분을 효율적으로 검색

### 시스템 구조
1. **지식 그래프 구축**: 법률 문서에서 엔티티와 관계를 추출하여 Neo4j 그래프 데이터베이스 구축
2. **검색 Agent**: LangGraph 기반 ReAct 패턴을 활용한 지능형 검색 시스템
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
│   │       ├── chunker.py         # 문서 분할(Chunk) 처리
│   │       ├── entity_relation_extractor.py # 엔티티/관계 추출 및 검증
│   │       ├── graph_db_writer.py # 추출된 정보 Neo4j 그래프DB에 저장
│   │       └── entity_resolver.py # 엔티티 정규화 및 중복 해결 기능
│   └── search_knowledge_graph/    # 지식 그래프 검색 관련 모듈
│       ├── state.py               # 검색 agent 상태 및 설정 데이터 클래스
│       ├── agent.py               # ReAct agent 구현 (LangGraph 기반)
│       ├── prompt.py              # 지식 그래프 검색용 프롬프트 정의
│       └── tools/                 # 검색용 도구 모음
│           ├── __init__.py        # 도구 모듈 임포트 관리
│           ├── search_entity.py   # 엔티티 벡터 유사도 검색 도구
│           ├── search_relationship.py # 관계 벡터 유사도 검색 도구
│           ├── search_connected_entity.py # 연결된 엔티티 검색 도구
│           ├── get_mention_chunk.py # 엔티티 언급 청크 검색 도구
│           └── get_chunk_info.py  # 최종 청크 정보 반환 도구
```

</br>

## 주요 컴포넌트 설명

### 1. 지식 그래프 생성 (Generate Knowledge Graph)
- **목적**: 법률 문서에서 엔티티와 관계를 추출하여 Neo4j 그래프 데이터베이스 구축
- **핵심 파일**:
  - `src/generate.py`: 전체 파이프라인 실행 엔트리포인트
  - `src/generate_knowledge_graph/nodes/entity_relation_extractor.py`: LLM 기반 엔티티/관계 추출
  - `src/generate_knowledge_graph/nodes/entity_resolver.py`: 엔티티 정규화 및 중복 해결

### 2. 검색 Agent (Search Knowledge Graph)
- **목적**: 사용자 질문에 대해 지식 그래프를 체계적으로 탐색하여 최적의 문서 청크 검색
- **핵심 파일**:
  - `src/search.py`: 검색 agent 실행 엔트리포인트
  - `src/search_knowledge_graph/agent.py`: ReAct 패턴 기반 검색 로직
  - `src/search_knowledge_graph/tools/`: 전문화된 검색 도구들
    - `search_entity.py`: 엔티티 벡터 유사도 검색
    - `search_relationship.py`: 관계 벡터 유사도 검색
    - `search_connected_entity.py`: 특정 엔티티와 연결된 엔티티 검색
    - `get_mention_chunk.py`: 엔티티를 언급하는 텍스트 청크 검색
    - `get_chunk_info.py`: 최종 답변용 청크 정보 반환 (Return Direct)

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

본 시스템에서 사용하는 Neo4j 지식 그래프의 스키마는 다음과 같습니다:

```JSON
{
    "Node": [
        {
            "type": "Chunk",
            "description": "A chunk of text from a document.",
            "properties": {
                "id": "chunk id",
                "content": "chunk content",
                "file_path": "file path",
                "span": "span of the chunk",
                "vector": "chunk embedding vector"
            }
        },
        {
            "type": "Acquirer",
            "description": "The company or entity that is purchasing or taking control of another company in a merger or acquisition transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "TargetCompany",
            "description": "The company being acquired, purchased, or merged into another entity in an M&A transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "MergerVehicle",
            "description": "A subsidiary entity created specifically to facilitate the merger transaction, which typically merges with and into the target company",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "HoldingCompany",
            "description": "A parent company that owns controlling interests in other companies and is used to structure complex acquisition transactions",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "FinancialAdvisor",
            "description": "Investment banks or financial advisory firms that provide valuation opinions, strategic advice, and facilitate M&A transactions",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "LawFirm",
            "description": "Legal counsel representing parties in M&A transactions, providing legal advice and drafting transaction documents",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "PayingAgent",
            "description": "A financial institution designated to handle the exchange of cash and securities to target company shareholders in an M&A transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "ExchangeAgent",
            "description": "A financial institution that facilitates the exchange of target company shares for merger consideration (cash and/or acquirer shares)",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "Trustee",
            "description": "A financial institution serving as trustee for debt instruments (bonds, notes) that may be affected by the M&A transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "KnowledgeDefinition",
            "description": "Legal standard defining what constitutes \"knowledge\" for representation and warranty purposes, typically limited to specific individuals' actual knowledge",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "MaterialAdverseEffectDefinition",
            "description": "Legal definition of changes or events significant enough to materially impact a company's business, allowing deal termination or price adjustment",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "CompanyMaterialAdverseEffect",
            "description": "Specific definition of material adverse effect applicable to the target company, with detailed carve-outs for general economic conditions",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "AcquisitionProposalDefinition",
            "description": "Legal definition of competing takeover proposals that trigger no-shop restrictions and disclosure obligations in M&A agreements",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "SuperiorProposalDefinition",
            "description": "Legal definition of a competing proposal that is more favorable to shareholders, allowing target board to change recommendation or terminate agreement",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "AcceptableConfidentialityAgreement",
            "description": "Standard for confidentiality agreements that must be signed before providing due diligence information to potential competing bidders",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "ChangeOfRecommendation",
            "description": "Actions by target company board that constitute withdrawal or modification of their recommendation in favor of the merger agreement",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "MergerConsideration",
            "description": "The price and form of payment (cash, stock, or mixed) that target company shareholders receive in exchange for their shares",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "TerminationFee",
            "description": "Penalty fee paid by target company to acquirer if the deal is terminated under specific circumstances, designed to deter competing bids",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "HSRApproval",
            "description": "Required antitrust clearance from U.S. federal agencies under the Hart-Scott-Rodino Act for transactions above certain size thresholds",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "AntitrustApproval",
            "description": "Required regulatory clearances from competition authorities to ensure the transaction doesn't violate antitrust laws",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "SECFilingRequirements",
            "description": "Required filings and approvals from the Securities and Exchange Commission for public company transactions involving securities issuance",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "CompanyStockholderApproval",
            "description": "Required vote by target company shareholders to approve the merger agreement and authorize the transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "NoLegalProhibition",
            "description": "Condition ensuring no court orders, injunctions, or laws prohibit the completion of the merger transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "NoShopProvision",
            "description": "Contractual restriction preventing target company from soliciting or encouraging competing acquisition proposals during the deal process",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "FiduciaryOut",
            "description": "Exception allowing target company board to change recommendation or terminate agreement when required by fiduciary duties to shareholders",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "FiduciaryTerminationRight",
            "description": "Target company's right to terminate the merger agreement to accept a superior proposal, typically requiring payment of termination fee",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "TailProvision",
            "description": "Mechanism extending termination fee obligations for a specified period after deal termination if target enters alternative transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "OrdinaryCourseCovenant",
            "description": "Contractual obligation for target company to operate its business in the ordinary course during the period between signing and closing",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "NegativeCovenant",
            "description": "Contractual restrictions on specific actions target company cannot take without acquirer consent during the interim period",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "ConsiderationStructure",
            "description": "The specific terms and structure of payment to target shareholders, including cash, stock, or combination thereof",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "TransactionType",
            "description": "The legal structure used to complete the acquisition, such as one-step merger, two-step merger, or tender offer followed by merger",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "ExchangeRatio",
            "description": "The ratio determining how many acquirer shares target shareholders receive for each target share in a stock-for-stock transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "MinimumCondition",
            "description": "Minimum ownership threshold that must be achieved in a tender offer before the acquirer is obligated to purchase tendered shares",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "CovenantCompliance",
            "description": "Requirement that target company has complied with all interim period covenants as a condition to closing the transaction",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "NoMaterialAdverseEffect",
            "description": "Requirement that no material adverse effect has occurred to target company as a condition to completing the merger",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "RepresentationAccuracy",
            "description": "Requirement that target company's representations and warranties remain true and correct as of closing date",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "OrdinaryCourseOperations",
            "description": "Target company's commitment to operate business normally during interim period to preserve business value until closing",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "ProhibitedActions",
            "description": "Specific corporate actions target company is prohibited from taking during interim period without acquirer consent",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "SpecificRestrictions",
            "description": "Detailed restrictions on specific business activities like employee compensation changes or new acquisitions during interim period",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "SpecificPerformance",
            "description": "Contractual provision allowing parties to seek court orders compelling performance of merger agreement obligations",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "TerminationFeeTriggers",
            "description": "Specific events or circumstances that trigger obligation to pay termination fee to protect acquirer's deal investment",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        },
        {
            "type": "DamagesLimitations",
            "description": "Contractual caps on monetary damages available to parties, typically limiting remedies to termination fees in certain circumstances",
            "properties": {
                "id": "entity name",
                "vector": "entity name embedding vector"
            }
        }
    ],
    "Edge": [
        {
            "type": "MENTIONS",
            "description": "A relationship between a chunk and an entity."
        },
        {
            "type": "Acquires",
            "description": "Relationship indicating one company is acquiring ownership or control of another company through the merger or acquisition transaction",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "Owns",
            "description": "Ownership relationship between parent companies and their subsidiaries, often used to structure complex acquisition transactions",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "MergesWith",
            "description": "Legal relationship where one entity merges with and into another, typically with one entity surviving and the other ceasing to exist",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "Advises",
            "description": "Professional advisory relationship where financial advisors provide valuation opinions, strategic advice, and transaction facilitation services",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "Represents",
            "description": "Legal representation relationship where law firms provide legal counsel and services to parties in M&A transactions",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "DefinedIn",
            "description": "Relationship indicating where legal terms and definitions are specifically defined within merger agreement documents",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "CrossReferences",
            "description": "Internal document references linking related provisions within merger agreements to create coherent legal framework",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "IncorporatesByReference",
            "description": "Legal mechanism incorporating external documents or schedules into the main merger agreement by reference",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "GovernedBy",
            "description": "Relationship indicating which state or jurisdiction's laws govern the interpretation and enforcement of the merger agreement",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "SubjectTo",
            "description": "Conditional relationship where transaction completion depends on satisfaction of specific regulatory or other conditions",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "RequiresApprovalFrom",
            "description": "Relationship indicating which parties must provide formal approval for the transaction to proceed legally",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "Pays",
            "description": "Financial relationship indicating payment obligations from acquirer to target shareholders as merger consideration",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "TriggersFee",
            "description": "Conditional relationship where specific events or breaches trigger obligation to pay termination or other fees",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "TriggersIf",
            "description": "Conditional logic relationship where specific actions automatically trigger other consequences or obligations",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "ExceptWhen",
            "description": "Exception relationship allowing deviation from general rules under specific circumstances, often related to fiduciary duties",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        },
        {
            "type": "SubjectToCondition",
            "description": "Conditional relationship where actions or closing depends on satisfaction of specific conditions or performance standards",
            "properties": {
                "description": "edge description",
                "vector": "edge description embedding vector"
            }
        }
    ]
}
```

### 스키마 특징
- **Chunk 노드**: 원본 텍스트 청크와 임베딩 벡터 저장
- **법률 엔티티**: M&A 거래의 핵심 참여자와 법적 개념들 (40+ 타입)
- **관계 타입**: 법률 문서의 의미적 관계를 표현하는 17가지 관계 타입
- **벡터 검색**: 모든 노드와 엣지에 임베딩 벡터 저장으로 의미적 유사도 검색 지원
- **청크 추적**: 모든 관계에 `chunk_id` 저장으로 원본 텍스트 추적 가능
