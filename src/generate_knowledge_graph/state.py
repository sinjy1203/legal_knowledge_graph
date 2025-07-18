from dataclasses import dataclass, field
from typing import List, Optional
from generate_knowledge_graph.utils.model import Document, Chunk


@dataclass
class Config:
    chunk_size: int = field(default=500)


@dataclass
class State:
    clear_database: bool = field(default=False)
    benchmark_name: str = field(default="")
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)




    # items: list[dict] = field(default_factory=list)
    # responses: list[dict] = field(default_factory=list)
    # start_date: Optional[str] = None  # "YYYY-MM-DD" 형식의 문자열
    # end_date: Optional[str] = None    # "YYYY-MM-DD" 형식의 문자열
    # tables: List[str] = field(default_factory=lambda: ["news", "market_news", "sec", "earnings_call"])
    # clear_database: bool = False  # 데이터베이스 초기화 여부