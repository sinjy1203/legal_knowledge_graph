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