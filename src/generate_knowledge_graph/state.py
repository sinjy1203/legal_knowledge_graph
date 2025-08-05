from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ContextSchema:
    clear_database: bool = field(default=True)
    benchmark_name: str = field(default="maud")
    
    chunk_size: int = field(default=500)



@dataclass
class State:
    documents: list = field(default_factory=list)
    chunks: list = field(default_factory=list)