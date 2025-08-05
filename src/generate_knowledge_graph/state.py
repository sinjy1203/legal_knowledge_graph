from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class ContextSchema:
    clear_database: bool = field(default=True)
    benchmark_name: str = field(default="maud")
    
    chunking_strategy: Literal["naive", "rcts"] = field(default="naive")
    chunk_size: int = field(default=500)




@dataclass
class State:
    documents: list = field(default_factory=list)
    chunks: list = field(default_factory=list)