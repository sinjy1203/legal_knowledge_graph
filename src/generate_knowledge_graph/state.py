from dataclasses import dataclass, field
from typing import List, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class ContextSchema:
    clear_database: bool = field(default=True)
    benchmark_name: str = field(default="maud")
    table_of_contents_extractor_prompt: ChatPromptTemplate = field(default=None)
    summarizer_prompt: ChatPromptTemplate = field(default=None)
    
    semantic_chunking_config: dict = field(default_factory=dict)
    use_cache: bool = field(default=False)


@dataclass
class State:
    documents: list = field(default_factory=list)