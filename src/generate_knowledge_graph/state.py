from dataclasses import dataclass, field
from typing import List, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class ContextSchema:
    clear_database: bool = field(default=True)
    benchmark_name: str = field(default="maud")
    table_of_contents_extractor_prompt: ChatPromptTemplate = field(default=None)
    contract_start_finder_prompt: ChatPromptTemplate = field(default=None)
    document_structure_detector_prompt: ChatPromptTemplate = field(default=None)
    summarizer_prompt: ChatPromptTemplate = field(default=None)
    
    chunking_strategy: Literal["naive", "rcts"] = field(default="naive")
    chunk_size: int = field(default=500)




@dataclass
class State:
    documents: list = field(default_factory=list)
    table_of_contents: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    hierarchical_chunk_ids: dict = field(default_factory=dict)