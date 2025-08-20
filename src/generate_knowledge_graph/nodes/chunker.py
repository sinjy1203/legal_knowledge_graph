import random
from typing_extensions import Self
from collections.abc import Sequence
from pydantic import BaseModel, model_validator, computed_field
from logger import setup_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command
from langgraph.runtime import Runtime

from generate_knowledge_graph.utils.model import Chunk
from generate_knowledge_graph.state import ContextSchema


logger = setup_logger()


class Chunker:
    def __call__(self, state, runtime: Runtime[ContextSchema]):
        logger.info("chunking data...")
        
        chunks: list[Chunk] = []
        for document in state.documents:
            # Use document.body for chunking
            body_text = document.body if hasattr(document, "body") and document.body is not None else document.content
            body_span = document.body_span if hasattr(document, "body_span") and document.body_span is not None else (0, len(document.content))

            if runtime.context.chunking_strategy == "naive":
                text_splits: list[str] = []
                for i in range(0, len(body_text), runtime.context.chunk_size):
                    text_splits.append(body_text[i : i + runtime.context.chunk_size])
            elif runtime.context.chunking_strategy == "rcts":
                synthetic_data_splitter = RecursiveCharacterTextSplitter(
                    separators=[
                        "ARTICLE",
                        "________________",
                        "\n\n",
                        "\n"
                    ],
                    chunk_size=runtime.context.chunk_size,
                    chunk_overlap=0,
                    length_function=len,
                    is_separator_regex=False,
                    strip_whitespace=False,
                )
                text_splits = synthetic_data_splitter.split_text(body_text)
            # Get spans from chunks relative to body, then map to absolute using body_span
            current_rel_index: int = 0
            for text_split in text_splits:
                start_rel = current_rel_index
                end_rel = start_rel + len(text_split)
                span = (body_span[0] + start_rel, body_span[0] + end_rel)
                chunks.append(
                    Chunk(
                        file_path=document.file_path,
                        span=span,
                        content=text_split,
                    )
                )
                current_rel_index = end_rel

        logger.info(f"chunked {len(chunks)} chunks")
        
        return Command(
            update={
                "chunks": chunks
            },
            goto="DocumentStructureDetector"
        )