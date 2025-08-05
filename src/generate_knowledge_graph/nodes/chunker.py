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
            if runtime.context.chunking_strategy == "naive":
                text_splits: list[str] = []
                for i in range(0, len(document.content), runtime.context.chunk_size):
                    text_splits.append(document.content[i : i + runtime.context.chunk_size])
            elif runtime.context.chunking_strategy == "rcts":
                synthetic_data_splitter = RecursiveCharacterTextSplitter(
                    separators=[
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
                text_splits = synthetic_data_splitter.split_text(document.content)
            # Get spans from chunks
            prev_span: tuple[int, int] | None = None
            for text_split in text_splits:
                prev_index = prev_span[1] if prev_span is not None else 0
                span = (prev_index, prev_index + len(text_split))
                chunks.append(
                    Chunk(
                        file_path=document.file_path,
                        span=span,
                        content=text_split,
                    )
                )
                prev_span = span

        logger.info(f"chunked {len(chunks)} chunks")
        
        return Command(
            update={
                "chunks": chunks[:10]
            },
            goto="EntityRelationExtractor"
        )