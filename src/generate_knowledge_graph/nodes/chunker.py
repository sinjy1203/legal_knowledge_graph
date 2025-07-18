import random
from typing_extensions import Self
from collections.abc import Sequence
from pydantic import BaseModel, model_validator, computed_field
from logger import setup_logger
from langgraph.types import Command
from generate_knowledge_graph.utils.model import Chunk


CHUNK_SIZE = 500

logger = setup_logger()


class Chunker:
    def __call__(self, state):
        logger.info("chunking data...")
        
        chunks: list[Chunk] = []
        for document in state.documents:
            text_splits: list[str] = []
            for i in range(0, len(document.content), CHUNK_SIZE):
                text_splits.append(document.content[i : i + CHUNK_SIZE])

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
                "chunks": chunks
            },
            goto="EntityRelationExtractor"
        )