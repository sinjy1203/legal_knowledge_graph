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
            elif runtime.context.chunking_strategy == "page":
                # 페이지 구분선 "________________" 기준으로 분리
                delimiter = "________________"
                # 분리 후 각 페이지 조각을 다시 chunk_size로 재분할하지 않고 그대로 사용
                text_splits = body_text.split(delimiter)
                # 구분선 자체 길이를 고려하여 span 계산의 상대 인덱스를 맞추기 위해
                # 아래 span 계산 루프에서 조각 길이만큼만 이동하고, 구분선 길이는 건너뜁니다.
                # 이를 위해 delimiter 길이를 저장합니다.
                page_mode = True
            else:
                text_splits = [body_text]
            # Get spans from chunks relative to body, then map to absolute using body_span
            current_rel_index: int = 0
            for idx, text_split in enumerate(text_splits):
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
                # page 모드인 경우, 원문에서 구분선 문자열도 존재하므로 그 길이만큼 상대 인덱스를 추가로 전진
                if runtime.context.chunking_strategy == "page" and idx < len(text_splits) - 1:
                    current_rel_index += len("________________")

        logger.info(f"chunked {len(chunks)} chunks")
        
        return Command(
            update={
                "chunks": chunks
            },
            goto="DocumentStructureDetector"
        )