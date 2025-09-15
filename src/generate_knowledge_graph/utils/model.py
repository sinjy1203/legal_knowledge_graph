from pydantic import BaseModel, Field
from typing import List, Tuple, ForwardRef


ChunkRef = ForwardRef('Chunk')


class Document(BaseModel):
    file_path: str = ""
    intro: str = ""
    table_of_contents: dict = {}
    span: Tuple[int, int] = (0, 0)
    content: str = ""
    summary: str = ""
    # dict 트리 구조를 보관하고, 리프는 Chunk 인스턴스로 치환하여 저장
    children: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    name: str = ""
    span: Tuple[int, int] = (0, 0)
    content: str = ""
    summary: str = ""
    children: List[ChunkRef] = []
