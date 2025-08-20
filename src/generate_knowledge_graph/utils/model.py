from pydantic import BaseModel
from typing import List, Tuple


class Document(BaseModel):
    file_path: str = ""
    content: str = ""
    intro: str = ""
    body: str = ""
    body_span: Tuple[int, int] = (0, 0)


class Chunk(BaseModel):
    file_path: str = ""
    span: Tuple[int, int] = (0, 0)
    content: str = ""
    order: int = 0
    summary: str = ""
    clause_list: list = []