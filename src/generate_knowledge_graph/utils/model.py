from pydantic import BaseModel
from typing import List, Tuple


class Document(BaseModel):
    file_path: str = ""
    content: str = ""


class Chunk(BaseModel):
    file_path: str = ""
    span: Tuple[int, int] = (0, 0)
    content: str = ""
    entities: List = []
    relationships: List = []