from dataclasses import dataclass, field
from typing import Annotated, Optional


def custom_add_messages(existing: list, update: list):
    return existing + update

@dataclass
class Config:
    max_execute_tool_count: int = field(default=5)

@dataclass
class State:
    messages: Annotated[list, custom_add_messages] = field(default_factory=list)
    execute_tool_count: int = field(default=0)