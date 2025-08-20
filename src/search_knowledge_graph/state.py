from dataclasses import dataclass, field
from typing import Annotated, Optional
from tqdm.auto import tqdm


def custom_add_messages(existing: list, update: list):
    return existing + update

@dataclass
class ContextSchema:
    max_execute_tool_count: int = field(default=5)
    progress_bar: tqdm = field(default=None)

@dataclass
class State:
    messages: Annotated[list, custom_add_messages] = field(default_factory=list)
    execute_tool_count: int = field(default=0)