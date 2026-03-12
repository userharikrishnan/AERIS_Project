from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TextBlock:
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class WindowInfo:
    title: str
    bbox: Tuple[int, int, int, int]
    active: bool


@dataclass
class VisualState:
    resolution: Tuple[int, int]
    active_window: str
    windows: List[WindowInfo]
    text_blocks: List[TextBlock]
