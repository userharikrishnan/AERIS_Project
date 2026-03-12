from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class UIElement:
    """
    Read-only representation of a perceived UI affordance.
    """
    element_type: str            # "button", "link", "field", "menu"
    label: str                   # visible text
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    window_title: Optional[str]
    confidence: float            # perception confidence (0..1)
