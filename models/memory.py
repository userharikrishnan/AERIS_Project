from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


@dataclass
class TextMemory:
    text: str
    timestamp: float = time.time()


@dataclass
class VisualMemory:
    """
    Distilled visual memory (NO images).
    """
    active_window: str
    salient_text: str
    intent_type: str
    confidence: float
    timestamp: float = time.time()


@dataclass
class EpisodicMemoryRecord:
    """
    A single remembered episode (what + why).
    """
    summary: str
    metadata: Dict[str, Any]
    timestamp: float = time.time()
