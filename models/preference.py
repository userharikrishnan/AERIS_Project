from dataclasses import dataclass
import time


@dataclass
class PreferenceSignal:
    """
    A learned preference signal.
    """
    key: str
    value: float
    confidence: float
    timestamp: float = time.time()
