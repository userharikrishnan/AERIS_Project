from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationResult:
    preview: dict
    safe: bool
    reason: str
