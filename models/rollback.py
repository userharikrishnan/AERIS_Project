from dataclasses import dataclass
from typing import Dict, Any
import time


@dataclass
class RollbackRecord:
    action: str
    params: Dict[str, Any]
    rollback_params: Dict[str, Any]
    timestamp: float = time.time()
