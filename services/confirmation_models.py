from dataclasses import dataclass
from typing import Dict, Any, Optional
import uuid
import time


@dataclass
class PendingConfirmation:
    confirmation_id: str
    action: str
    params: Dict[str, Any]
    ui_candidates: Optional[list]
    created_at: float
