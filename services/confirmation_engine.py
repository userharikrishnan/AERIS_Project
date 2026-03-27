import time
from typing import Dict
from services.confirmation_models import PendingConfirmation


class ConfirmationEngine:
    """
    Holds pending user confirmations.
    In-memory by design (ephemeral, safe).
    """

    def __init__(self, ttl_seconds: int = 120):
        self.ttl = ttl_seconds
        self.pending: Dict[str, PendingConfirmation] = {}

    def create(self, action: str, params: dict, ui_candidates=None) -> PendingConfirmation:
        confirmation = PendingConfirmation(
            confirmation_id=str(time.time_ns()),
            action=action,
            params=params,
            ui_candidates=ui_candidates,
            created_at=time.time()
        )
        self.pending[confirmation.confirmation_id] = confirmation
        return confirmation

    def get(self, confirmation_id: str) -> PendingConfirmation | None:
        self._cleanup()
        return self.pending.get(confirmation_id)

    def consume(self, confirmation_id: str) -> PendingConfirmation | None:
        self._cleanup()
        return self.pending.pop(confirmation_id, None)

    def _cleanup(self):
        now = time.time()
        expired = [
            cid for cid, c in self.pending.items()
            if now - c.created_at > self.ttl
        ]
        for cid in expired:
            self.pending.pop(cid, None)
