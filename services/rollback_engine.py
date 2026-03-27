from typing import Dict, List
from models.rollback import RollbackRecord


class RollbackEngine:
    """
    Manages reversible actions and explicit undo.
    Supports transactional (multi-step) rollback.
    """

    def __init__(self):
        # Single-action rollback (existing behavior)
        self._stack: Dict[str, RollbackRecord] = {}

        # Transactional rollback: tx_id -> ordered records
        self._transactions: Dict[str, List[RollbackRecord]] = {}

    # --------------------------------------------------
    # Single-action rollback (UNCHANGED)
    # --------------------------------------------------

    def register(self, execution_id: str, record: RollbackRecord):
        self._stack[execution_id] = record

    def rollback(self, execution_id: str) -> RollbackRecord | None:
        return self._stack.pop(execution_id, None)

    def has_rollback(self, execution_id: str) -> bool:
        return execution_id in self._stack

    # --------------------------------------------------
    # Transactional rollback (9.3)
    # --------------------------------------------------

    def begin_transaction(self, tx_id: str):
        if tx_id not in self._transactions:
            self._transactions[tx_id] = []

    def register_in_transaction(
        self,
        tx_id: str,
        record: RollbackRecord
    ):
        if tx_id not in self._transactions:
            self._transactions[tx_id] = []

        self._transactions[tx_id].append(record)

    def commit_transaction(self, tx_id: str):
        """
        Clears rollback history after successful completion.
        """
        self._transactions.pop(tx_id, None)

    def rollback_transaction(self, tx_id: str) -> List[RollbackRecord]:
        """
        Rolls back all steps in reverse execution order.
        """
        records = self._transactions.pop(tx_id, [])
        return list(reversed(records))
