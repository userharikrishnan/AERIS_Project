import json
import threading
import os

from services.security_models import AuditEvent
from services.audit_hash_chain import AuditHashChain


class AuditLogger:
    """
    Append-only, hash-chained audit logger.
    Any tampering breaks the chain.
    """

    def __init__(self, file_path="audit.log"):
        self.file_path = file_path
        self._lock = threading.Lock()
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        """
        Load the last hash from the audit log if it exists.
        """
        if not os.path.exists(self.file_path):
            return "GENESIS"

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                last_line = None
                for line in f:
                    last_line = line
                if not last_line:
                    return "GENESIS"

                record = json.loads(last_line)
                return record.get("hash", "GENESIS")
        except Exception:
            # If corrupted, force visible break
            return "CORRUPTED"

    def log(self, event: AuditEvent):
        record = event.to_dict()

        payload = json.dumps(record, sort_keys=True)
        current_hash = AuditHashChain.compute_hash(
            self._last_hash,
            payload
        )

        chained_record = {
            **record,
            "prev_hash": self._last_hash,
            "hash": current_hash
        }

        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(chained_record) + "\n")

            self._last_hash = current_hash
