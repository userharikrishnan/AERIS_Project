import hashlib


class AuditHashChain:
    """
    Maintains hash chaining for audit records.
    """

    @staticmethod
    def compute_hash(previous_hash: str, payload: str) -> str:
        data = (previous_hash + payload).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
