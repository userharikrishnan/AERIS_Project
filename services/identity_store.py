import sqlite3
import json
from models.identity import IdentityState


class IdentityStore:
    """
    Persists IdentityState to SQLite.
    """

    def __init__(self, db_path="db/identity.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS identity (
            id INTEGER PRIMARY KEY,
            payload TEXT
        )
        """)
        self.conn.commit()

    def save(self, state: IdentityState):
        self.conn.execute("DELETE FROM identity")
        self.conn.execute(
            "INSERT INTO identity(payload) VALUES (?)",
            (json.dumps(state.to_dict()),)
        )
        self.conn.commit()

    def load(self) -> IdentityState | None:
        cursor = self.conn.execute(
            "SELECT payload FROM identity ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            return None

        data = json.loads(row[0])
        return IdentityState.from_dict(data)
