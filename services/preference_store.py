import sqlite3
import json
from models.preference import PreferenceSignal


class PreferenceStore:
    """
    Persistent user preference storage.
    """

    def __init__(self, db_path="db/preferences.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY,
            key TEXT,
            payload TEXT
        )
        """)
        self.conn.commit()

    def save(self, pref: PreferenceSignal):
        self.conn.execute(
            "INSERT INTO preferences(key, payload) VALUES (?, ?)",
            (pref.key, json.dumps(pref.__dict__))
        )
        self.conn.commit()

    def get_recent(self, key: str, limit: int = 5):
        cursor = self.conn.execute(
            "SELECT payload FROM preferences WHERE key=? ORDER BY id DESC LIMIT ?",
            (key, limit)
        )
        return [json.loads(row[0]) for row in cursor.fetchall()]
