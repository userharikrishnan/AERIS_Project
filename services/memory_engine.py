import sqlite3
import json
import time
from models.memory import (
    VisualMemory,
    EpisodicMemoryRecord,
    LongTermMemoryRecord
)


class MemoryEngine:
    """
    Persistent semantic memory store.
    """

    def __init__(self, db_path="db/memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        # Raw text memory (UNCHANGED)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY,
            text TEXT
        )
        """)

        # Episodic + visual memory
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS multimodal_memory (
            id INTEGER PRIMARY KEY,
            type TEXT,
            payload TEXT
        )
        """)

        # 🔹 Long-term consolidated memory
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY,
            payload TEXT
        )
        """)

        self.conn.commit()

    # -------------------------
    # Existing behavior (UNCHANGED)
    # -------------------------
    def remember(self, text: str):
        self.conn.execute(
            "INSERT INTO memory(text) VALUES (?)",
            (text,)
        )
        self.conn.commit()

    # -------------------------
    # Episodic / visual memory
    # -------------------------
    def remember_visual(self, visual_memory: VisualMemory):
        self._insert_multimodal("visual", visual_memory.__dict__)

    def remember_episode(self, episode: EpisodicMemoryRecord):
        self._insert_multimodal("episode", episode.__dict__)

    def _insert_multimodal(self, mem_type: str, payload: dict):
        self.conn.execute(
            "INSERT INTO multimodal_memory(type, payload) VALUES (?, ?)",
            (mem_type, json.dumps(payload))
        )
        self.conn.commit()

    # -------------------------
    # Long-term memory (NEW)
    # -------------------------
    def remember_long_term(self, record: LongTermMemoryRecord):
        self.conn.execute(
            "INSERT INTO long_term_memory(payload) VALUES (?)",
            (json.dumps(record.__dict__),)
        )
        self.conn.commit()

    def fetch_multimodal(self, mem_type: str):
        cursor = self.conn.execute(
            "SELECT payload FROM multimodal_memory WHERE type = ?",
            (mem_type,)
        )
        return [json.loads(r[0]) for r in cursor.fetchall()]

    def fetch_long_term(self):
        cursor = self.conn.execute(
            "SELECT payload FROM long_term_memory"
        )
        return [json.loads(r[0]) for r in cursor.fetchall()]


class EpisodicMemory:
    """
    Backward compatibility preserved.
    """
    def __init__(self, db_path="db/memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def log(self, text: str):
        self.conn.execute(
            "INSERT INTO memory(text) VALUES (?)",
            (text,)
        )
        self.conn.commit()
