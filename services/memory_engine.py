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
    Persistent semantic memory store with interaction tracking, outcome learning,
    and intelligent retrieval capabilities.
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

        # 🔹 NEW: Structured interaction memory
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY,
            user_input TEXT,
            intent TEXT,
            reasoning TEXT,
            confidence REAL,
            importance REAL,
            timestamp REAL
        )
        """)

        # 🔹 NEW: Outcome tracking memory
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY,
            action TEXT,
            success INTEGER,
            context TEXT,
            timestamp REAL
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

    # -------------------------
    # 🔹 NEW: Interaction memory
    # -------------------------
    def remember_interaction(self, user_input, intent, reasoning, confidence):
        importance = confidence  # Importance weighting based on confidence
        self.conn.execute("""
            INSERT INTO interactions(user_input, intent, reasoning, confidence, importance, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_input, intent, reasoning, confidence, importance, time.time()))
        self.conn.commit()

    # -------------------------
    # 🔹 NEW: Recent memory retrieval
    # -------------------------
    def get_recent(self, limit=5):
        cursor = self.conn.execute("""
            SELECT user_input, intent, reasoning, confidence
            FROM interactions
            ORDER BY importance DESC, id DESC
            LIMIT ?
        """, (limit,))
        
        return [
            {
                "text": r[0],
                "intent": r[1],
                "reasoning": r[2],
                "confidence": r[3]
            }
            for r in cursor.fetchall()
        ]

    # -------------------------
    # 🔹 NEW: Outcome tracking
    # -------------------------
    def record_outcome(self, action, success, context):
        self.conn.execute("""
            INSERT INTO outcomes(action, success, context, timestamp)
            VALUES (?, ?, ?, ?)
        """, (action, int(success), json.dumps(context), time.time()))
        self.conn.commit()

    def get_recent_outcomes(self, limit=5):
        cursor = self.conn.execute("""
            SELECT action, success, context, timestamp
            FROM outcomes
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        return [
            {
                "action": r[0],
                "success": bool(r[1]),
                "context": json.loads(r[2]),
                "timestamp": r[3]
            }
            for r in cursor.fetchall()
        ]

    # -------------------------
    # 🔹 NEW: Relevance scoring
    # -------------------------
    def search_memory(self, query: str, limit=5):
        cursor = self.conn.execute("""
            SELECT user_input, intent, reasoning
            FROM interactions
        """)

        results = []
        for r in cursor.fetchall():
            score = query.lower() in r[0].lower()
            if score:
                results.append({
                    "text": r[0],
                    "intent": r[1],
                    "reasoning": r[2]
                })

        return results[:limit]

    # -------------------------
    # 🔹 NEW: Unified recall
    # -------------------------
    def get_context_bundle(self):
        return {
            "recent": self.get_recent(),
            "long_term": self.fetch_long_term()[:5],
            "visual": self.fetch_multimodal("visual")[:3]
        }

    # -------------------------
    # 🔹 NEW: Memory interface layer
    # -------------------------
    def build_context(self, query: str):
        return {
            "relevant": self.search_memory(query),
            "recent": self.get_recent(),
            "outcomes": self.get_recent_outcomes()
        }


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