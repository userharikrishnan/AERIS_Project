import sqlite3
from datetime import datetime

class ExperienceBuffer:
    def __init__(self, db_path="db/memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            input_text TEXT,
            intent TEXT,
            action TEXT,
            plan TEXT,
            response TEXT,
            confidence REAL,
            success INTEGER,
            reward REAL
        )
        """)
        self.conn.commit()

    def _compute_reward(self, success, confidence):
        if success:
            return 1.0 * confidence
        else:
            return -1.0 * (1 - confidence)

    def add_experience(self, input_text, intent, action, plan, response, confidence, success: bool):
        reward = self._compute_reward(success, confidence)

        self.conn.execute(
            "INSERT INTO experiences VALUES (NULL,?,?,?,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                input_text,
                intent,
                action,
                str(plan),
                response,
                confidence,
                int(success),
                reward
            )
        )
        self.conn.commit()

    def fetch_all(self, limit=1000):
        cursor = self.conn.execute(
            "SELECT input_text, response FROM experiences WHERE reward > 0 ORDER BY reward DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()