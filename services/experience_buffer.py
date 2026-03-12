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
            plan TEXT,
            response TEXT,
            success INTEGER
        )
        """)
        self.conn.commit()

    def add_experience(self, input_text, plan, response, success: bool):
        self.conn.execute(
            "INSERT INTO experiences VALUES (NULL,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                input_text,
                str(plan),
                response,
                int(success)
            )
        )
        self.conn.commit()

    def fetch_all(self, limit=1000):
        cursor = self.conn.execute(
            "SELECT input_text, response FROM experiences ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()
