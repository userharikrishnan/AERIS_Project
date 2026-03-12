import time
import json
from services.memory_engine import MemoryEngine


class MemoryDecayEngine:
    """
    Applies time-based decay to long-term memory.
    """

    HALF_LIFE_SECONDS = 60 * 60 * 24 * 7  # 7 days

    def __init__(self, memory_engine: MemoryEngine):
        self.memory = memory_engine

    def decay(self):
        records = self.memory.fetch_long_term()
        now = time.time()

        retained = []
        for r in records:
            age = now - r["last_accessed"]
            decay_factor = 0.5 ** (age / self.HALF_LIFE_SECONDS)

            if r["importance"] * decay_factor > 0.2:
                r["importance"] *= decay_factor
                retained.append(r)

        # Rewrite table safely
        self.memory.conn.execute("DELETE FROM long_term_memory")
        for r in retained:
            self.memory.conn.execute(
                "INSERT INTO long_term_memory(payload) VALUES (?)",
                (json.dumps(r),)
            )
        self.memory.conn.commit()
