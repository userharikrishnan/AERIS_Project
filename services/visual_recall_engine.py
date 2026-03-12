import json
import sqlite3
from typing import List, Dict, Any


class VisualRecallEngine:
    """
    Read-only semantic recall over visual memory.
    """

    def __init__(self, db_path="db/memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def recall_by_window(self, window_title: str, limit: int = 5) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT payload FROM multimodal_memory
            WHERE type = 'visual'
            ORDER BY id DESC
            """
        )

        results = []
        for row in cursor.fetchall():
            payload = json.loads(row[0])
            if payload.get("active_window") == window_title:
                results.append(payload)
                if len(results) >= limit:
                    break

        return results

    def recall_by_intent(self, intent_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT payload FROM multimodal_memory
            WHERE type = 'visual'
            ORDER BY id DESC
            """
        )

        results = []
        for row in cursor.fetchall():
            payload = json.loads(row[0])
            if payload.get("intent_type") == intent_type:
                results.append(payload)
                if len(results) >= limit:
                    break

        return results

    def recall_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT payload FROM multimodal_memory
            WHERE type = 'visual'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )

        return [json.loads(row[0]) for row in cursor.fetchall()]
