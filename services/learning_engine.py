"""
AERIS Learning Engine
Passive and active learning from user interactions.

Two learning modes:
1. Passive Learning: Observes user behavior and records patterns
   - Action approvals/rejections → smart confirmation trust
   - Entity corrections (e.g., "not notepad, wordpad") → app registry updates
   - Preference signals (browser choice, report format, etc.)

2. Active Learning: Triggered by user corrections
   - User says "that's wrong, I meant X" → updates personal knowledge base
   - User corrects an extracted entity → updates extraction preference
   - User teaches a new app mapping → updates app registry permanently

Personal data is always stored locally (never uploaded).
"""

import sqlite3
import json
import time
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class LearningEngine:
    """
    Observes and learns from AERIS interactions.
    Stores personalized knowledge in a local SQLite DB.
    """

    def __init__(
        self,
        db_path: str = "db/preferences.db",
        app_registry=None,
        smart_confirmation=None
    ):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.app_registry = app_registry
        self.smart_confirmation = smart_confirmation
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS personal_knowledge (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT,
                category TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at REAL,
                updated_at REAL,
                use_count INTEGER DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS correction_log (
                id INTEGER PRIMARY KEY,
                original_input TEXT,
                original_intent TEXT,
                corrected_intent TEXT,
                correction_detail TEXT,
                timestamp REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_overrides (
                id INTEGER PRIMARY KEY,
                intent_type TEXT,
                entity_key TEXT,
                original_value TEXT,
                corrected_value TEXT,
                confidence REAL DEFAULT 1.0,
                use_count INTEGER DEFAULT 0,
                timestamp REAL
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Passive learning — called automatically during normal operation
    # ------------------------------------------------------------------

    def observe_action_outcome(
        self,
        action: str,
        params: dict,
        success: bool,
        user_input: str = "",
        intent: str = ""
    ):
        """
        Observe how an action went.
        Success → boost trust, Failure → degrade trust (in smart confirmation).
        """
        if self.smart_confirmation:
            if success:
                self.smart_confirmation.record_approval(action, params)
            else:
                self.smart_confirmation.record_error(action, params, "execution_failed")

        logger.debug(f"[LearningEngine] Observed: action={action}, success={success}")

    def observe_preference(self, category: str, value: str, confidence: float = 0.8):
        """
        Learn a user preference (e.g., preferred browser, report format).
        """
        self._store_knowledge(
            key=f"preference:{category}",
            value=value,
            category="preference",
            confidence=confidence,
            source="passive_observation"
        )
        logger.info(f"[LearningEngine] Preference learned: {category} → {value}")

    def observe_entity_preference(self, intent_type: str, entity_key: str, value: str):
        """
        Learn preferred entity value for a given intent type.
        E.g., for OPEN_APP intent, 'browser' should always resolve to 'Firefox'.
        """
        self.conn.execute("""
            INSERT INTO entity_overrides (intent_type, entity_key, original_value, corrected_value, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
        """, (intent_type, entity_key, "", value, time.time()))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Active learning — triggered by user corrections
    # ------------------------------------------------------------------

    def learn_correction(
        self,
        original_input: str,
        original_intent: str,
        corrected_intent: str,
        correction_detail: str = ""
    ):
        """
        User corrected AERIS's interpretation.
        Logs the correction and queues it for model fine-tuning.
        """
        self.conn.execute("""
            INSERT INTO correction_log (original_input, original_intent, corrected_intent, correction_detail, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (original_input, original_intent, corrected_intent, correction_detail, time.time()))
        self.conn.commit()
        logger.info(f"[LearningEngine] Correction logged: '{original_intent}' → '{corrected_intent}'")

    def learn_app_mapping(self, app_name: str, path: str):
        """
        User taught a new app name → executable path mapping.
        """
        if self.app_registry:
            self.app_registry.teach(app_name, path)
        self._store_knowledge(
            key=f"app:{app_name.lower()}",
            value=path,
            category="app_mapping",
            confidence=1.0,
            source="user_taught"
        )
        logger.info(f"[LearningEngine] App mapping learned: '{app_name}' → {path}")

    def learn_browser_preference(self, browser_name: str):
        """User specified their preferred browser."""
        self.observe_preference("default_browser", browser_name, confidence=1.0)
        if self.app_registry:
            self.app_registry.learn_user_preference("browser", browser_name)
        logger.info(f"[LearningEngine] Browser preference: {browser_name}")

    # ------------------------------------------------------------------
    # Retrieval — used during inference
    # ------------------------------------------------------------------

    def get_preference(self, category: str, default=None):
        """Get a learned preference by category."""
        key = f"preference:{category}"
        row = self.conn.execute(
            "SELECT value, confidence FROM personal_knowledge WHERE key = ?", (key,)
        ).fetchone()
        if row and row[1] >= 0.5:
            return row[0]
        return default

    def get_entity_override(self, intent_type: str, entity_key: str) -> Optional[str]:
        """Check if there's a learned override for an entity in a given intent context."""
        row = self.conn.execute("""
            SELECT corrected_value FROM entity_overrides
            WHERE intent_type = ? AND entity_key = ?
            ORDER BY use_count DESC, timestamp DESC LIMIT 1
        """, (intent_type, entity_key)).fetchone()
        return row[0] if row else None

    def get_pending_corrections(self, limit: int = 50) -> List[dict]:
        """Get corrections not yet used for fine-tuning."""
        cursor = self.conn.execute("""
            SELECT original_input, original_intent, corrected_intent, correction_detail, timestamp
            FROM correction_log ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        return [
            {
                "input": r[0],
                "original": r[1],
                "corrected": r[2],
                "detail": r[3],
                "timestamp": r[4]
            }
            for r in cursor.fetchall()
        ]

    def get_personal_knowledge_summary(self) -> dict:
        """Return a summary of all learned personal knowledge."""
        cursor = self.conn.execute(
            "SELECT category, COUNT(*) FROM personal_knowledge GROUP BY category"
        )
        categories = {r[0]: r[1] for r in cursor.fetchall()}

        corrections_count = self.conn.execute("SELECT COUNT(*) FROM correction_log").fetchone()[0]
        overrides_count = self.conn.execute("SELECT COUNT(*) FROM entity_overrides").fetchone()[0]

        return {
            "knowledge_by_category": categories,
            "total_corrections": corrections_count,
            "entity_overrides": overrides_count,
        }

    # ------------------------------------------------------------------
    # Internal storage
    # ------------------------------------------------------------------

    def _store_knowledge(self, key: str, value: str, category: str, confidence: float, source: str):
        self.conn.execute("""
            INSERT INTO personal_knowledge (key, value, category, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = ?,
                confidence = MAX(confidence, ?),
                updated_at = ?,
                use_count = use_count + 1
        """, (key, value, category, confidence, source, time.time(), time.time(),
              value, confidence, time.time()))
        self.conn.commit()
