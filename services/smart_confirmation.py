"""
AERIS Smart Confirmation Engine
Dual-mode confirmation: ask first time, auto-execute on repeat.

Logic:
- First time an action+context is seen → request user confirmation
- After user confirms → record trust signal with a fingerprint
- Next time same action+similar context → auto-execute (no confirmation needed)
- UNLESS: params changed significantly, error occurred last time, or action is high-risk
"""

import hashlib
import json
import sqlite3
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Actions that ALWAYS require confirmation regardless of history (high-risk)
ALWAYS_CONFIRM_ACTIONS = {
    "file_delete", "process_kill", "format_drive", "shutdown",
    "rm", "delete", "wipe", "disable_firewall"
}

# Actions that can be auto-approved after first confirmation (low-risk)
AUTO_APPROVABLE_ACTIONS = {
    "open_app", "launch_app", "web_search", "web_navigate",
    "web_scrape", "file_read", "file_list", "generate_report",
    "screenshot", "system_info", "clipboard_read"
}


@dataclass
class SmartConfirmationDecision:
    requires_confirmation: bool
    auto_approved: bool
    reason: str
    fingerprint: str
    trust_score: float = 1.0


class SmartConfirmationEngine:
    """
    Dual-mode confirmation system.
    Records action fingerprints after first approval,
    then auto-executes on future matching requests.
    """

    def __init__(self, db_path: str = "db/preferences.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_approvals (
                id INTEGER PRIMARY KEY,
                fingerprint TEXT UNIQUE,
                action TEXT,
                params_summary TEXT,
                approved_count INTEGER DEFAULT 0,
                last_approved REAL,
                last_error TEXT,
                error_count INTEGER DEFAULT 0,
                trust_score REAL DEFAULT 1.0
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def evaluate(self, action: str, params: dict) -> SmartConfirmationDecision:
        """
        Decide whether this action needs confirmation.

        Returns SmartConfirmationDecision with:
        - requires_confirmation: bool
        - auto_approved: bool (True = will run without user input)
        - reason: explanation
        """
        fingerprint = self._fingerprint(action, params)

        # High-risk actions always need confirmation
        if action.lower() in ALWAYS_CONFIRM_ACTIONS:
            return SmartConfirmationDecision(
                requires_confirmation=True,
                auto_approved=False,
                reason=f"'{action}' is a high-risk action — always requires confirmation.",
                fingerprint=fingerprint
            )

        # Check if we have a prior approval
        prior = self._lookup(fingerprint)

        if prior is None:
            # First time — ask for confirmation
            return SmartConfirmationDecision(
                requires_confirmation=True,
                auto_approved=False,
                reason=f"First time executing '{action}' — confirmation required.",
                fingerprint=fingerprint
            )

        approved_count = prior["approved_count"]
        error_count = prior["error_count"]
        trust_score = prior["trust_score"]

        # If there were recent errors, ask again
        if error_count > 0 and error_count >= approved_count:
            return SmartConfirmationDecision(
                requires_confirmation=True,
                auto_approved=False,
                reason=f"Previous errors detected for '{action}' — requesting re-confirmation.",
                fingerprint=fingerprint,
                trust_score=trust_score
            )

        # Trust has degraded below threshold
        if trust_score < 0.4:
            return SmartConfirmationDecision(
                requires_confirmation=True,
                auto_approved=False,
                reason=f"Trust score degraded for '{action}' — confirmation required.",
                fingerprint=fingerprint,
                trust_score=trust_score
            )

        # Previously confirmed — auto-approve
        return SmartConfirmationDecision(
            requires_confirmation=False,
            auto_approved=True,
            reason=f"Auto-approved: '{action}' confirmed {approved_count} time(s) previously.",
            fingerprint=fingerprint,
            trust_score=trust_score
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def record_approval(self, action: str, params: dict, fingerprint: str = None):
        """Record that the user approved this action."""
        fp = fingerprint or self._fingerprint(action, params)
        params_summary = self._params_summary(params)

        self.conn.execute("""
            INSERT INTO auto_approvals (fingerprint, action, params_summary, approved_count, last_approved, trust_score)
            VALUES (?, ?, ?, 1, ?, 1.0)
            ON CONFLICT(fingerprint) DO UPDATE SET
                approved_count = approved_count + 1,
                last_approved = ?,
                trust_score = MIN(1.0, trust_score + 0.1)
        """, (fp, action, params_summary, time.time(), time.time()))
        self.conn.commit()
        logger.info(f"[SmartConfirmation] Recorded approval for '{action}' (fp={fp[:8]}...)")

    def record_error(self, action: str, params: dict, error: str, fingerprint: str = None):
        """Record an execution error — degrades trust."""
        fp = fingerprint or self._fingerprint(action, params)

        self.conn.execute("""
            INSERT INTO auto_approvals (fingerprint, action, params_summary, last_error, error_count, trust_score)
            VALUES (?, ?, ?, ?, 1, 0.5)
            ON CONFLICT(fingerprint) DO UPDATE SET
                error_count = error_count + 1,
                last_error = ?,
                trust_score = MAX(0.0, trust_score - 0.2)
        """, (fp, action, self._params_summary(params), error, error))
        self.conn.commit()
        logger.warning(f"[SmartConfirmation] Recorded error for '{action}': {error}")

    def reset_trust(self, action: str, params: dict):
        """Reset trust for an action (e.g., after user-initiated reset)."""
        fp = self._fingerprint(action, params)
        self.conn.execute("DELETE FROM auto_approvals WHERE fingerprint = ?", (fp,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fingerprint(self, action: str, params: dict) -> str:
        """
        Create a stable fingerprint from action + key params.
        Ignores volatile fields like timestamps, session IDs.
        """
        # Keep only semantically meaningful params
        clean_params = {
            k: v for k, v in params.items()
            if k not in {"_extracted_at", "timestamp", "session_id", "meta"}
            and not k.startswith("_")
        }

        payload = json.dumps(
            {"action": action.lower(), "params": clean_params},
            sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _params_summary(self, params: dict) -> str:
        """Human-readable summary of params for display."""
        clean = {k: v for k, v in params.items() if not k.startswith("_")}
        return json.dumps(clean, default=str)[:200]

    def _lookup(self, fingerprint: str) -> Optional[dict]:
        """Fetch prior approval record."""
        cursor = self.conn.execute("""
            SELECT action, approved_count, error_count, trust_score, last_error
            FROM auto_approvals WHERE fingerprint = ?
        """, (fingerprint,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "action": row[0],
            "approved_count": row[1],
            "error_count": row[2],
            "trust_score": row[3],
            "last_error": row[4]
        }

    def get_all_approvals(self) -> list:
        """Return all stored auto-approval records (for debugging/UI)."""
        cursor = self.conn.execute("""
            SELECT action, params_summary, approved_count, error_count, trust_score, last_approved
            FROM auto_approvals ORDER BY last_approved DESC
        """)
        return [
            {
                "action": r[0],
                "params": r[1],
                "approved": r[2],
                "errors": r[3],
                "trust": r[4],
                "last_used": r[5]
            }
            for r in cursor.fetchall()
        ]
