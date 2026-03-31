"""
AERIS Session Engine
Manages the lifecycle of user interaction sessions.

Session model:
- A session begins when the user sends any input
- All actions within the session share a session_id for memory scoping
- A session ends when the user's intent signals completion
  (thanks, done, bye, that's all, etc.) — detected by CHAT intent + closing keywords
- Session history is preserved in memory for context
"""

import uuid
import time
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Patterns that signal the user wants to close the session
SESSION_CLOSE_PATTERNS = [
    r"\b(thanks|thank you|thankyou|thx)\b",
    r"\b(bye|goodbye|good bye|see you|cya|later|take care)\b",
    r"\b(that'?s all|that is all|we('?re)? done|i('?m)? done|all done)\b",
    r"\b(close session|end session|stop|quit|exit)\b",
    r"\b(great|perfect|awesome|excellent|good job|well done)\b.*\b(done|finished|complete)\b",
    r"\b(no more|nothing else|that'?s it|i('?m)? good)\b",
]

COMPILED_CLOSE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SESSION_CLOSE_PATTERNS]


@dataclass
class Session:
    session_id: str
    started_at: float
    ended_at: Optional[float] = None
    is_active: bool = True
    turn_count: int = 0
    context: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def add_turn(self, user_input: str, intent: str, response: str, action: Optional[str] = None):
        """Record a completed interaction turn."""
        self.turn_count += 1
        self.history.append({
            "turn": self.turn_count,
            "user_input": user_input,
            "intent": intent,
            "response": response,
            "action": action,
            "timestamp": time.time()
        })

    def close(self):
        """Mark session as ended."""
        self.is_active = False
        self.ended_at = time.time()
        logger.info(f"[Session] Session {self.session_id[:8]}... closed after {self.turn_count} turns.")

    @property
    def duration(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at

    def to_context_dict(self) -> dict:
        """Return session context for use in NLP/reasoning."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "recent_history": self.history[-5:],  # Last 5 turns
            "context": self.context
        }


class SessionEngine:
    """
    Manages active AERIS sessions.
    Provides start, update, close, and context retrieval.
    """

    def __init__(self):
        self._active_session: Optional[Session] = None
        self._session_history: List[Session] = []

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def get_or_create_session(self) -> Session:
        """
        Return the current active session, or create a new one.
        Called at the start of every user input.
        """
        if self._active_session and self._active_session.is_active:
            return self._active_session

        return self.start_new_session()

    def start_new_session(self) -> Session:
        """Explicitly start a new session."""
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            started_at=time.time()
        )
        self._active_session = session
        logger.info(f"[SessionEngine] New session started: {session_id[:8]}...")
        return session

    def close_current_session(self) -> Optional[Session]:
        """Close the current active session."""
        if self._active_session and self._active_session.is_active:
            self._active_session.close()
            self._session_history.append(self._active_session)
            closed = self._active_session
            self._active_session = None
            return closed
        return None

    def record_turn(
        self,
        user_input: str,
        intent: str,
        response: str,
        action: Optional[str] = None
    ):
        """Record a completed turn in the current session."""
        session = self.get_or_create_session()
        session.add_turn(user_input, intent, response, action)

    # ------------------------------------------------------------------
    # Session close detection
    # ------------------------------------------------------------------

    def detect_close_intent(self, text: str, intent_type: str) -> bool:
        """
        Determine if the user wants to close the session.
        Uses both pattern matching and intent type heuristics.
        """
        text_lower = text.lower().strip()

        # Check closing patterns
        for pattern in COMPILED_CLOSE_PATTERNS:
            if pattern.search(text_lower):
                logger.info(f"[SessionEngine] Close pattern matched: '{text[:50]}'")
                return True

        # Short "thanks" or "done" with CHAT intent
        if intent_type == "CHAT" and len(text_lower.split()) <= 4:
            short_closers = {"thanks", "thank you", "thx", "done", "bye", "ok thanks", "cool thanks"}
            if text_lower in short_closers:
                return True

        return False

    # ------------------------------------------------------------------
    # Context / state
    # ------------------------------------------------------------------

    def get_session_context(self) -> dict:
        """Get current session context for NLP/reasoning enrichment."""
        if not self._active_session:
            return {"session_id": None, "turn_count": 0, "recent_history": []}
        return self._active_session.to_context_dict()

    def update_context(self, key: str, value):
        """Store arbitrary context within the session."""
        session = self.get_or_create_session()
        session.context[key] = value

    @property
    def current_session_id(self) -> Optional[str]:
        if self._active_session and self._active_session.is_active:
            return self._active_session.session_id
        return None

    @property
    def has_active_session(self) -> bool:
        return self._active_session is not None and self._active_session.is_active

    def get_session_summary(self) -> dict:
        """Return summary of current session for status endpoint."""
        if not self._active_session:
            return {"active": False}
        s = self._active_session
        return {
            "active": s.is_active,
            "session_id": s.session_id[:8] + "...",
            "turns": s.turn_count,
            "duration_seconds": round(s.duration, 1),
            "last_intent": s.history[-1]["intent"] if s.history else None
        }
