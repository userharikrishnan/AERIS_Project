from enum import Enum
from datetime import datetime


class AuditEventType(Enum):
    USER_INPUT = "user_input"
    GOAL_CREATED = "goal_created"
    ACTION_PLANNED = "action_planned"
    ACTION_DELEGATED = "action_delegated"
    ACTION_BLOCKED = "action_blocked"
    AUTONOMOUS_ACTION = "autonomous_action"
    SYSTEM_ERROR = "system_error"


class AuditEvent:
    def __init__(
        self,
        event_type: AuditEventType,
        actor: str,
        details: dict
    ):
        self.timestamp = datetime.utcnow().isoformat()
        self.event_type = event_type
        self.actor = actor  # "user", "goal_executor", "system"
        self.details = details

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "actor": self.actor,
            "details": self.details
        }
