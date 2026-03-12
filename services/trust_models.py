from enum import Enum

class TrustLevel(Enum):
    UNTRUSTED = 0
    LIMITED = 1
    STANDARD = 2
    ADMIN = 3


class ActionSensitivity(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
