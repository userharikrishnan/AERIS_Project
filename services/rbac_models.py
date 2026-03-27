from enum import Enum, auto


class Role(Enum):
    USER = auto()
    ADMIN = auto()
    SYSTEM = auto()
    AUTONOMOUS = auto()


class Capability(Enum):
    CREATE_GOAL = auto()
    EXECUTE_ACTION = auto()
    DELEGATE_ACTION = auto()
    MODIFY_SYSTEM = auto()
