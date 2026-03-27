from abc import ABC, abstractmethod

class ToolResult:
    def __init__(self, success: bool, data=None, error=None):
        self.success = success
        self.data = data
        self.error = error


class Tool(ABC):
    """
    Base class for all tools.
    Tools NEVER reason. They only execute validated actions.
    """

    name: str = ""
    description: str = ""
    requires_confirmation: bool = False

    @abstractmethod
    def execute(self, params: dict) -> ToolResult:
        pass
