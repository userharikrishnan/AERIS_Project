from services.tool_registry import ToolRegistry
from services.tools.browser_tool import BrowserTool
from services.tools.filesystem_tool import FileSystemTool
from services.tools.app_tool import AppTool

class ActionRouter:
    """
    Routes validated intents to registered tools.
    No execution happens here.
    """

    def __init__(self):
        self.registry = ToolRegistry()
        self._register_default_tools()

    def _register_default_tools(self):
        self.registry.register(BrowserTool())
        self.registry.register(FileSystemTool())
        self.registry.register(AppTool())

    def route(self, tool_name: str, params: dict, context: dict = None):
        tool = self.registry.get(tool_name)

        # Fallback: try capability match
        if not tool:
            tool = self.registry.find_by_capability(tool_name)

        if not tool:
            return None

        # Context-aware routing: handle past failures
        if context:
            if context.get("last_failed_tool") == tool.name:
                alt_tool = self.registry.find_alternative(tool_name, exclude=tool.name)
                if alt_tool:
                    tool = alt_tool

        return {
            "tool": tool.name,
            "action": tool_name,
            "params": params,
            "requires_confirmation": tool.requires_confirmation,
            "capability": getattr(tool, "capability", tool_name)
        }