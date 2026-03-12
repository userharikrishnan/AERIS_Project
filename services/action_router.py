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

    def route(self, tool_name: str, params: dict):
        tool = self.registry.get(tool_name)
        if not tool:
            return None

        return {
            "tool": tool.name,
            "params": params,
            "requires_confirmation": tool.requires_confirmation
        }
