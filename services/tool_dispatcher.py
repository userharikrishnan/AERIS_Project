"""
AERIS Tool Dispatcher
The critical "last-mile" execution layer.
Maps action names to registered tool instances and executes them.
All tool calls pass through here after confirmation and delegation.
"""

import logging
from typing import Dict, Optional
from services.tool_base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolDispatcher:
    """
    Central registry and executor for all AERIS tools.
    Tools are registered by capability name and dispatched on demand.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._action_map: Dict[str, str] = {}  # action_name -> tool_name
        self._register_builtin_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool):
        """Register a tool instance by its name."""
        self._tools[tool.name] = tool
        for cap in tool.capabilities:
            self._action_map[cap.lower()] = tool.name
        logger.info(f"[ToolDispatcher] Registered tool: {tool.name} → capabilities: {tool.capabilities}")

    def _register_builtin_tools(self):
        """Auto-register all built-in tools."""
        try:
            from services.tools.app_tool import AppTool
            self.register(AppTool())
        except Exception as e:
            logger.warning(f"[ToolDispatcher] AppTool load failed: {e}")

        try:
            from services.tools.browser_tool import BrowserTool
            self.register(BrowserTool())
        except Exception as e:
            logger.warning(f"[ToolDispatcher] BrowserTool load failed: {e}")

        try:
            from services.tools.filesystem_tool import FileSystemTool
            self.register(FileSystemTool())
        except Exception as e:
            logger.warning(f"[ToolDispatcher] FileSystemTool load failed: {e}")

        try:
            from services.tools.web_scraper_tool import WebScraperTool
            self.register(WebScraperTool())
        except Exception as e:
            logger.warning(f"[ToolDispatcher] WebScraperTool load failed: {e}")

        try:
            from services.tools.report_generator_tool import ReportGeneratorTool
            self.register(ReportGeneratorTool())
        except Exception as e:
            logger.warning(f"[ToolDispatcher] ReportGeneratorTool load failed: {e}")

        try:
            from services.tools.system_tool import SystemTool
            self.register(SystemTool())
        except Exception as e:
            logger.warning(f"[ToolDispatcher] SystemTool load failed: {e}")

        try:
            from services.tools.process_tool import ProcessTool
            self.register(ProcessTool())
        except Exception as e:
            logger.warning(f"[ToolDispatcher] ProcessTool load failed: {e}")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, action: str, params: dict) -> ToolResult:
        """
        Resolve action name to tool and execute it.
        Falls back gracefully with a ToolResult(success=False) if not found.
        """
        action_lower = action.lower().replace(" ", "_")

        # Direct tool name lookup first
        tool = self._tools.get(action_lower)

        # Then try capability mapping
        if not tool:
            tool_name = self._action_map.get(action_lower)
            if tool_name:
                tool = self._tools.get(tool_name)

        # Fuzzy match for common alias patterns
        if not tool:
            tool = self._fuzzy_resolve(action_lower)

        if not tool:
            logger.warning(f"[ToolDispatcher] No tool found for action: '{action}'")
            return ToolResult(
                success=False,
                error=f"No tool registered for action: '{action}'. Available: {list(self._action_map.keys())}",
                meta={"action": action, "dispatcher": "tool_not_found"}
            )

        logger.info(f"[ToolDispatcher] Dispatching action='{action}' → tool='{tool.name}' params={params}")

        try:
            result = tool.execute(params)
            logger.info(f"[ToolDispatcher] Result: success={result.success}")
            return result
        except Exception as e:
            logger.error(f"[ToolDispatcher] Tool execution error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                meta={"action": action, "tool": tool.name}
            )

    def _fuzzy_resolve(self, action: str) -> Optional[Tool]:
        """Resolve common action name aliases to tools."""
        aliases = {
            # App actions
            "open_app": "app",
            "close_app": "app",
            "launch_app": "app",
            "launch": "app",
            "open": "app",
            # Web actions
            "web_search": "browser",
            "web_navigate": "browser",
            "search": "browser",
            "navigate": "browser",
            "browse": "browser",
            "web_scrape": "web_scraper",
            "scrape": "web_scraper",
            # File actions
            "file_read": "filesystem",
            "file_write": "filesystem",
            "file_delete": "filesystem",
            "file_list": "filesystem",
            "read_file": "filesystem",
            "write_file": "filesystem",
            # Report
            "generate_report": "report_generator",
            "report": "report_generator",
            # System
            "screenshot": "system",
            "screen_capture": "system",
            "clipboard_read": "system",
            "clipboard_write": "system",
            "system_info": "system",
            # Process
            "list_processes": "process",
            "kill_process": "process",
            "process_info": "process",
        }
        tool_name = aliases.get(action)
        return self._tools.get(tool_name) if tool_name else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def list_capabilities(self) -> dict:
        """Return a map of all available capabilities."""
        return {
            "tools": list(self._tools.keys()),
            "action_map": self._action_map
        }

    def has_capability(self, action: str) -> bool:
        """Check if a given action is dispatchable."""
        action_lower = action.lower().replace(" ", "_")
        return (
            action_lower in self._tools
            or action_lower in self._action_map
            or self._fuzzy_resolve(action_lower) is not None
        )
