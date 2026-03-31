"""
AERIS App Tool
Launches and closes desktop applications.
Uses the dynamic AppRegistry instead of a hardcoded app map.
The registry discovers apps from the Windows filesystem and registry,
and learns from user corrections.
"""

import subprocess
import time
import logging
from services.tool_base import Tool, ToolResult

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency issues at startup
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        try:
            from services.app_registry import AppRegistry
            _registry = AppRegistry()
        except Exception as e:
            logger.warning(f"[AppTool] AppRegistry failed to load: {e}")
    return _registry


class AppTool(Tool):
    name = "app"
    description = "Launch and close desktop applications dynamically"
    requires_confirmation = False
    capabilities = ["open_app", "close_app", "launch_app", "launch", "open"]

    def execute(self, params: dict) -> ToolResult:
        start_time = time.time()
        action = params.get("action", "open").lower()
        app = params.get("app", "").strip()

        if not app:
            return ToolResult(
                success=False,
                error="Missing 'app' parameter. Please specify which app to open.",
                meta={"tool": self.name, "execution_time": 0}
            )

        # Close app action
        if action in ("close", "close_app"):
            return self._close_app(app, start_time)

        # Open app action
        return self._open_app(app, params, start_time)

    def _open_app(self, app: str, params: dict, start_time: float) -> ToolResult:
        registry = _get_registry()

        # Try resolving via registry
        resolved = registry.resolve(app) if registry else None

        if not resolved:
            # Check if user already specified a path
            user_path = params.get("path", "")
            if user_path:
                resolved = user_path
                if registry:
                    registry.teach(app, user_path)
            else:
                # Cannot find app — ask for clarification
                known_apps = []
                if registry:
                    known = registry.list_known_apps()
                    known_apps = [k["name"] for k in known[:10]]

                return ToolResult(
                    success=False,
                    error=f"Cannot find app: '{app}'. Please tell me the path or which app you mean.",
                    data={
                        "needs_clarification": True,
                        "app_name": app,
                        "known_apps": known_apps,
                        "clarification_question": (
                            f"I couldn't find '{app}' on your system. "
                            f"Could you tell me the full path, or which application you mean? "
                            f"Next time, I'll remember it."
                        )
                    },
                    meta={"tool": self.name, "execution_time": time.time() - start_time}
                )

        try:
            # Handle special cases (ms-settings: etc.)
            if resolved.startswith("ms-"):
                import os
                os.startfile(resolved)
            else:
                subprocess.Popen([resolved], shell=False)

            # Record usage
            if registry:
                registry._record_usage(app.lower())

            execution_time = time.time() - start_time
            return ToolResult(
                success=True,
                data={
                    "launched": app,
                    "resolved_path": resolved,
                    "status": "started"
                },
                meta={"tool": self.name, "execution_time": execution_time}
            )

        except FileNotFoundError:
            # Path resolved but exe not found — try shell launch
            try:
                import os
                os.startfile(resolved)
                return ToolResult(
                    success=True,
                    data={"launched": app, "status": "started_via_shell"},
                    meta={"tool": self.name, "execution_time": time.time() - start_time}
                )
            except Exception as e2:
                return ToolResult(
                    success=False,
                    error=f"Failed to launch '{app}' from '{resolved}': {e2}",
                    meta={"tool": self.name, "execution_time": time.time() - start_time}
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to launch '{app}': {str(e)}",
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )

    def _close_app(self, app: str, start_time: float) -> ToolResult:
        try:
            import subprocess
            # taskkill by window title or image name
            result = subprocess.run(
                ["taskkill", "/IM", f"{app}.exe", "/F"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    data={"closed": app},
                    meta={"tool": self.name, "execution_time": time.time() - start_time}
                )
            # Try with process name without .exe
            result2 = subprocess.run(
                ["taskkill", "/FI", f"WINDOWTITLE eq {app}*", "/F"],
                capture_output=True, text=True, timeout=5
            )
            return ToolResult(
                success=result2.returncode == 0,
                data={"closed": app} if result2.returncode == 0 else {},
                error=result.stderr.strip() if result2.returncode != 0 else None,
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to close '{app}': {str(e)}",
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )