import subprocess
import time
from services.tool_base import Tool, ToolResult

class AppTool(Tool):
    name = "app"
    description = "Launch desktop applications"
    requires_confirmation = False
    capabilities = ["open_app", "close_app"]

    # Safe application mapping to prevent shell injection
    APP_MAP = {
        "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "explorer": "explorer.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe"
    }

    def execute(self, params: dict) -> ToolResult:
        start_time = time.time()
        app = params.get("app")

        if not app:
            return ToolResult(
                success=False,
                error="Missing 'app' parameter",
                meta={"execution_time": 0, "tool": self.name}
            )

        # Resolve app name to safe executable path
        resolved = self.APP_MAP.get(app.lower())

        if not resolved:
            return ToolResult(
                success=False,
                error=f"Unknown app: {app}. Available apps: {list(self.APP_MAP.keys())}",
                meta={"execution_time": time.time() - start_time, "tool": self.name}
            )

        try:
            # Use list format to avoid shell injection
            subprocess.Popen([resolved])
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data={
                    "launched": app,
                    "path": resolved,
                    "status": "started"
                },
                meta={
                    "execution_time": execution_time,
                    "tool": self.name
                }
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error=str(e),
                meta={"execution_time": execution_time, "tool": self.name}
            )