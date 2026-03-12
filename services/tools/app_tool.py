import subprocess
from services.tool_base import Tool, ToolResult

class AppTool(Tool):
    name = "app"
    description = "Launch desktop applications"
    requires_confirmation = False

    def execute(self, params: dict) -> ToolResult:
        app = params.get("app")
        if not app:
            return ToolResult(False, error="Missing 'app' parameter")

        try:
            subprocess.Popen(app, shell=True)
            return ToolResult(True, data={"launched": app})
        except Exception as e:
            return ToolResult(False, error=str(e))
