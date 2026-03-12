import os
from services.tool_base import Tool, ToolResult

class FileSystemTool(Tool):
    name = "filesystem"
    description = "Read directory listings"
    requires_confirmation = False

    def execute(self, params: dict) -> ToolResult:
        path = params.get("path", ".")
        try:
            items = os.listdir(path)
            return ToolResult(True, data={"path": path, "items": items})
        except Exception as e:
            return ToolResult(False, error=str(e))
