import webbrowser
from services.tool_base import Tool, ToolResult

class BrowserTool(Tool):
    name = "browser"
    description = "Open URLs or perform simple searches"
    requires_confirmation = False

    def execute(self, params: dict) -> ToolResult:
        url = params.get("url")
        if not url:
            return ToolResult(False, error="Missing 'url' parameter")

        try:
            webbrowser.open(url)
            return ToolResult(True, data={"opened": url})
        except Exception as e:
            return ToolResult(False, error=str(e))
