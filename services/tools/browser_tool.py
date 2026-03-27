import webbrowser
import time
from services.tool_base import Tool, ToolResult

class BrowserTool(Tool):
    name = "browser"
    description = "Open URLs or perform simple searches"
    requires_confirmation = False
    capabilities = ["web_search", "web_navigate"]

    def execute(self, params: dict) -> ToolResult:
        start_time = time.time()
        url = params.get("url")
        query = params.get("query")

        # Handle search fallback
        if not url and not query:
            return ToolResult(
                success=False,
                error="Missing 'url' or 'query' parameter",
                meta={"execution_time": 0, "tool": self.name}
            )

        # If no URL but query exists, construct search URL
        if not url and query:
            url = f"https://www.google.com/search?q={query}"

        # Normalize URL by adding https:// if missing
        if url and not url.startswith("http"):
            url = "https://" + url

        try:
            webbrowser.open(url)
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data={
                    "opened": url,
                    "status": "opened"
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