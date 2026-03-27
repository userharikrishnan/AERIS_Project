import os
import time
from services.tool_base import Tool, ToolResult

class FileSystemTool(Tool):
    name = "filesystem"
    description = "Read, write, list, and delete files and directories"
    requires_confirmation = False
    capabilities = ["file_read", "file_list", "file_write", "file_delete"]

    def execute(self, params: dict) -> ToolResult:
        start_time = time.time()
        action = params.get("action", "list")
        path = params.get("path", ".")

        try:
            # LIST action
            if action == "list":
                items = os.listdir(path)
                execution_time = time.time() - start_time
                return ToolResult(
                    success=True,
                    data={"path": path, "items": items},
                    meta={"execution_time": execution_time, "tool": self.name}
                )

            # READ action
            elif action == "read":
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                execution_time = time.time() - start_time
                return ToolResult(
                    success=True,
                    data={"path": path, "content": content},
                    meta={"execution_time": execution_time, "tool": self.name}
                )

            # WRITE action
            elif action == "write":
                content = params.get("content", "")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                execution_time = time.time() - start_time
                return ToolResult(
                    success=True,
                    data={"written": path, "bytes": len(content)},
                    meta={"execution_time": execution_time, "tool": self.name}
                )

            # DELETE action
            elif action == "delete":
                if os.path.isdir(path):
                    os.rmdir(path)
                else:
                    os.remove(path)
                execution_time = time.time() - start_time
                return ToolResult(
                    success=True,
                    data={"deleted": path},
                    meta={"execution_time": execution_time, "tool": self.name}
                )

            # Unknown action
            else:
                execution_time = time.time() - start_time
                return ToolResult(
                    success=False,
                    error=f"Unknown filesystem action: '{action}'. Available: list, read, write, delete",
                    meta={"execution_time": execution_time, "tool": self.name}
                )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error=str(e),
                meta={"execution_time": execution_time, "tool": self.name}
            )