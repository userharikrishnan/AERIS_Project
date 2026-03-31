"""
AERIS Process Tool
List, inspect, and terminate running processes.
Uses psutil for cross-platform process management.

Capabilities:
- list_processes: Show all running processes
- process_info: Get details about a specific process
- kill_process: Terminate a process by name or PID
- is_running: Check if a process is running
"""

import time
import logging
from services.tool_base import Tool, ToolResult

logger = logging.getLogger(__name__)

# Processes that AERIS will never kill (protected system processes)
PROTECTED_PROCESSES = {
    "system", "smss.exe", "csrss.exe", "wininit.exe", "services.exe",
    "lsass.exe", "svchost.exe", "winlogon.exe", "explorer.exe",
    "dwm.exe", "audiodg.exe", "pythonw.exe", "python.exe",
    "aeris", "uvicorn", "cmd.exe", "powershell.exe"
}


class ProcessTool(Tool):
    name = "process"
    description = "List, inspect, and manage running processes"
    requires_confirmation = True  # Killing processes always needs confirmation
    capabilities = ["list_processes", "kill_process", "process_info", "is_running"]

    def execute(self, params: dict) -> ToolResult:
        start_time = time.time()
        action = params.get("action", "list").lower()

        try:
            import psutil
        except ImportError:
            return ToolResult(
                success=False,
                error="psutil not installed. Run: pip install psutil",
                meta={"tool": self.name}
            )

        dispatch = {
            "list": self._list_processes,
            "list_processes": self._list_processes,
            "info": self._process_info,
            "process_info": self._process_info,
            "kill": self._kill_process,
            "kill_process": self._kill_process,
            "is_running": self._is_running,
            "check": self._is_running,
        }

        handler = dispatch.get(action, self._list_processes)
        return handler(params, start_time)

    def _list_processes(self, params: dict, start_time: float) -> ToolResult:
        import psutil
        filter_name = params.get("name", "").lower()
        limit = params.get("limit", 30)

        processes = []
        for proc in psutil.process_iter(["pid", "name", "status", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                if filter_name and filter_name not in info["name"].lower():
                    continue
                processes.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "status": info["status"],
                    "cpu_percent": round(info["cpu_percent"] or 0, 2),
                    "memory_percent": round(info["memory_percent"] or 0, 2),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by memory usage
        processes.sort(key=lambda x: x["memory_percent"], reverse=True)

        return ToolResult(
            success=True,
            data={
                "processes": processes[:limit],
                "total_count": len(processes),
                "filter": filter_name or "none"
            },
            meta={"tool": self.name, "execution_time": time.time() - start_time}
        )

    def _process_info(self, params: dict, start_time: float) -> ToolResult:
        import psutil
        name = params.get("name", "")
        pid = params.get("pid")

        try:
            if pid:
                proc = psutil.Process(int(pid))
            elif name:
                matching = [p for p in psutil.process_iter(["pid", "name"]) if name.lower() in p.info["name"].lower()]
                if not matching:
                    return ToolResult(success=False, error=f"No process found matching: '{name}'", meta={"tool": self.name})
                proc = psutil.Process(matching[0].info["pid"])
            else:
                return ToolResult(success=False, error="Provide 'name' or 'pid'", meta={"tool": self.name})

            info = {
                "pid": proc.pid,
                "name": proc.name(),
                "status": proc.status(),
                "cpu_percent": proc.cpu_percent(),
                "memory_mb": round(proc.memory_info().rss / (1024 ** 2), 2),
                "create_time": proc.create_time(),
                "exe": proc.exe() if hasattr(proc, "exe") else None,
            }
            return ToolResult(
                success=True,
                data=info,
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), meta={"tool": self.name})

    def _kill_process(self, params: dict, start_time: float) -> ToolResult:
        import psutil
        name = params.get("name", "").lower()
        pid = params.get("pid")

        # Safety check
        if name in PROTECTED_PROCESSES:
            return ToolResult(
                success=False,
                error=f"Cannot kill protected system process: '{name}'",
                meta={"tool": self.name, "protected": True}
            )

        try:
            killed = []
            if pid:
                proc = psutil.Process(int(pid))
                pname = proc.name()
                if pname.lower() in PROTECTED_PROCESSES:
                    return ToolResult(success=False, error=f"Protected process: {pname}", meta={"tool": self.name})
                proc.terminate()
                killed.append({"pid": pid, "name": pname})
            elif name:
                for proc in psutil.process_iter(["pid", "name"]):
                    if name in proc.info["name"].lower() and proc.info["name"].lower() not in PROTECTED_PROCESSES:
                        proc.terminate()
                        killed.append({"pid": proc.info["pid"], "name": proc.info["name"]})

            if not killed:
                return ToolResult(success=False, error=f"No process found matching: '{name}'", meta={"tool": self.name})

            return ToolResult(
                success=True,
                data={"terminated": killed},
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), meta={"tool": self.name})

    def _is_running(self, params: dict, start_time: float) -> ToolResult:
        import psutil
        name = params.get("name", "").lower()
        running = any(name in p.info["name"].lower() for p in psutil.process_iter(["name"]))
        return ToolResult(
            success=True,
            data={"name": name, "running": running},
            meta={"tool": self.name, "execution_time": time.time() - start_time}
        )
