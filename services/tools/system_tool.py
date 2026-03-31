"""
AERIS System Tool
Handles OS-level actions: screenshots, clipboard, system info, volume, brightness.

Capabilities:
- screenshot: Capture the current screen
- clipboard_read: Read clipboard content
- clipboard_write: Write to clipboard
- system_info: Get OS/hardware info (RAM, CPU, disk)
- get_volume / set_volume: Audio control
- lock_screen: Lock Windows session
- open_url_in_browser: Cross-platform browser launch

Uses:
- pyautogui (already in requirements) — screenshot, automation
- psutil — system info
- pyperclip — clipboard (pip install pyperclip)
- ctypes (built-in) — Windows API for lock/volume
"""

import os
import time
import logging
import platform
from services.tool_base import Tool, ToolResult

logger = logging.getLogger(__name__)


class SystemTool(Tool):
    name = "system"
    description = "OS-level actions: screenshots, clipboard, system info, screen lock"
    requires_confirmation = False
    capabilities = [
        "screenshot", "screen_capture",
        "clipboard_read", "clipboard_write",
        "system_info", "get_system_info",
        "lock_screen", "set_volume",
    ]

    SCREENSHOT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "AERIS_Screenshots")

    def execute(self, params: dict) -> ToolResult:
        start_time = time.time()
        action = params.get("action", "screenshot").lower()

        dispatch = {
            "screenshot": self._screenshot,
            "screen_capture": self._screenshot,
            "clipboard_read": self._clipboard_read,
            "clipboard_write": self._clipboard_write,
            "system_info": self._system_info,
            "get_system_info": self._system_info,
            "lock_screen": self._lock_screen,
            "set_volume": self._set_volume,
        }

        handler = dispatch.get(action)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown system action: '{action}'. Available: {list(dispatch.keys())}",
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )

        return handler(params, start_time)

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def _screenshot(self, params: dict, start_time: float) -> ToolResult:
        try:
            import pyautogui
            from datetime import datetime

            os.makedirs(self.SCREENSHOT_DIR, exist_ok=True)
            save_path = params.get("path", "")
            if not save_path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.SCREENSHOT_DIR, f"screenshot_{ts}.png")

            screenshot = pyautogui.screenshot()
            screenshot.save(save_path)

            return ToolResult(
                success=True,
                data={"saved_to": save_path, "resolution": screenshot.size},
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Screenshot failed: {e}",
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def _clipboard_read(self, params: dict, start_time: float) -> ToolResult:
        try:
            import pyperclip
            content = pyperclip.paste()
            return ToolResult(
                success=True,
                data={"content": content, "length": len(content)},
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except ImportError:
            # Fallback using subprocess on Windows
            try:
                import subprocess
                result = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=5
                )
                return ToolResult(
                    success=True,
                    data={"content": result.stdout.strip()},
                    meta={"tool": self.name, "execution_time": time.time() - start_time}
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Clipboard read failed: {e}. Install pyperclip: pip install pyperclip",
                    meta={"tool": self.name}
                )

    def _clipboard_write(self, params: dict, start_time: float) -> ToolResult:
        content = params.get("content", "")
        try:
            import pyperclip
            pyperclip.copy(content)
            return ToolResult(
                success=True,
                data={"written": len(content), "preview": content[:100]},
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except ImportError:
            try:
                import subprocess
                subprocess.run(
                    ["powershell", "-command", f"Set-Clipboard -Value '{content}'"],
                    timeout=5
                )
                return ToolResult(
                    success=True,
                    data={"written": len(content)},
                    meta={"tool": self.name, "execution_time": time.time() - start_time}
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Clipboard write failed: {e}",
                    meta={"tool": self.name}
                )

    # ------------------------------------------------------------------
    # System Info
    # ------------------------------------------------------------------

    def _system_info(self, params: dict, start_time: float) -> ToolResult:
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }

        try:
            import psutil
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_usage_percent"] = psutil.cpu_percent(interval=0.5)

            mem = psutil.virtual_memory()
            info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
            info["ram_used_gb"] = round(mem.used / (1024 ** 3), 2)
            info["ram_percent"] = mem.percent

            disk = psutil.disk_usage("C:\\")
            info["disk_total_gb"] = round(disk.total / (1024 ** 3), 2)
            info["disk_used_gb"] = round(disk.used / (1024 ** 3), 2)
            info["disk_percent"] = disk.percent

            info["battery"] = None
            if psutil.sensors_battery():
                bat = psutil.sensors_battery()
                info["battery"] = {"percent": bat.percent, "plugged": bat.power_plugged}

        except ImportError:
            info["note"] = "Install psutil for detailed stats: pip install psutil"
        except Exception as e:
            info["psutil_error"] = str(e)

        return ToolResult(
            success=True,
            data=info,
            meta={"tool": self.name, "execution_time": time.time() - start_time}
        )

    # ------------------------------------------------------------------
    # Lock screen
    # ------------------------------------------------------------------

    def _lock_screen(self, params: dict, start_time: float) -> ToolResult:
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            return ToolResult(
                success=True,
                data={"status": "Screen locked"},
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Lock screen failed: {e}",
                meta={"tool": self.name}
            )

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def _set_volume(self, params: dict, start_time: float) -> ToolResult:
        level = params.get("level", 50)
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            # Volume range: 0.0 to 1.0
            volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level / 100.0)), None)

            return ToolResult(
                success=True,
                data={"volume_set_to": level},
                meta={"tool": self.name, "execution_time": time.time() - start_time}
            )
        except ImportError:
            # Fallback: use PowerShell
            try:
                import subprocess
                script = f"(New-Object -ComObject WScript.Shell).SendKeys([char]173)"
                subprocess.run(["powershell", "-command", script], timeout=3)
                return ToolResult(
                    success=True,
                    data={"note": "Volume adjust attempted via PowerShell"},
                    meta={"tool": self.name}
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Volume control failed: {e}. Install pycaw: pip install pycaw",
                    meta={"tool": self.name}
                )
