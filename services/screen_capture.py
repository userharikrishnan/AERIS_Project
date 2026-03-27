import pyautogui
import ctypes
from typing import Tuple


class ScreenCapture:
    """
    Responsible ONLY for screen capture.
    """

    @staticmethod
    def get_resolution() -> Tuple[int, int]:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    @staticmethod
    def capture():
        """
        Returns a PIL Image of the current screen.
        """
        return pyautogui.screenshot()
