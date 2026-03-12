from services.screen_capture import ScreenCapture
from services.ocr_engine import OCREngine
from services.visual_models import VisualState, WindowInfo
from services.ui_grounding_engine import UIGroundingEngine
import pygetwindow as gw


class VisionEngine:
    """
    Produces a structured snapshot of the visual environment.
    Vision is READ-ONLY.
    """

    def __init__(self):
        self.ocr = OCREngine()
        self.ui_grounder = UIGroundingEngine()

    def perceive(self) -> VisualState:
        # Screen
        image = ScreenCapture.capture()
        resolution = ScreenCapture.get_resolution()

        # Windows
        windows = []
        active_title = ""

        for w in gw.getAllWindows():
            if not w.title:
                continue

            info = WindowInfo(
                title=w.title,
                bbox=(w.left, w.top, w.width, w.height),
                active=w.isActive
            )

            if w.isActive:
                active_title = w.title

            windows.append(info)

        # OCR
        text_blocks = self.ocr.extract(image)

        return VisualState(
            resolution=resolution,
            active_window=active_title,
            windows=windows,
            text_blocks=text_blocks
        )

    def perceive_with_ui(self):
        """
        Extended perception including grounded UI elements.
        """
        visual_state = self.perceive()
        ui_elements = self.ui_grounder.ground(visual_state)
        return visual_state, ui_elements
