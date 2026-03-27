from typing import List
from services.ui_models import UIElement
from services.visual_models import VisualState, TextBlock


class UIGroundingEngine:
    """
    Converts visual perception into candidate UI elements.
    READ-ONLY. NO EXECUTION.
    """

    BUTTON_KEYWORDS = {"ok", "submit", "save", "cancel", "next", "yes", "no", "apply"}
    LINK_KEYWORDS = {"click", "open", "learn", "more", "details"}

    def ground(self, visual_state: VisualState) -> List[UIElement]:
        elements: List[UIElement] = []

        active_window = visual_state.active_window

        for block in visual_state.text_blocks:
            label = block.text.strip()
            if not label:
                continue

            label_l = label.lower()

            element_type = None
            confidence = 0.4

            if label_l in self.BUTTON_KEYWORDS:
                element_type = "button"
                confidence = 0.75
            elif any(k in label_l for k in self.LINK_KEYWORDS):
                element_type = "link"
                confidence = 0.6
            elif len(label_l) < 20:
                element_type = "field"
                confidence = 0.45

            if element_type:
                elements.append(
                    UIElement(
                        element_type=element_type,
                        label=label,
                        bbox=block.bbox,
                        window_title=active_window,
                        confidence=confidence
                    )
                )

        return elements
