from services.symbol_registry import SymbolRegistry
from services.action_router import ActionRouter
from services.vision_engine import VisionEngine


class CommandEngine:
    def __init__(self):
        self.registry = SymbolRegistry()
        self.router = ActionRouter(self.registry)
        self.vision = VisionEngine()

    def plan(self, intent):
        """
        Plans an action symbolically.
        If vision is relevant, attach UI candidates (non-executable).
        """
        action_payload = self.router.route(intent, plan=None)

        # Attach grounded UI candidates ONLY for relevant intents
        if intent.type in {
            "OPEN_APP",
            "CLICK",
            "SUBMIT",
            "VISION_QUERY"
        }:
            visual_state, ui_elements = self.vision.perceive_with_ui()
            if action_payload is not None:
                action_payload["ui_candidates"] = [
                    {
                        "type": el.element_type,
                        "label": el.label,
                        "bbox": el.bbox,
                        "window": el.window_title,
                        "confidence": el.confidence
                    }
                    for el in ui_elements
                ]

        return action_payload
