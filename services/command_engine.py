from services.symbol_registry import SymbolRegistry
from services.action_router import ActionRouter
from services.vision_engine import VisionEngine
from services.trust_models import ActionSensitivity

class CommandEngine:
    def __init__(self):
        self.registry = SymbolRegistry()
        self.router = ActionRouter()
        self.vision = VisionEngine()

    def plan(self, reasoning_output, context=None):
        """
        Plans an action symbolically using reasoning output.
        Extracts intent, plan, and confidence from reasoning layer.
        Validates actions, applies confidence gating, and enriches execution context.
        """
        # Extract reasoning components
        intent = reasoning_output.trace.get("intent") if reasoning_output.trace else None
        plan = reasoning_output.plan
        confidence = reasoning_output.confidence

        # Confidence gating - prevent low confidence executions
        if confidence < 0.4:
            return {
                "action": "clarify",
                "params": {"reason": "Low confidence"},
                "sensitivity": ActionSensitivity.LOW
            }

        # Respect multi-step planning - execute first step
        if not plan:
            return None

        current_step = plan[0]

        # Route the action based on current step
        action_payload = self.router.route(
            current_step["action"],
            params=current_step.get("params", {})
        )

        # Smart fallback if router fails
        if action_payload is None:
            return {
                "action": "respond",
                "params": {"message": "I couldn't determine an action."},
                "sensitivity": ActionSensitivity.LOW
            }

        # Validate action using SymbolRegistry
        symbol = self.registry.resolve(current_step["action"])
        if not symbol:
            return {
                "action": "respond",
                "params": {"message": "Unsupported action"},
                "sensitivity": ActionSensitivity.LOW
            }
        # Context-aware vision grounding - attach UI candidates for specific actions
        if current_step["action"] in {"click", "open_app", "submit"}:
            visual_state, ui_elements = self.vision.perceive_with_ui()
            if action_payload is not None:
                # Rank UI candidates by confidence and limit to top 5
                ranked_candidates = sorted(
                    [
                        {
                            "type": el.element_type,
                            "label": el.label,
                            "bbox": el.bbox,
                            "window": el.window_title,
                            "confidence": el.confidence
                        }
                        for el in ui_elements
                    ],
                    key=lambda x: x["confidence"],
                    reverse=True
                )[:5]
                
                action_payload["ui_candidates"] = ranked_candidates

        # Attach execution metadata for debugging, learning, and audit
        action_payload["meta"] = {
            "confidence": confidence,
            "trace": reasoning_output.trace
        }

        return {
            "action": current_step["action"],
            "params": action_payload["params"],
            "sensitivity": symbol.sensitivity if symbol else ActionSensitivity.LOW,
            "tool": action_payload.get("tool"),
            "meta": {
                "confidence": confidence,
                "trace": reasoning_output.trace
            }
        }