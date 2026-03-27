from services.self_model import SelfModel
from services.signal_bus import SignalBus


class ReflectionDecision:
    def __init__(self, retrain=False, lower_confidence=False, note=None):
        self.retrain = retrain
        self.lower_confidence = lower_confidence
        self.note = note


class ReflectionEngine:
    """
    Meta-cognition engine.
    Evaluates past reasoning + outcomes and updates Aeris self-model.
    NEVER executes actions.
    NEVER bypasses permissions.
    """

    def __init__(self, self_model: SelfModel, signal_bus: SignalBus):
        self.self_model = self_model
        self.signal_bus = signal_bus

    def reflect(self, action: str, confidence: float, success: bool, context: dict = None) -> ReflectionDecision:
        # Enrich context
        context = context or {}
        intent = context.get("intent")
        plan = context.get("plan")

        # Update internal self model
        self.self_model.update(confidence, success)
        metrics = self.self_model.snapshot()

        # Action-specific adaptation
        if not success:
            self.self_model.record_failure(action)
        else:
            self.self_model.record_success(action)

        # Decide reflection outcome
        if metrics["failure_rate"] > 0.3:
            decision = ReflectionDecision(
                retrain=True,
                lower_confidence=True,
                note="High failure rate detected"
            )

        elif confidence > 0.7 and not success:
            decision = ReflectionDecision(
                retrain=True,
                lower_confidence=True,
                note="Overconfidence detected"
            )

        else:
            decision = ReflectionDecision(
                note="Stable performance"
            )

        # Emit reflection result as a signal (non-blocking)
        self.signal_bus.emit(
            "REFLECTION_RESULT",
            {
                "confidence": confidence,
                "success": success,
                "metrics": metrics,
                "note": decision.note,
                "retrain": decision.retrain
            }
        )

        # Emit learning signal for training system
        self.signal_bus.emit(
            "LEARNING_SIGNAL",
            {
                "action": action,
                "confidence": confidence,
                "success": success,
                "intent": intent,
                "plan": plan,
                "retrain": decision.retrain
            }
        )

        return decision