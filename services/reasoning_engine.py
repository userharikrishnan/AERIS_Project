from services.planner import Planner
from services.neural_scorer import NeuralScorer
from services.verifier import Verifier
from services.language_engine import LanguageEngine
from services.reflection_engine import ReflectionEngine
from services.self_model import SelfModel
from services.identity_engine import IdentityEngine
from services.clarification_engine import ClarificationEngine

from models.slm import AerisSLM
from models.tokenizer import Tokenizer

# 👁 Vision
from services.vision_engine import VisionEngine
from services.visual_models import VisualState


class ReasoningResult:
    def __init__(
        self,
        plan,
        confidence,
        verified,
        response=None,
        visual_state=None,
        identity_signal=None,
        needs_clarification=False,
        clarification_question=None
    ):
        self.plan = plan
        self.confidence = confidence
        self.verified = verified
        self.response = response
        self.visual_state = visual_state
        self.identity_signal = identity_signal
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question

        self.self_model = SelfModel()
        self.reflection_engine = ReflectionEngine(self.self_model)


class ReasoningEngine:
    def __init__(self):
        self.planner = Planner()
        self.scorer = NeuralScorer()
        self.verifier = Verifier()

        # Language components
        self.tokenizer = Tokenizer()
        self.model = AerisSLM(vocab_size=1000)
        self.language_engine = LanguageEngine(self.model, self.tokenizer)

        # 👁 Vision
        self.vision = VisionEngine()

        # 🧬 Identity
        self.identity = IdentityEngine()

        # ❓ Clarification
        self.clarifier = ClarificationEngine()

    def reason(self, intent):
        visual_state = None
        identity_signal = None

        # ---------- VISION ----------
        if intent.type in {
            "VISION_QUERY",
            "READ_SCREEN",
            "ACTIVE_WINDOW",
            "LIST_WINDOWS"
        }:
            visual_state = self.vision.perceive()

        # ---------- PLANNING ----------
        plan = self.planner.create_plan(intent)

        # ---------- SCORING ----------
        raw_confidence = self.scorer.score(plan)
        verified = self.verifier.verify(plan)
        confidence = raw_confidence

        capability = plan.get("action") if isinstance(plan, dict) else None
        capability_belief = None

        # ---------- IDENTITY SIGNAL ----------
        if capability:
            capability_belief = self.identity.state.capabilities.get(capability)
            if capability_belief:
                success_rate = capability_belief.success_rate
                identity_signal = {
                    "capability": capability,
                    "success_rate": success_rate
                }

                if success_rate < 0.5:
                    confidence *= (0.5 + success_rate)

        # ---------- SELF-QUESTIONING (8.3) ----------
        if verified and confidence < 0.45:
            question = self.clarifier.generate(
                intent_type=intent.type,
                capability=capability
            )

            return ReasoningResult(
                plan=plan,
                confidence=confidence,
                verified=verified,
                response=None,
                visual_state=visual_state,
                identity_signal=identity_signal,
                needs_clarification=True,
                clarification_question=question
            )

        # ---------- RESPONSE ----------
        response = None

        if intent.type == "VISION_QUERY" and visual_state:
            response = self._describe_visual_state(visual_state)

        elif verified and confidence > 0.3:
            response = self.language_engine.generate_from_plan(
                plan,
                confidence=confidence,
                verified=verified
            )

        return ReasoningResult(
            plan=plan,
            confidence=confidence,
            verified=verified,
            response=response,
            visual_state=visual_state,
            identity_signal=identity_signal
        )

    def _describe_visual_state(self, visual_state: VisualState) -> str:
        lines = [
            f"Screen resolution: {visual_state.resolution[0]}x{visual_state.resolution[1]}",
            f"Active window: {visual_state.active_window or 'Unknown'}",
            f"Open windows: {len(visual_state.windows)}"
        ]

        if visual_state.text_blocks:
            sample = " ".join(tb.text for tb in visual_state.text_blocks[:10])
            lines.append(f"Visible text sample: {sample}")

        return "\n".join(lines)

    def post_execute_reflection(self, action: str, confidence: float, success: bool):
        self.identity.record_action(
            action=action,
            success=success,
            confidence=confidence
        )
        return self.reflection_engine.reflect(confidence, success)
