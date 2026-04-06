from services.planner import Planner
from services.neural_scorer import NeuralScorer
from services.verifier import Verifier
from services.language_engine import LanguageEngine
from services.reflection_engine import ReflectionEngine
from services.self_model import SelfModel
from services.identity_engine import IdentityEngine
from services.clarification_engine import ClarificationEngine
from services.signal_bus import SignalBus

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
        clarification_question=None,
        trace=None
    ):
        self.plan = plan
        self.confidence = confidence
        self.verified = verified
        self.response = response
        self.visual_state = visual_state
        self.identity_signal = identity_signal
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question
        self.trace = trace or {}


class ReasoningEngine:
    def __init__(self, system=None):
        self.planner = Planner()
        self.scorer = NeuralScorer()
        self.verifier = Verifier()

        # Language components — dims match train.py v2 (loaded by LanguageEngine)
        self.tokenizer = Tokenizer()
        self.model = AerisSLM(
            vocab_size  = 1000,   # placeholder; LanguageEngine reloads from checkpoint
            embed_dim   = 384,
            num_layers  = 6,
            num_heads   = 6,
            ffn_dim     = 1536,
            dropout     = 0.0,
        )
        self.language_engine = LanguageEngine(self.model, self.tokenizer)

        # 👁 Vision
        self.vision = VisionEngine()

        # 🧬 Identity
        self.identity = IdentityEngine()

        # ❓ Clarification
        self.clarifier = ClarificationEngine()

        # Inject dependencies from system if provided
        if system:
            self.self_model = system.self_model
            self.reflection_engine = system.reflection_engine
        else:
            # Fallback for standalone usage (maintains backward compatibility)
            self.self_model = SelfModel()
            self._signal_bus = SignalBus()
            self.reflection_engine = ReflectionEngine(self.self_model, self._signal_bus)

    def reason(self, intent, original_text="", memory=None, session_context=None):
        visual_state = None
        identity_signal = None

        if intent.type in {
            "VISION_QUERY", "READ_SCREEN",
            "ACTIVE_WINDOW", "LIST_WINDOWS"
        }:
            visual_state = self.vision.perceive()

        # Get memory context if available
        context_memory = memory.get_context_bundle() if memory else {}

        # -----------------------------------------------------------------
        # CONTEXTUAL ENTITY MERGING (AERIS 4.0)
        # -----------------------------------------------------------------
        if session_context and "recent_history" in session_context:
            history = session_context["recent_history"]
            if history:
                last_turn = history[-1]
                last_entities = last_turn.get("entities", {})
                
                # If current intent is an action targeting something but lacks the target, inherit it
                # For example: "open c drive" -> Target: "c drive". Next command: "go to downloads" -> No target, just implicit path.
                if getattr(intent, 'entities', None) is not None:
                    # Specific merge: if we need a 'target' or 'app' or 'path' and don't have it
                    for key in ["target", "app", "path", "filename", "url"]:
                        if key not in intent.entities and key in last_entities:
                            intent.entities[key] = last_entities[key]
                            
                # Context-aware Path Assembly ("open c drive", "go to downloads" -> c:\downloads)
                if "target" in intent.entities and last_entities.get("target"):
                    curr_tgt = intent.entities["target"].lower()
                    last_tgt = last_entities["target"].lower()
                    
                    # If current is just a folder name and last was a drive/path
                    if ("drive" in last_tgt or "\\" in last_tgt or ":" in last_tgt) and ":" not in curr_tgt:
                        # Extract root
                        root = last_tgt.split()[0] if " " in last_tgt and ":" not in last_tgt else last_tgt
                        if "c drive" in last_tgt: root = "c:\\"
                        elif "d drive" in last_tgt: root = "d:\\"
                        
                        root_clean = root.rstrip('\\')
                        if curr_tgt != root_clean.lower():
                            intent.entities["target"] = f"{root_clean}\\{curr_tgt.split()[-1]}"
        # -----------------------------------------------------------------

        # Context-aware planning with memory and alternatives
        plan = self.planner.create_plan(
            intent=intent,
            context={
                "memory": context_memory,
                "alternatives": getattr(intent, "alternatives", []),
                "uncertainty": getattr(intent, "uncertainty", 0.0)
            }
        )
        raw_confidence = self.scorer.score(plan)
        verified = self.verifier.verify(plan)
        confidence = raw_confidence

        # Memory-based confidence correction
        if memory:
            outcomes = memory.search_memory(intent.raw_text)
            if outcomes:
                confidence *= 1.1

        capability = plan[0].get("action") if plan and isinstance(plan[0], dict) else None
        capability_belief = None
        identity_signal = None

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

        # Pre-execution risk check — only block on truly abysmal scores
        # Speech-to-text input naturally lowers confidence; margin matters more
        low_conf   = confidence < 0.15
        low_margin = getattr(intent, 'margin', 1.0) < 0.08
        if low_conf and low_margin:
            trace = {
                "intent": intent.type,
                "confidence": confidence,
                "alternatives": getattr(intent, 'alternatives', []),
                "plan": plan,
                "risk_check": "confidence_below_threshold"
            }
            return ReasoningResult(
                plan=plan,
                confidence=confidence,
                verified=verified,
                response="I'm not exactly sure what to do here.",
                visual_state=visual_state,
                identity_signal=identity_signal,
                needs_clarification=True,
                clarification_question="Can you clarify that for me?",
                trace=trace
            )

        # Removed aggressive second clarification gate — it was blocking
        # all speech commands with natural confidence variance.
        # Clarification now only triggers at the first gate (conf < 0.15 + low margin).

        response = None

        if intent.type == "VISION_QUERY" and visual_state:
            response = self._describe_visual_state(visual_state)
        elif verified and confidence > 0.12:
            response = self.language_engine.generate_from_text(
                original_text=original_text,
                intent_type=intent.type,
                confidence=confidence
            )
            if response:
                if len(response.split()) < 3:
                    response = f"On it — {intent.type.lower().replace('_', ' ')}."

        # Build reasoning trace for debugging and explainability
        trace = {
            "intent": intent.type,
            "confidence": confidence,
            "alternatives": getattr(intent, 'alternatives', []),
            "plan": plan
        }

        return ReasoningResult(
            plan=plan,
            confidence=confidence,
            verified=verified,
            response=response,
            visual_state=visual_state,
            identity_signal=identity_signal,
            trace=trace
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
        return self.reflection_engine.reflect(
            action=action,
            confidence=confidence,
            success=success,
            context={
                "intent": action,
                "confidence": confidence
            }
        )