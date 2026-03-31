from fastapi import APIRouter, Request

# -------------------------
# Core cognition
# -------------------------
from services.attention_engine import AttentionEngine
from services.nlp_processor import NLPProcessor
from services.reasoning_engine import ReasoningEngine
from services.command_engine import CommandEngine
from services.memory_engine import MemoryEngine

# -------------------------
# Governance
# -------------------------
from services.permission_engine import PermissionEngine
from services.trust_models import TrustLevel
from services.rbac_models import Role

# -------------------------
# Delegation (3.4)
# -------------------------
from services.device_registry import DeviceRegistry, Device
from services.delegation_engine import DelegationEngine

# -------------------------
# Goals & Autonomy (4.0 / 4.2)
# -------------------------
from services.goal_engine import GoalEngine
from services.planner import Planner
from services.scheduler import Scheduler
from services.goal_executor import GoalExecutor

# -------------------------
# Focus & Priority Management (4.3)
# -------------------------
from services.focus_engine import FocusEngine, FocusItem
from services.priority_models import PriorityLevel

# -------------------------
# Security & Audit (5.x)
# -------------------------
from services.audit_logger import AuditLogger
from services.security_models import AuditEvent, AuditEventType
from services.security_context import SecurityContext

# -------------------------
# User Confirmation (6.2)
# -------------------------
from services.confirmation_engine import ConfirmationEngine

# -------------------------
# Preferences (8.4 / 8.5)
# -------------------------
from services.preference_engine import PreferenceEngine

# -------------------------
# Simulator
# -------------------------
from services.dry_run_simulator import DryRunSimulator

# -------------------------
# Rollback
# -------------------------
from services.rollback_engine import RollbackEngine
from models.rollback import RollbackRecord
import uuid
import logging

# -------------------------
# Explanation
# -------------------------
from services.explanation_engine import ExplanationEngine

# ==============================
# NEW: Core execution pipeline
# ==============================
from services.tool_dispatcher import ToolDispatcher
from services.smart_confirmation import SmartConfirmationEngine
from services.session_engine import SessionEngine
from services.plan_executor import PlanExecutor

logger = logging.getLogger(__name__)


# =========================================================
# Router
# =========================================================

core_router = APIRouter()

# =========================================================
# Module-level singletons for new services
# (These are created once and shared across all requests)
# =========================================================
_tool_dispatcher = ToolDispatcher()
_smart_confirmation = SmartConfirmationEngine()
_session_engine = SessionEngine()
_plan_executor = PlanExecutor(_tool_dispatcher)

# =========================================================
# Device Registry & Delegation
# =========================================================

# NOTE: These are kept but will be accessed through system in endpoints
# They are initialized here but endpoints will use system instances
device_registry = DeviceRegistry()
delegation_engine = DelegationEngine(device_registry)

device_registry.register(
    Device(
        device_id="local",
        name="Primary Machine",
        trust=TrustLevel.ADMIN,
        capabilities=["browser", "filesystem", "app"]
    )
)

# =========================================================
# Autonomous Goal Execution (4.2)
# =========================================================

# REMOVED: scheduler = Scheduler(interval_seconds=10)
# REMOVED: scheduler.start() - Moved to /autonomy/start control

# NOTE: goal_executor is kept but will be accessed through system
goal_executor = None  # Will be initialized in main.py with system


# =========================================================
# Core Input Endpoint
# =========================================================

@core_router.post("/input")
def process_input(payload: dict, request: Request):
    # Dependency injection from system container
    system = request.app.state.system

    # Map engines from system
    attention = system.attention_engine
    nlp = system.nlp_processor
    reasoning = system.reasoning_engine
    command_engine = system.command_engine
    permission_engine = system.permission_engine
    memory = system.memory_engine
    focus_engine = system.focus_engine
    audit = system.audit_logger
    confirmations = system.confirmation_engine

    text = payload.get("text", "").strip()
    user_trust = TrustLevel.STANDARD

    # FIX 2: Build Security Context properly
    ctx = SecurityContext(
        actor="local_user",
        user_trust=user_trust,
        device_id="local",
        autonomous=False
    )

    # ---- Session Management ----
    session = _session_engine.get_or_create_session()
    session_id = session.session_id

    # User always preempts autonomy
    focus_engine.submit(
        FocusItem("user", PriorityLevel.CRITICAL)
    )

    audit.log(
        AuditEvent(
            AuditEventType.USER_INPUT,
            "user",
            {"text": text, "session_id": session_id[:8]}
        )
    )

    try:
        # 1. Attention
        attention_decision = attention.evaluate(text)
        if not attention_decision.allow:
            return {"blocked": True, "reason": attention_decision.reason}

        # 2. NLP
        nlp_output = nlp.process(text)

        intent_type = nlp_output.type
        confidence = nlp_output.confidence

        # ---- Session close detection ----
        session_closing = _session_engine.detect_close_intent(text, intent_type)
        if session_closing and _session_engine.has_active_session:
            closed = _session_engine.close_current_session()
            response = reasoning.language_engine.generate_from_text(
                original_text=text,
                intent_type="CHAT",
                confidence=confidence
            )
            _session_engine.record_turn(text, intent_type, response or "Session closed.")
            return {
                "mode": "chat",
                "response": response or "Understood. Session ended. Let me know when you need me again.",
                "session_closed": True,
                "session_id": session_id[:8]
            }

        # ==========================
        # CHAT MODE (ML-BASED)
        # ==========================
        if nlp_output.mode == "CHAT":
            response = reasoning.language_engine.generate_from_text(
                original_text=text,
                intent_type=nlp_output.type,
                confidence=nlp_output.confidence,
                entities=nlp_output.entities
            )
            _session_engine.record_turn(text, intent_type, response or "")
            return {
                "mode": "chat",
                "response": response,
                "confidence": nlp_output.confidence,
                "session_id": session_id[:8]
            }

        # 3. Reasoning with context
        reasoning_result = reasoning.reason(
            intent=nlp_output,
            original_text=text,
            memory=memory
        )

        # Response validation layer
        if not reasoning_result.response or reasoning_result.confidence < 0.3:
            return {
                "response": reasoning_result.response or "Action planned",
                "confidence": reasoning_result.confidence,
                "plan": reasoning_result.plan
            }

        # 4. Memory with rich context
        memory.remember_interaction(
            user_input=text,
            intent=nlp_output.type,
            reasoning=reasoning_result.summary if hasattr(reasoning_result, 'summary') else str(reasoning_result),
            confidence=reasoning_result.confidence
        )

        # 5. Command planning with reasoning context
        action_payload = command_engine.plan(
            reasoning_output=reasoning_result,
            context={
                "confidence": reasoning_result.confidence,
                "memory": memory.get_recent()
            }
        )

        if not action_payload:
            response = reasoning_result.response or "I understand. Let me know how you'd like to proceed."
            _session_engine.record_turn(text, intent_type, response)
            return {
                "mode": "chat",
                "response": response,
                "confidence": reasoning_result.confidence
            }

        # === SMART DUAL-MODE CONFIRMATION ===
        action_name = action_payload.get("action", "")
        action_params = action_payload.get("params", {})

        # -------------------------------------------------------
        # NON-DESTRUCTIVE BYPASS: chat/identity/memory actions
        # never require confirmation — they don't touch system state
        # -------------------------------------------------------
        BYPASS_ACTIONS = {
            "respond", "identity_query", "reason",
            "memory_store", "memory_recall", "memory_forget",
            "clarify", "goal_list", "active_window", "list_windows",
        }

        if action_name in BYPASS_ACTIONS:
            # Always use template-based response for non-destructive actions
            # SLM output quality is unreliable for short operational phrases
            response = reasoning.language_engine.generate_from_text(
                original_text=text,
                intent_type=intent_type,
                confidence=reasoning_result.confidence,
                entities=nlp_output.entities
            )
            _session_engine.record_turn(text, intent_type, response or "")
            return {
                "mode": "chat",
                "response": response or "How can I help?",
                "confidence": reasoning_result.confidence,
                "session_id": session_id[:8]
            }

        smart_decision = _smart_confirmation.evaluate(action_name, action_params)

        from services.trust_models import ActionSensitivity
        sensitivity = action_payload.get("sensitivity", ActionSensitivity.LOW)
        confidence = reasoning_result.confidence

        # Permission evaluation
        permission = permission_engine.evaluate(
            ctx=ctx,
            action_sensitivity=sensitivity,
            confidence=confidence
        )

        if not permission.allowed:
            return {
                "status": "blocked",
                "reason": permission.reason
            }

        # Auto-approve if smart confirmation says OK
        if smart_decision.auto_approved and not permission.require_confirmation:
            # Execute immediately without asking
            execution_result = _plan_executor.execute_plan(
                reasoning_result.plan,
                initial_context={"session_id": session_id, "original_text": text}
            )

            # Record outcome for learning
            _smart_confirmation.record_approval(action_name, action_params, smart_decision.fingerprint)

            response = reasoning_result.response or reasoning.language_engine.generate_from_text(
                original_text=text,
                intent_type=intent_type,
                confidence=confidence,
                entities=nlp_output.entities
            )
            _session_engine.record_turn(text, intent_type, response or "", action_name)

            result_summary = execution_result.to_summary()
            return {
                "mode": "auto_executed",
                "auto_approved": True,
                "reason": smart_decision.reason,
                "response": response,
                "execution": result_summary,
                "session_id": session_id[:8]
            }

        # First time — request confirmation
        confirmation = confirmations.create(
            action=action_payload["action"],
            params=action_payload["params"],
            ui_candidates=action_payload.get("ui_candidates")
        )

        audit.log(
            AuditEvent(
                AuditEventType.ACTION_PLANNED,
                "user",
                {"confirmation_id": confirmation.confirmation_id}
            )
        )

        return {
            "confirmation_required": True,
            "first_time": True,
            "smart_confirmation_note": smart_decision.reason,
            "confirmation_id": confirmation.confirmation_id,
            "action": confirmation.action,
            "params": confirmation.params,
            "ui_candidates": confirmation.ui_candidates,
            "session_id": session_id[:8]
        }

    finally:
        focus_engine.consume("user")


# =========================================================
# Confirmation Endpoint (6.2 + 8.5)
# =========================================================

@core_router.post("/confirm")
def confirm_action(payload: dict, request: Request):
    system = request.app.state.system

    # Map engines from system
    confirmations = system.confirmation_engine
    preferences = system.preference_engine
    audit = system.audit_logger
    dry_run = system.dry_run_simulator
    delegation_engine = system.delegation_engine
    explainer = system.explanation_engine
    rollback_engine = system.rollback_engine
    memory = system.memory_engine

    confirmation_id = payload.get("confirmation_id")
    approved = payload.get("approved", False)
    preview_only = payload.get("preview_only", False)

    confirmation = confirmations.consume(confirmation_id)
    if not confirmation:
        return {"error": "Invalid or expired confirmation"}

    preferences.record_confirmation(
        action=confirmation.action,
        approved=approved
    )

    if not approved:
        audit.log(
            AuditEvent(
                AuditEventType.ACTION_BLOCKED,
                "user",
                {"confirmation_id": confirmation_id, "reason": "User rejected"}
            )
        )
        return {"status": "cancelled"}

    # -------------------------
    # 9.1 — Dry-run simulation (best-effort, never blocks execution)
    # -------------------------
    try:
        preview = dry_run.simulate(
            action=confirmation.action,
            params=confirmation.params
        )
    except Exception as e:
        logger.warning(f"[confirm] dry_run.simulate failed (non-fatal): {e}")
        # Create a minimal safe preview so execution continues
        from types import SimpleNamespace
        preview = SimpleNamespace(
            action=confirmation.action,
            params=confirmation.params,
            affected_resources=[],
            risk_level="medium",
            reversible=False,
            rollback_params=None,
            notes="Simulation skipped"
        )

    if preview_only:
        return {
            "preview": {
                "action": preview.action,
                "params": preview.params,
                "affected_resources": preview.affected_resources,
                "risk_level": preview.risk_level,
                "reversible": preview.reversible,
                "notes": preview.notes
            }
        }

    # =========================================================
    # ACTUAL TOOL EXECUTION (The missing last mile — now wired!)
    # =========================================================

    audit.log(
        AuditEvent(
            AuditEventType.ACTION_DELEGATED,
            "user",
            {
                "tool": confirmation.action,
                "params": confirmation.params,
                "preview": preview.notes
            }
        )
    )

    execution_id = str(uuid.uuid4())

    # Register rollback if possible
    if preview.reversible and getattr(preview, "rollback_params", None):
        rollback_engine.register(
            execution_id,
            RollbackRecord(
                action=confirmation.action,
                params=confirmation.params,
                rollback_params=preview.rollback_params
            )
        )

    # === DISPATCH TO ACTUAL TOOL ===
    tool_result = _tool_dispatcher.dispatch(
        action=confirmation.action,
        params=confirmation.params
    )

    # Record approval for smart confirmation (so next time it's auto-approved)
    _smart_confirmation.record_approval(
        action=confirmation.action,
        params=confirmation.params
    )

    # Record outcome in memory
    memory.record_outcome(
        action=confirmation.action,
        success=tool_result.success,
        context=confirmation.params
    )

    if not tool_result.success:
        # Record error to degrade trust for this action
        _smart_confirmation.record_error(
            action=confirmation.action,
            params=confirmation.params,
            error=tool_result.error or "unknown error"
        )
        return {
            "status": "execution_failed",
            "execution_id": execution_id,
            "tool": confirmation.action,
            "error": tool_result.error,
            "data": tool_result.data  # May contain clarification info
        }

    audit.log(
        AuditEvent(
            AuditEventType.ACTION_SUCCEEDED,
            "user",
            {
                "execution_id": execution_id,
                "tool": confirmation.action,
                "success": True
            }
        )
    )

    return {
        "status": "executed",
        "execution_id": execution_id,
        "tool": confirmation.action,
        "success": tool_result.success,
        "result": tool_result.data,
        "preview": {
            "risk_level": preview.risk_level,
            "reversible": preview.reversible,
        },
        "smart_confirmation_note": "This action will be auto-approved next time."
    }


# =========================================================
# Goal Management Endpoints (4.0)
# =========================================================

@core_router.post("/goals/create")
def create_goal(payload: dict, request: Request):
    system = request.app.state.system
    goal_engine = system.goal_engine
    planner = system.planner
    audit = system.audit_logger
    
    description = payload.get("description", "").strip()
    if not description:
        return {"error": "Goal description required"}

    goal = goal_engine.create_goal(description)

    objectives = planner.expand_goal(description)
    for obj in objectives:
        goal.add_objective(obj)

    goal.update_progress()

    audit.log(
        AuditEvent(
            AuditEventType.GOAL_CREATED,
            "user",
            {"goal_id": goal.goal_id, "description": description}
        )
    )

    return {
        "goal_id": goal.goal_id,
        "description": goal.description,
        "objectives": goal.objectives,
        "status": goal.status.value
    }


@core_router.get("/goals")
def list_goals(request: Request):
    system = request.app.state.system
    goal_engine = system.goal_engine
    
    goals = goal_engine.list_active_goals()
    return [
        {
            "goal_id": g.goal_id,
            "description": g.description,
            "progress": g.progress,
            "status": g.status.value
        }
        for g in goals
    ]


@core_router.post("/goals/pause")
def pause_goal(payload: dict, request: Request):
    system = request.app.state.system
    goal_engine = system.goal_engine
    
    goal_engine.pause_goal(payload.get("goal_id"))
    return {"status": "paused"}


@core_router.post("/goals/resume")
def resume_goal(payload: dict, request: Request):
    system = request.app.state.system
    goal_engine = system.goal_engine
    
    goal_engine.resume_goal(payload.get("goal_id"))
    return {"status": "resumed"}


# =========================================================
# Autonomy Control Endpoints (4.2)
# =========================================================

@core_router.post("/autonomy/start")
def start_autonomy(request: Request):
    system = request.app.state.system
    scheduler = system.scheduler
    
    scheduler.start()
    return {"status": "autonomous execution started"}


@core_router.post("/autonomy/stop")
def stop_autonomy(request: Request):
    system = request.app.state.system
    scheduler = system.scheduler
    
    scheduler.stop()
    return {"status": "autonomous execution stopped"}


# =========================================================
# Memory Maintenance Scheduler (7.1)
# =========================================================

try:
    from services.memory_maintenance_scheduler import MemoryMaintenanceScheduler

    @core_router.on_event("startup")
    def start_memory_maintenance():
        # This will be called when app starts - implementation depends on FastAPI version
        pass
except Exception as e:
    logger.error(f"Memory Maintenance Scheduler module error: {e}")


# =========================================================
# Visual Recall Endpoints (6.4)
# =========================================================

try:
    from services.visual_recall_engine import VisualRecallEngine

    @core_router.get("/recall/visual/recent")
    def recall_recent_visual(limit: int = 5, request: Request = None):
        if request and hasattr(request.app.state, 'system') and hasattr(request.app.state.system, 'visual_recall'):
            visual_recall = request.app.state.system.visual_recall
            return {"memories": visual_recall.recall_recent(limit)}
        return {"error": "Visual recall not available"}

    @core_router.get("/recall/visual/window")
    def recall_by_window(window_title: str, limit: int = 5, request: Request = None):
        if request and hasattr(request.app.state, 'system') and hasattr(request.app.state.system, 'visual_recall'):
            visual_recall = request.app.state.system.visual_recall
            return {
                "window": window_title,
                "memories": visual_recall.recall_by_window(window_title, limit)
            }
        return {"error": "Visual recall not available"}

    @core_router.get("/recall/visual/intent")
    def recall_by_intent(intent_type: str, limit: int = 5, request: Request = None):
        if request and hasattr(request.app.state, 'system') and hasattr(request.app.state.system, 'visual_recall'):
            visual_recall = request.app.state.system.visual_recall
            return {
                "intent": intent_type,
                "memories": visual_recall.recall_by_intent(intent_type, limit)
            }
        return {"error": "Visual recall not available"}
except Exception as e:
    logger.error(f"Visual recall init failed: {e}")


# -------------------------
# Rollback
# -------------------------

@core_router.post("/rollback")
def rollback_action(payload: dict, request: Request):
    system = request.app.state.system
    rollback_engine = system.rollback_engine
    audit = system.audit_logger
    explainer = system.explanation_engine
    
    execution_id = payload.get("execution_id")

    record = rollback_engine.rollback(execution_id)
    if not record:
        return {"error": "No rollback available for this execution"}

    audit.log(
        AuditEvent(
            AuditEventType.ACTION_ROLLBACK,
            "user",
            {
                "execution_id": execution_id,
                "action": record.action
            }
        )
    )

    # 🧠 9.5 — Human explanation
    explanation = explainer.explain_failure(
        action=record.action,
        reason="User requested rollback",
        rollback=record,
        confidence=1.0
    )

    return {
        "rollback": True,
        "action": record.action,
        "rollback_params": record.rollback_params,
        "explanation": explanation
    }