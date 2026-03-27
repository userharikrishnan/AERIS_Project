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

logger = logging.getLogger(__name__)


# =========================================================
# Router
# =========================================================

core_router = APIRouter()

# =========================================================
# Engines (singletons) - REMOVED - Now using dependency injection
# =========================================================

# REMOVED: attention = AttentionEngine()
# REMOVED: nlp = NLPProcessor()
# REMOVED: reasoning = ReasoningEngine()
# REMOVED: command_engine = CommandEngine()
# REMOVED: permission_engine = PermissionEngine()
# REMOVED: memory = MemoryEngine()
# REMOVED: goal_engine = GoalEngine()
# REMOVED: planner = Planner()
# REMOVED: focus_engine = FocusEngine(max_active_items=1)
# REMOVED: audit = AuditLogger()
# REMOVED: confirmations = ConfirmationEngine()
# REMOVED: preferences = PreferenceEngine()
# REMOVED: dry_run = DryRunSimulator()
# REMOVED: rollback_engine = RollbackEngine()
# REMOVED: explainer = ExplanationEngine()

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

    # User always preempts autonomy
    focus_engine.submit(
        FocusItem("user", PriorityLevel.CRITICAL)
    )

    audit.log(
        AuditEvent(
            AuditEventType.USER_INPUT,
            "user",
            {"text": text}
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


        # ==========================
        # CHAT MODE (ML-BASED)
        # ==========================
        if nlp_output.mode == "CHAT":

            response = reasoning.language_engine.generate_from_text(
                original_text=text,
                intent_type=nlp_output.type,
                confidence=nlp_output.confidence
            )

            return {
                "mode": "chat",
                "response": response,
                "confidence": nlp_output.confidence
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
            return {
                "mode": "chat",
                "response": reasoning_result.response,
                "confidence": reasoning_result.confidence
            }

        # FIX 3 & 4: Pass correct parameters and handle decision properly
        from services.trust_models import ActionSensitivity
        
        sensitivity = action_payload.get("sensitivity", ActionSensitivity.LOW)
        confidence = reasoning_result.confidence
        
        # FIX: Permission evaluation with correct parameters
        permission = permission_engine.evaluate(
            ctx=ctx,
            action_sensitivity=sensitivity,
            confidence=confidence
        )
        
        # FIX 4: Handle decision properly
        if not permission.allowed:
            return {
                "status": "blocked",
                "reason": permission.reason
            }
        
        if permission.require_confirmation:
            return {
                "status": "confirmation_required",
                "message": permission.reason,
                "action": action_payload
            }

        # 7. ALWAYS require confirmation
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
            "confirmation_id": confirmation.confirmation_id,
            "action": confirmation.action,
            "params": confirmation.params,
            "ui_candidates": confirmation.ui_candidates
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
    # 9.1 — Dry-run simulation
    # -------------------------
    preview = dry_run.simulate(
        action=confirmation.action,
        params=confirmation.params
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

    # -------------------------
    # Actual delegation
    # -------------------------
    delegation = delegation_engine.choose_device(
        required_capability=confirmation.action,
        min_trust=TrustLevel.STANDARD,
        params=confirmation.params
    )

    if not delegation.allowed:
        explanation = explainer.explain_failure(
            action=confirmation.action,
            reason=delegation.reason,
            rollback=None,
            confidence=0.4
        )

        # Record failure in memory
        memory.record_outcome(
            action=confirmation.action,
            success=False,
            context={"reason": delegation.reason}
        )

        return {
            "blocked": True,
            "reason": delegation.reason,
            "explanation": explanation
        }

    audit.log(
        AuditEvent(
            AuditEventType.ACTION_DELEGATED,
            "user",
            {
                "device_id": delegation.device_id,
                "tool": confirmation.action,
                "preview": preview.notes
            }
        )
    )

    # -------------------------
    # Rollback
    # -------------------------

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
    
    # Record successful outcome in memory
    memory.record_outcome(
        action=confirmation.action,
        success=True,
        context=confirmation.params
    )

    return {
        "delegate": True,
        "execution_id": execution_id,
        "device_id": delegation.device_id,
        "tool": confirmation.action,
        "params": confirmation.params,
        "preview": {
            "risk_level": preview.risk_level,
            "reversible": preview.reversible,
            "notes": preview.notes
        }
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