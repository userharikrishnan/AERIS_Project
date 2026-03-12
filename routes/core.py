from fastapi import APIRouter

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

# -------------------------
# Explanation
# -------------------------
from services.explanation_engine import ExplanationEngine



# =========================================================
# Router
# =========================================================

core_router = APIRouter()

# =========================================================
# Engines (singletons)
# =========================================================

attention = AttentionEngine()
nlp = NLPProcessor()
reasoning = ReasoningEngine()
command_engine = CommandEngine()
permission_engine = PermissionEngine()
memory = MemoryEngine()

goal_engine = GoalEngine()
planner = Planner()

focus_engine = FocusEngine(max_active_items=1)
audit = AuditLogger()
confirmations = ConfirmationEngine()
preferences = PreferenceEngine()   # ✅ ADDED (8.5)

dry_run = DryRunSimulator()

rollback_engine = RollbackEngine()

explainer = ExplanationEngine()

# =========================================================
# Device Registry & Delegation
# =========================================================

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

scheduler = Scheduler(interval_seconds=10)

goal_executor = GoalExecutor(
    goal_engine=goal_engine,
    planner=planner,
    reasoning_engine=reasoning,
    permission_engine=permission_engine,
    delegation_engine=delegation_engine,
    focus_engine=focus_engine,
    audit_logger=audit
)

scheduler.add_job(goal_executor.tick)
scheduler.start()

# =========================================================
# Core Input Endpoint
# =========================================================

@core_router.post("/input")
def process_input(payload: dict):
    text = payload.get("text", "").strip()
    user_trust = TrustLevel.STANDARD

    ctx = SecurityContext(
        actor="user",
        user_trust=user_trust,
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
        intent = nlp.extract_intent(text)

        # 3. Reasoning
        reasoning_result = reasoning.reason(intent)

        # 4. Memory
        memory.remember(text)

        # 5. Command planning
        action_payload = command_engine.plan(intent)

        if not action_payload:
            return {
                "response": reasoning_result.response,
                "confidence": reasoning_result.confidence
            }

        # 6. Permission
        permission = permission_engine.evaluate(
            ctx=ctx,
            action_sensitivity=action_payload["sensitivity"],
            confidence=reasoning_result.confidence
        )

        if not permission.allowed:
            return {"blocked": True, "reason": permission.reason}

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
def confirm_action(payload: dict):
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
def create_goal(payload: dict):
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
def list_goals():
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
def pause_goal(payload: dict):
    goal_engine.pause_goal(payload.get("goal_id"))
    return {"status": "paused"}


@core_router.post("/goals/resume")
def resume_goal(payload: dict):
    goal_engine.resume_goal(payload.get("goal_id"))
    return {"status": "resumed"}

# =========================================================
# Autonomy Control Endpoints (4.2)
# =========================================================

@core_router.post("/autonomy/start")
def start_autonomy():
    scheduler.start()
    return {"status": "autonomous execution started"}


@core_router.post("/autonomy/stop")
def stop_autonomy():
    scheduler.stop()
    return {"status": "autonomous execution stopped"}

# =========================================================
# Memory Maintenance Scheduler (7.1)
# =========================================================

try:
    from services.memory_maintenance_scheduler import MemoryMaintenanceScheduler

    memory_maintenance = MemoryMaintenanceScheduler(memory)
    memory_maintenance.start()
except:
    print("Memory Maintenance Scheduler module error")

# =========================================================
# Visual Recall Endpoints (6.4)
# =========================================================

try:
    from services.visual_recall_engine import VisualRecallEngine

    visual_recall = VisualRecallEngine()

    @core_router.get("/recall/visual/recent")
    def recall_recent_visual(limit: int = 5):
        return {"memories": visual_recall.recall_recent(limit)}

    @core_router.get("/recall/visual/window")
    def recall_by_window(window_title: str, limit: int = 5):
        return {
            "window": window_title,
            "memories": visual_recall.recall_by_window(window_title, limit)
        }

    @core_router.get("/recall/visual/intent")
    def recall_by_intent(intent_type: str, limit: int = 5):
        return {
            "intent": intent_type,
            "memories": visual_recall.recall_by_intent(intent_type, limit)
        }
except:
    print("Error for visual recall endpoint module")


# -------------------------
# Rollback
# -------------------------


@core_router.post("/rollback")
def rollback_action(payload: dict):
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
