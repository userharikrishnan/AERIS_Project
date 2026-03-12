from services.trust_models import TrustLevel
from services.focus_engine import FocusEngine, FocusItem
from services.priority_models import PriorityLevel

# 🔐 Security & Audit
from services.audit_logger import AuditLogger
from services.security_models import AuditEvent, AuditEventType
from services.security_context import SecurityContext

# 🔁 Rollback
from services.rollback_engine import RollbackEngine
from models.rollback import RollbackRecord
import uuid


class GoalExecutor:
    """
    Executes goals incrementally with transactional rollback safety.
    """

    def __init__(
        self,
        goal_engine,
        planner,
        reasoning_engine,
        permission_engine,
        delegation_engine,
        focus_engine: FocusEngine,
        audit_logger: AuditLogger | None = None,
        rollback_engine: RollbackEngine | None = None
    ):
        self.goal_engine = goal_engine
        self.planner = planner
        self.reasoning = reasoning_engine
        self.permission = permission_engine
        self.delegation = delegation_engine
        self.focus = focus_engine

        self.audit = audit_logger or AuditLogger()
        self.rollback = rollback_engine or RollbackEngine()

    def tick(self):
        self.focus.submit(
            FocusItem(source="goal", priority=PriorityLevel.MEDIUM)
        )

        if not self.focus.allow_execution("goal"):
            return

        active_goals = self.goal_engine.list_active_goals()
        if not active_goals:
            self.focus.consume("goal")
            return

        for goal in active_goals:
            self._execute_goal_transaction(goal)
            self.focus.consume("goal")
            return

    # --------------------------------------------------
    # Transactional execution
    # --------------------------------------------------

    def _execute_goal_transaction(self, goal):
        tx_id = f"goal-{goal.goal_id}"
        self.rollback.begin_transaction(tx_id)

        for idx, obj in enumerate(goal.objectives):
            if obj["done"]:
                continue

            success = self._attempt_objective(
                goal=goal,
                index=idx,
                text=obj["text"],
                tx_id=tx_id
            )

            if not success:
                # 🔁 Rollback everything executed so far
                rollbacks = self.rollback.rollback_transaction(tx_id)

                for record in rollbacks:
                    self.audit.log(
                        AuditEvent(
                            event_type=AuditEventType.ACTION_ROLLBACK,
                            actor="goal_executor",
                            details={
                                "goal_id": goal.goal_id,
                                "action": record.action,
                                "rollback_params": record.rollback_params
                            }
                        )
                    )

                return

        # ✅ All objectives succeeded
        self.rollback.commit_transaction(tx_id)

    # --------------------------------------------------
    # Single objective attempt
    # --------------------------------------------------

    def _attempt_objective(self, goal, index, text, tx_id: str) -> bool:
        ctx = SecurityContext(actor="goal_executor", autonomous=True)

        intent = self.reasoning.nlp.extract_intent(text)
        reasoning_result = self.reasoning.reason(intent)

        if not reasoning_result.verified or reasoning_result.confidence < 0.4:
            self.audit.log(
                AuditEvent(
                    event_type=AuditEventType.ACTION_BLOCKED,
                    actor="goal_executor",
                    details={
                        "goal_id": goal.goal_id,
                        "objective": text,
                        "reason": "low confidence or unverified"
                    }
                )
            )
            return False

        action_payload = self.reasoning.command_engine.plan(intent)
        if not action_payload:
            self.goal_engine.mark_objective_done(goal.goal_id, index)
            return True

        permission = self.permission.evaluate(
            user_trust=TrustLevel.STANDARD,
            device_trust=TrustLevel.STANDARD,
            action_sensitivity=action_payload["sensitivity"],
            confidence=reasoning_result.confidence
        )

        if not permission.allowed or permission.require_confirmation:
            return False

        delegation = self.delegation.choose_device(
            required_capability=action_payload["action"],
            min_trust=TrustLevel.STANDARD
        )

        if not delegation.allowed:
            return False

        # -------------------------
        # Register rollback (if possible)
        # -------------------------
        preview = getattr(reasoning_result, "preview", None)
        if preview and preview.reversible and preview.rollback_params:
            self.rollback.register_in_transaction(
                tx_id,
                RollbackRecord(
                    action=action_payload["action"],
                    params=action_payload["params"],
                    rollback_params=preview.rollback_params
                )
            )

        self.goal_engine.mark_objective_done(goal.goal_id, index)

        self.audit.log(
            AuditEvent(
                event_type=AuditEventType.ACTION_DELEGATED,
                actor="goal_executor",
                details={
                    "goal_id": goal.goal_id,
                    "objective": text,
                    "device_id": delegation.device_id,
                    "tool": action_payload["action"]
                }
            )
        )

        return True
