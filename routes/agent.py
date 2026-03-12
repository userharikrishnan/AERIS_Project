from fastapi import APIRouter

from services.audit_logger import AuditLogger
from services.security_models import AuditEvent, AuditEventType
from services.security_context import SecurityContext
from services.role_resolver import RoleResolver
from services.rbac_models import Role

agent_router = APIRouter()
audit = AuditLogger()


# =========================================================
# EXECUTION ENDPOINT (UNCHANGED)
# =========================================================

@agent_router.post("/execute")
def execute_tool(payload: dict):
    """
    FINAL execution boundary.
    This endpoint MUST NEVER execute without:
    - explicit user confirmation
    - RBAC approval
    """

    # -------------------------
    # 1. Confirmation gate (6.2)
    # -------------------------
    if not payload.get("confirmed", False):
        audit.log(
            AuditEvent(
                event_type=AuditEventType.ACTION_BLOCKED,
                actor="agent",
                details={
                    "reason": "Missing explicit user confirmation",
                    "payload": payload
                }
            )
        )
        return {
            "success": False,
            "error": "Execution requires explicit user confirmation"
        }

    # -------------------------
    # 2. Extract execution data
    # -------------------------
    device_id = payload.get("device_id")
    tool = payload.get("tool")
    params = payload.get("params", {})
    execution_id = payload.get("execution_id")

    if not device_id or not tool or not execution_id:
        audit.log(
            AuditEvent(
                event_type=AuditEventType.SYSTEM_ERROR,
                actor="agent",
                details={
                    "error": "Missing device_id, tool, or execution_id",
                    "payload": payload
                }
            )
        )
        return {
            "success": False,
            "error": "Invalid execution payload"
        }

    # -------------------------
    # 3. Security context
    # -------------------------
    ctx = SecurityContext(
        actor="agent",
        device_id=device_id,
        autonomous=False
    )

    # -------------------------
    # 4. RBAC enforcement (5.2)
    # -------------------------
    role = RoleResolver.resolve(ctx)

    if role not in {Role.ADMIN, Role.AUTONOMOUS}:
        audit.log(
            AuditEvent(
                event_type=AuditEventType.ACTION_BLOCKED,
                actor="agent",
                details={
                    "reason": f"Role {role.name} not permitted to execute tools",
                    "device_id": device_id,
                    "tool": tool
                }
            )
        )
        return {
            "success": False,
            "error": "RBAC violation"
        }

    # -------------------------
    # 5. Audit execution intent
    # -------------------------
    audit.log(
        AuditEvent(
            event_type=AuditEventType.ACTION_DELEGATED,
            actor="agent",
            details={
                "execution_id": execution_id,
                "device_id": device_id,
                "tool": tool,
                "params": params,
                "confirmed": True
            }
        )
    )

    # -------------------------
    # 6. Execute (local only)
    # -------------------------
    # Actual OS execution happens BELOW this layer

    return {
        "success": True,
        "execution_id": execution_id,
        "device_id": device_id,
        "tool": tool,
        "params": params
    }


# =========================================================
# RUNTIME RESULT REPORTING (9.4) ✅ NEW
# =========================================================

@agent_router.post("/report")
def report_execution_result(payload: dict):
    """
    Reports the actual execution outcome back to core.
    """

    execution_id = payload.get("execution_id")
    success = payload.get("success", False)
    error = payload.get("error")

    if not execution_id:
        audit.log(
            AuditEvent(
                event_type=AuditEventType.SYSTEM_ERROR,
                actor="agent",
                details={
                    "error": "Missing execution_id in report",
                    "payload": payload
                }
            )
        )
        return {"ack": False, "error": "Missing execution_id"}

    event_type = (
        AuditEventType.ACTION_SUCCEEDED
        if success
        else AuditEventType.ACTION_FAILED
    )

    audit.log(
        AuditEvent(
            event_type=event_type,
            actor="agent",
            details={
                "execution_id": execution_id,
                "success": success,
                "error": error
            }
        )
    )

    return {
        "ack": True,
        "execution_id": execution_id,
        "success": success
    }
