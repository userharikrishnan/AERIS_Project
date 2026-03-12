from services.trust_models import TrustLevel, ActionSensitivity
from services.rbac_models import Capability
from services.role_registry import RoleRegistry
from services.role_resolver import RoleResolver
from services.security_context import SecurityContext


class PermissionDecision:
    def __init__(self, allowed: bool, reason: str, require_confirmation=False):
        self.allowed = allowed
        self.reason = reason
        self.require_confirmation = require_confirmation


class PermissionEngine:
    def evaluate(
        self,
        *,
        ctx: SecurityContext,
        action_sensitivity: ActionSensitivity,
        confidence: float
    ) -> PermissionDecision:

        role = RoleResolver.resolve(ctx)

        # RBAC gate
        if not RoleRegistry.has_capability(role, Capability.EXECUTE_ACTION):
            return PermissionDecision(
                False,
                f"Role {role.name} not permitted to execute actions"
            )

        # Trust gates (unchanged logic)
        if ctx.user_trust == TrustLevel.UNTRUSTED:
            return PermissionDecision(False, "Untrusted user")

        if action_sensitivity == ActionSensitivity.CRITICAL and role != role.ADMIN:
            return PermissionDecision(False, "Admin role required")

        # Confidence gate
        if confidence < 0.3:
            return PermissionDecision(False, "Low confidence in reasoning")

        # Confirmation gates
        if action_sensitivity >= ActionSensitivity.HIGH:
            return PermissionDecision(
                True,
                "High-risk action",
                require_confirmation=True
            )

        return PermissionDecision(True, "Permission granted")
