from services.rbac_models import Role
from services.security_context import SecurityContext


class RoleResolver:
    """
    Resolves role from explicit security context.
    NEVER inferred from text.
    """

    @staticmethod
    def resolve(ctx: SecurityContext) -> Role:
        if ctx.autonomous:
            return Role.AUTONOMOUS

        if ctx.actor == "system":
            return Role.SYSTEM

        if ctx.user_trust and ctx.user_trust.name == "ADMIN":
            return Role.ADMIN

        return Role.USER
