from services.rbac_models import Role, Capability


class RoleRegistry:
    """
    Static RBAC policy definition.
    """

    ROLE_CAPABILITIES = {
        Role.USER: {
            Capability.CREATE_GOAL,
            Capability.EXECUTE_ACTION,
        },
        Role.ADMIN: {
            Capability.CREATE_GOAL,
            Capability.EXECUTE_ACTION,
            Capability.DELEGATE_ACTION,
            Capability.MODIFY_SYSTEM,
        },
        Role.AUTONOMOUS: {
            Capability.EXECUTE_ACTION,
            Capability.DELEGATE_ACTION,
        },
        Role.SYSTEM: {
            Capability.MODIFY_SYSTEM,
        },
    }

    @classmethod
    def has_capability(cls, role: Role, capability: Capability) -> bool:
        return capability in cls.ROLE_CAPABILITIES.get(role, set())
