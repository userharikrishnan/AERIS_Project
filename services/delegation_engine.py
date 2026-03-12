from services.trust_models import TrustLevel
from services.sandbox_manager import SandboxManager, SandboxViolation


class DelegationDecision:
    def __init__(self, allowed: bool, device_id=None, reason=None):
        self.allowed = allowed
        self.device_id = device_id
        self.reason = reason


class DelegationEngine:
    """
    Chooses execution device AND enforces sandbox constraints (9.0).
    """

    def __init__(self, device_registry):
        self.device_registry = device_registry
        self.sandbox = SandboxManager()

    def choose_device(
        self,
        required_capability: str,
        min_trust: TrustLevel,
        params: dict | None = None
    ) -> DelegationDecision:
        """
        Delegation with sandbox validation.
        """

        # -------------------------
        # Sandbox validation (9.0)
        # -------------------------
        try:
            self.sandbox.validate(
                tool=required_capability,
                params=params or {}
            )
        except SandboxViolation as e:
            return DelegationDecision(
                False,
                reason=f"Sandbox violation: {str(e)}"
            )

        # -------------------------
        # Existing behavior (UNCHANGED)
        # -------------------------
        candidates = self.device_registry.capable_devices(required_capability)

        if not candidates:
            return DelegationDecision(
                False,
                reason="No capable device available"
            )

        # Pick highest trust device
        candidates.sort(key=lambda d: d.trust.value, reverse=True)
        device = candidates[0]

        if device.trust.value < min_trust.value:
            return DelegationDecision(
                False,
                reason="No device meets trust requirement"
            )

        return DelegationDecision(
            True,
            device_id=device.device_id
        )
