class SecurityContext:
    """
    Passed implicitly through the system.
    NEVER inferred. Always explicit.
    """

    def __init__(
        self,
        actor: str,
        user_trust=None,
        device_id=None,
        autonomous=False
    ):
        self.actor = actor
        self.user_trust = user_trust
        self.device_id = device_id
        self.autonomous = autonomous

    def to_dict(self):
        return {
            "actor": self.actor,
            "user_trust": str(self.user_trust),
            "device_id": self.device_id,
            "autonomous": self.autonomous
        }
