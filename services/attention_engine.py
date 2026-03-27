class AttentionDecision:
    def __init__(self, allow: bool, reason: str, requires_confirmation=False):
        self.allow = allow
        self.reason = reason
        self.requires_confirmation = requires_confirmation


class AttentionEngine:
    SENSITIVE_KEYWORDS = ["bank", "password", "payment", "delete"]

    def evaluate(self, text: str) -> AttentionDecision:
        lowered = text.lower()

        for word in self.SENSITIVE_KEYWORDS:
            if word in lowered:
                return AttentionDecision(
                    allow=False,
                    reason=f"Sensitive action detected: {word}",
                    requires_confirmation=True
                )

        if len(lowered.strip()) == 0:
            return AttentionDecision(False, "Empty input")

        return AttentionDecision(True, "Safe to proceed")
