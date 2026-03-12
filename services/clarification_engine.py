from services.preference_engine import PreferenceEngine


class ClarificationEngine:
    """
    Generates clarification questions when confidence or self-belief is low.
    Now preference-aware (8.5).
    """

    def __init__(self):
        self.preferences = PreferenceEngine()

    def generate(self, intent_type: str, capability: str | None = None) -> str:
        # Preference signal: how often user rejects this action
        rejection_bias = 0.0

        if capability:
            records = self.preferences.store.get_recent(
                key=f"confirm:{capability}",
                limit=5
            )
            if records:
                rejection_bias = 1.0 - sum(r["value"] for r in records) / len(records)

        # High rejection → more cautious wording
        if rejection_bias > 0.6:
            return (
                f"I’ve noticed you often want to be careful with '{capability}'. "
                f"Could you tell me exactly what you’d like me to do?"
            )

        # Existing behavior (UNCHANGED)
        if capability:
            return (
                f"I’m not fully confident performing '{capability}'. "
                f"Could you clarify what exactly you want me to do?"
            )

        if intent_type == "SEARCH":
            return "What exactly should I search for?"

        if intent_type == "OPEN_APP":
            return "Which application would you like me to open?"

        return "Could you clarify what you want me to do?"
