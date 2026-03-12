from models.preference import PreferenceSignal
from services.preference_store import PreferenceStore


class PreferenceEngine:
    """
    Learns user preferences passively.
    """

    def __init__(self, store: PreferenceStore | None = None):
        self.store = store or PreferenceStore()

    def record_confirmation(
        self,
        action: str,
        approved: bool
    ):
        """
        Learn from user confirmation behavior.
        """
        signal = PreferenceSignal(
            key=f"confirm:{action}",
            value=1.0 if approved else 0.0,
            confidence=0.7
        )
        self.store.save(signal)

    def record_style_preference(
        self,
        style_key: str,
        strength: float
    ):
        """
        Learn communication preferences (verbosity, caution).
        """
        signal = PreferenceSignal(
            key=style_key,
            value=strength,
            confidence=0.5
        )
        self.store.save(signal)
