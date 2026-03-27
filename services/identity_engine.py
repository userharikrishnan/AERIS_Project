from models.identity import IdentityState, SelfAction, CapabilityBelief
from services.identity_store import IdentityStore


class IdentityEngine:
    """
    Maintains self-identity across actions.
    """

    def __init__(self, store: IdentityStore | None = None):
        self.store = store or IdentityStore()
        loaded = self.store.load()
        self.state = loaded if loaded else IdentityState()

    # -------------------------
    # Existing behavior (UNCHANGED)
    # -------------------------
    def record_action(self, action: str, success: bool, confidence: float):
        outcome = "success" if success else "failure"

        self.state.action_history.append(
            SelfAction(
                action=action,
                outcome=outcome,
                confidence=confidence
            )
        )

        self._update_capability(action, success)
        self._persist()

    def _update_capability(self, action: str, success: bool):
        belief = self.state.capabilities.get(action)

        if not belief:
            belief = CapabilityBelief(
                capability=action,
                success_rate=1.0 if success else 0.0
            )
            self.state.capabilities[action] = belief
        else:
            belief.success_rate = (
                belief.success_rate * 0.8 + (1.0 if success else 0.0) * 0.2
            )
            belief.last_updated = belief.last_updated

    def get_identity_summary(self):
        return {
            "name": self.state.name,
            "known_capabilities": {
                k: v.success_rate
                for k, v in self.state.capabilities.items()
            },
            "actions_taken": len(self.state.action_history)
        }

    # -------------------------
    # Persistence (NEW)
    # -------------------------
    def _persist(self):
        self.store.save(self.state)
