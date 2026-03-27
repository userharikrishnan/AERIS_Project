from services.memory_engine import MemoryEngine
from services.identity_engine import IdentityEngine


class SelfContinuityEngine:
    """
    Bridges memory and identity.
    """

    def __init__(
        self,
        memory_engine: MemoryEngine,
        identity_engine: IdentityEngine
    ):
        self.memory = memory_engine
        self.identity = identity_engine

    def recall_self_actions(self, action: str):
        """
        Check if *I* have done this before.
        """
        return [
            a for a in self.identity.state.action_history
            if a.action == action
        ]

    def assess_self_confidence(self, action: str) -> float:
        belief = self.identity.state.capabilities.get(action)
        return belief.success_rate if belief else 0.5
