from dataclasses import dataclass, field, asdict
from typing import Dict, List
import time


@dataclass
class SelfAction:
    action: str
    outcome: str
    confidence: float
    timestamp: float = time.time()


@dataclass
class CapabilityBelief:
    capability: str
    success_rate: float
    last_updated: float = time.time()


@dataclass
class IdentityState:
    """
    Persistent self-model.
    """
    name: str = "Aeris"
    action_history: List[SelfAction] = field(default_factory=list)
    capabilities: Dict[str, CapabilityBelief] = field(default_factory=dict)

    # -------------------------
    # Serialization helpers
    # -------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "action_history": [asdict(a) for a in self.action_history],
            "capabilities": {
                k: asdict(v) for k, v in self.capabilities.items()
            }
        }

    @staticmethod
    def from_dict(data: dict) -> "IdentityState":
        state = IdentityState(name=data.get("name", "Aeris"))

        for a in data.get("action_history", []):
            state.action_history.append(SelfAction(**a))

        for k, v in data.get("capabilities", {}).items():
            state.capabilities[k] = CapabilityBelief(**v)

        return state
