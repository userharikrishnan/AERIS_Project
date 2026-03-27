from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from services.sandbox_manager import SandboxManager


@dataclass
class ExecutionPreview:
    action: str
    params: Dict[str, Any]
    affected_resources: List[str]
    risk_level: str
    reversible: bool
    notes: str
    rollback_params: Optional[Dict[str, Any]] = None  # ✅ ADDED


class DryRunSimulator:
    """
    Produces a non-executing, deterministic preview of an action.
    """

    def __init__(self):
        self.sandbox = SandboxManager()

    def simulate(self, action: str, params: dict) -> ExecutionPreview:
        # Defense-in-depth validation
        self.sandbox.validate(action, params)

        if action == "filesystem":
            return self._simulate_filesystem(params)

        if action == "browser":
            return self._simulate_browser(params)

        if action == "app":
            return self._simulate_app(params)

        # Fallback (non-reversible)
        return ExecutionPreview(
            action=action,
            params=params,
            affected_resources=[],
            risk_level="unknown",
            reversible=False,
            rollback_params=None,
            notes="No simulator available for this action"
        )

    # --------------------------------------------------
    # Tool simulators
    # --------------------------------------------------

    def _simulate_filesystem(self, params: dict) -> ExecutionPreview:
        path = params.get("path", "<unknown>")
        operation = params.get("operation", "read")

        reversible = operation in {"write", "delete"}

        rollback_params = None
        if reversible:
            rollback_params = {
                "operation": "restore_backup",
                "path": path
            }

        return ExecutionPreview(
            action="filesystem",
            params=params,
            affected_resources=[path],
            risk_level="medium" if operation != "read" else "low",
            reversible=reversible,
            rollback_params=rollback_params,
            notes=f"Filesystem {operation} on {path}"
        )

    def _simulate_browser(self, params: dict) -> ExecutionPreview:
        url = params.get("url", "<unknown>")

        rollback_params = {
            "operation": "navigate_back"
        }

        return ExecutionPreview(
            action="browser",
            params=params,
            affected_resources=[url],
            risk_level="low",
            reversible=True,
            rollback_params=rollback_params,
            notes=f"Browser navigation to {url}"
        )

    def _simulate_app(self, params: dict) -> ExecutionPreview:
        app = params.get("name", "<unknown>")

        # App launches are intentionally NOT reversible
        return ExecutionPreview(
            action="app",
            params=params,
            affected_resources=[app],
            risk_level="medium",
            reversible=False,
            rollback_params=None,
            notes=f"Launching application {app}"
        )
