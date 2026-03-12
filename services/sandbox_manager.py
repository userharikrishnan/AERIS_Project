import time
from services.tool_policies import TOOL_POLICIES


class SandboxViolation(Exception):
    pass


class SandboxManager:
    """
    Enforces sandbox rules before tool execution.
    """

    def __init__(self):
        self._usage = {}

    def validate(
        self,
        tool: str,
        params: dict
    ):
        policy = TOOL_POLICIES.get(tool)
        if not policy:
            raise SandboxViolation(f"No sandbox policy for tool '{tool}'")

        self._check_rate_limit(tool, policy)
        self._check_paths(tool, policy, params)
        self._check_operations(tool, policy, params)

        return True

    # -------------------------
    # Internal checks
    # -------------------------

    def _check_rate_limit(self, tool, policy):
        now = time.time()
        window = self._usage.setdefault(tool, [])
        window[:] = [t for t in window if now - t < 60]

        if len(window) >= policy.rate_limit_per_minute:
            raise SandboxViolation("Rate limit exceeded")

        window.append(now)

    def _check_paths(self, tool, policy, params):
        if not policy.allowed_paths:
            return

        path = params.get("path")
        if not path:
            return

        for allowed in policy.allowed_paths:
            if path.startswith(allowed):
                return

        raise SandboxViolation(f"Path '{path}' outside sandbox")

    def _check_operations(self, tool, policy, params):
        operation = params.get("operation")
        if not operation:
            return

        if operation not in policy.allowed_operations:
            raise SandboxViolation(
                f"Operation '{operation}' not permitted for {tool}"
            )
