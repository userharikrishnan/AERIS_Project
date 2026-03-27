from dataclasses import dataclass
from typing import List


@dataclass
class SandboxPolicy:
    """
    Defines the allowed blast radius for a tool.
    """
    tool: str
    allowed_paths: List[str]
    allowed_operations: List[str]
    rate_limit_per_minute: int
    dry_run_supported: bool
