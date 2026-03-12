from services.sandbox_policy import SandboxPolicy


TOOL_POLICIES = {
    "filesystem": SandboxPolicy(
        tool="filesystem",
        allowed_paths=[
            "/tmp",
            "/home",
            "C:\\Users"  # windows-safe root
        ],
        allowed_operations=[
            "read",
            "write",
            "list"
        ],
        rate_limit_per_minute=20,
        dry_run_supported=True
    ),

    "browser": SandboxPolicy(
        tool="browser",
        allowed_paths=[],
        allowed_operations=[
            "open",
            "search"
        ],
        rate_limit_per_minute=30,
        dry_run_supported=False
    ),

    "app": SandboxPolicy(
        tool="app",
        allowed_paths=[],
        allowed_operations=[
            "open",
            "close"
        ],
        rate_limit_per_minute=10,
        dry_run_supported=False
    )
}
