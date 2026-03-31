"""
AERIS Launcher
==============
Single entry point: starts the uvicorn server in a background thread,
then launches the CustomTkinter native Windows UI in the main thread.

Usage:
    python launcher.py

The browser UI remains available at http://localhost:8000
"""
import sys
import os
import threading
import time

# ── ensure project root is on the path ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────
# Server startup
# ─────────────────────────────────────────────────────────────────

def _run_server():
    """Start uvicorn (non-reload mode for embedded use)."""
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
        access_log=False,
    )


def _wait_for_server(timeout: int = 20) -> bool:
    """Poll /status until the server responds."""
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get("http://localhost:8000/status", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  A E R I S  — Launching...               ║")
    print("╚══════════════════════════════════════════╝")

    # 1. Start server in a daemon thread so it dies when UI closes
    server_thread = threading.Thread(target=_run_server, daemon=True, name="aeris-server")
    server_thread.start()

    print("[AERIS] Server starting on http://localhost:8000 ...")

    # 2. Wait until server is UP (max 20 s)
    if _wait_for_server():
        print("[AERIS] Server ready ✓")
    else:
        print("[AERIS] WARNING: Server did not respond in time. UI will retry automatically.")

    # 3. Launch the native UI in the main thread (tkinter requirement)
    from ui.aeris_window import AerisApp
    app = AerisApp()
    app.run()   # blocks until window is closed → daemon server thread also dies
