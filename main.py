import os
import threading

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.core import core_router
from routes.agent import agent_router
from routes.ws import ws_router
from utils.logger import setup_logger

# Continuous attention imports
from services.signal_bus import SignalBus
from services.self_model import SelfModel
from services.reflection_engine import ReflectionEngine
from services.attention_loop import AttentionLoop
from services.attention_engine import AttentionEngine
from services.nlp_processor import NLPProcessor
from services.memory_engine import MemoryEngine
from services.reasoning_engine import ReasoningEngine
from services.command_engine import CommandEngine
from services.permission_engine import PermissionEngine

from services.focus_engine import FocusEngine
from services.priority_models import PriorityLevel
from services.goal_engine import GoalEngine
from services.planner import Planner
from services.scheduler import Scheduler
from services.audit_logger import AuditLogger
from services.confirmation_engine import ConfirmationEngine
from services.preference_engine import PreferenceEngine
from services.dry_run_simulator import DryRunSimulator
from services.rollback_engine import RollbackEngine
from services.explanation_engine import ExplanationEngine
from services.delegation_engine import DelegationEngine
from services.device_registry import DeviceRegistry

# ==============================
# NEW services
# ==============================
from services.tool_dispatcher import ToolDispatcher
from services.smart_confirmation import SmartConfirmationEngine
from services.session_engine import SessionEngine
from services.plan_executor import PlanExecutor
from services.app_registry import AppRegistry
from services.learning_engine import LearningEngine
from services.update_manager import UpdateManager

logger = setup_logger("aeris-core")


# -------------------------
# System Container
# -------------------------

class AerisSystem:
    def __init__(self):
        # -------------------------
        # Core shared components
        # -------------------------
        self.signal_bus = SignalBus()
        self.self_model = SelfModel()

        # -------------------------
        # Cognitive engines
        # -------------------------
        self.attention_engine = AttentionEngine()
        self.nlp_processor = NLPProcessor()
        self.memory_engine = MemoryEngine()

        # Register Permission Engine
        self.permission_engine = PermissionEngine()

        self.reflection_engine = ReflectionEngine(
            self.self_model,
            self.signal_bus
        )

        self.reasoning_engine = ReasoningEngine(system=self)

        self.command_engine = CommandEngine()

        # -------------------------
        # Governance engines
        # -------------------------
        self.focus_engine = FocusEngine(max_active_items=1)
        self.audit_logger = AuditLogger()
        self.confirmation_engine = ConfirmationEngine()
        self.preference_engine = PreferenceEngine()
        self.dry_run_simulator = DryRunSimulator()
        self.rollback_engine = RollbackEngine()
        self.explanation_engine = ExplanationEngine()

        # -------------------------
        # Device & Delegation
        # -------------------------
        self.device_registry = DeviceRegistry()
        self.delegation_engine = DelegationEngine(self.device_registry)

        # Register default device
        from services.trust_models import TrustLevel
        from services.device_registry import Device
        self.device_registry.register(
            Device(
                device_id="local",
                name="Primary Machine",
                trust=TrustLevel.ADMIN,
                capabilities=["browser", "filesystem", "app"]
            )
        )

        # -------------------------
        # Goal & Autonomy engines
        # -------------------------
        self.goal_engine = GoalEngine()
        self.planner = Planner()
        self.scheduler = Scheduler(interval_seconds=10)
        self.goal_executor = None  # Will be initialized if needed

        # -------------------------
        # NEW: Execution Pipeline
        # -------------------------
        self.tool_dispatcher = ToolDispatcher()
        self.plan_executor = PlanExecutor(self.tool_dispatcher)
        self.smart_confirmation = SmartConfirmationEngine()
        self.session_engine = SessionEngine()

        # -------------------------
        # NEW: Dynamic App Registry
        # -------------------------
        self.app_registry = AppRegistry()

        # -------------------------
        # NEW: Learning Engine
        # -------------------------
        self.learning_engine = LearningEngine(
            app_registry=self.app_registry,
            smart_confirmation=self.smart_confirmation
        )

        # -------------------------
        # NEW: Update Manager
        # -------------------------
        self.update_manager = UpdateManager()

        # -------------------------
        # Background systems
        # -------------------------
        self.attention_loop = AttentionLoop(
            signal_bus=self.signal_bus,
            reflection_engine=self.reflection_engine,
            interval=0.5
        )


# -------------------------
# Core App Initialization
# -------------------------

app = FastAPI(
    title="AERIS Core",
    version="0.3.0",
    description="Autonomous Execution and Reasoning Intelligence System"
)

app.include_router(core_router, prefix="/core")
app.include_router(agent_router, prefix="/agent")
app.include_router(ws_router)  # WebSocket: /ws

# Mount static UI files (must be AFTER API routes)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Initialize ONE global system
system = AerisSystem()

# Expose system globally for other modules
app.state.system = system


# -------------------------
# FastAPI Lifecycle Events
# -------------------------

@app.on_event("startup")
def startup_event():
    threading.Thread(
        target=system.attention_loop.start,
        daemon=True
    ).start()
    logger.info("AERIS v0.2 — Autonomous Execution and Reasoning Intelligence System")
    logger.info("Attention loop started")
    logger.info(f"Tools registered: {list(system.tool_dispatcher._tools.keys())}")


@app.on_event("shutdown")
def shutdown_event():
    try:
        system.attention_loop.stop()
        logger.info("AERIS attention loop stopped")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# -------------------------
# Health & Status
# -------------------------

@app.get("/", response_class=None)
def health():
    from fastapi.responses import FileResponse
    import os
    idx = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.isfile(idx):
        return FileResponse(idx)
    return {
        "status": "AERIS Online",
        "version": "0.3.0",
        "tools": system.tool_dispatcher.list_capabilities(),
        "ui": "not found — check static/index.html"
    }


@app.get("/status")
def full_status():
    """Comprehensive system status."""
    return {
        "aeris": "online",
        "version_info": system.update_manager.get_version_info(),
        "session": system.session_engine.get_session_summary(),
        "tools": system.tool_dispatcher.list_capabilities(),
        "learning": system.learning_engine.get_personal_knowledge_summary(),
        "auto_approvals": system.smart_confirmation.get_all_approvals(),
        "known_apps": system.app_registry.list_known_apps()[:20]
    }


# -------------------------
# Learning & Teaching Endpoints
# -------------------------

@app.post("/teach/app")
def teach_app(payload: dict):
    """Teach AERIS a new app mapping: { app_name, path }"""
    app_name = payload.get("app_name", "")
    path = payload.get("path", "")
    if not app_name or not path:
        return {"error": "Provide app_name and path"}
    system.learning_engine.learn_app_mapping(app_name, path)
    return {"taught": True, "app_name": app_name, "path": path}


@app.post("/teach/browser")
def teach_browser(payload: dict):
    """Tell AERIS your preferred browser: { browser_name }"""
    browser = payload.get("browser_name", "")
    if not browser:
        return {"error": "Provide browser_name"}
    system.learning_engine.learn_browser_preference(browser)
    return {"taught": True, "preferred_browser": browser}


@app.post("/correct")
def correct_aeris(payload: dict):
    """Correct AERIS's interpretation: { original_input, original_intent, corrected_intent }"""
    system.learning_engine.learn_correction(
        original_input=payload.get("original_input", ""),
        original_intent=payload.get("original_intent", ""),
        corrected_intent=payload.get("corrected_intent", ""),
        correction_detail=payload.get("detail", "")
    )
    return {"correction_recorded": True}


# -------------------------
# Update Management
# -------------------------

@app.get("/updates/check")
def check_updates():
    """Check for pending model updates in the updates/ folder."""
    return system.update_manager.check_for_updates()


@app.post("/updates/apply")
def apply_update(payload: dict):
    """Apply a model update: { file: 'updates/aeris_v1.0.zip' }"""
    update_file = payload.get("file", "")
    if not update_file:
        pending = system.update_manager.check_for_updates()
        if pending["pending"]:
            update_file = pending["pending"][0]["path"]
        else:
            return {"error": "No update file specified and no pending updates found"}
    return system.update_manager.apply_update(update_file)
