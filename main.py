import threading

from fastapi import FastAPI
from routes.core import core_router
from routes.agent import agent_router
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
        
        # FIX 1: Register Permission Engine
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

app = FastAPI(title="Aeris Core", version="0.1")

app.include_router(core_router, prefix="/core")
app.include_router(agent_router, prefix="/agent")

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

    logger.info("Aeris continuous attention loop started")

@app.on_event("shutdown")
def shutdown_event():
    try:
        system.attention_loop.stop()
        logger.info("Aeris attention loop stopped")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# -------------------------
# Health Check
# -------------------------

@app.get("/")
def health():
    return {
        "status": "Aeris Core online",
        "attention_loop": "running"
    }