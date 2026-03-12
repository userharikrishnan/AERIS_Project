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

logger = setup_logger("aeris-core")

# -------------------------
# Core App Initialization
# -------------------------

app = FastAPI(title="Aeris Core", version="0.1")

app.include_router(core_router, prefix="/core")
app.include_router(agent_router, prefix="/agent")

# -------------------------
# Continuous Attention Setup
# -------------------------

signal_bus = SignalBus()
self_model = SelfModel()
reflection_engine = ReflectionEngine(self_model, signal_bus)
attention_loop = AttentionLoop(
    signal_bus=signal_bus,
    reflection_engine=reflection_engine,
    interval=0.5  # slow, safe, non-blocking
)

threading.Thread(
    target=attention_loop.start,
    daemon=True
).start()

logger.info("Aeris continuous attention loop started")

# -------------------------
# Health Check
# -------------------------

@app.get("/")
def health():
    return {
        "status": "Aeris Core online",
        "attention_loop": "running"
    }
