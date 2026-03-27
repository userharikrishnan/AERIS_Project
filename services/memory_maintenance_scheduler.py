import threading
import time
from services.memory_consolidator import MemoryConsolidator
from services.memory_decay_engine import MemoryDecayEngine
from services.memory_engine import MemoryEngine


class MemoryMaintenanceScheduler:
    """
    Background memory maintenance.
    Runs consolidation & decay during idle time.
    """

    def __init__(
        self,
        memory_engine: MemoryEngine,
        consolidate_interval: int = 300,   # 5 minutes
        decay_interval: int = 3600          # 1 hour
    ):
        self.memory_engine = memory_engine
        self.consolidator = MemoryConsolidator(memory_engine)
        self.decay_engine = MemoryDecayEngine(memory_engine)

        self.consolidate_interval = consolidate_interval
        self.decay_interval = decay_interval

        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop = True

    def _run(self):
        last_consolidate = 0
        last_decay = 0

        while not self._stop:
            now = time.time()

            # 🧠 Consolidation cycle
            if now - last_consolidate >= self.consolidate_interval:
                try:
                    self.consolidator.consolidate()
                except Exception as e:
                    print(f"[MemoryScheduler] Consolidation error: {e}")
                last_consolidate = now

            # 🧹 Decay cycle
            if now - last_decay >= self.decay_interval:
                try:
                    self.decay_engine.decay()
                except Exception as e:
                    print(f"[MemoryScheduler] Decay error: {e}")
                last_decay = now

            time.sleep(5)
