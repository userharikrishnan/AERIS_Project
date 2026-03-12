import time

class AttentionLoop:
    def __init__(self, signal_bus, reflection_engine, interval=0.5):
        self.signal_bus = signal_bus
        self.reflection_engine = reflection_engine
        self.interval = interval
        self.running = False

    def start(self):
        self.running = True
        while self.running:
            signal = self.signal_bus.consume()

            if signal:
                self._handle_signal(signal)

            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def _handle_signal(self, signal):
        signal_type = signal["type"]

        if signal_type == "USER_INPUT":
            # User input already handled elsewhere
            return

        if signal_type == "REFLECTION_TRIGGER":
            # Let reflection engine decide what to do
            self.reflection_engine.reflect(
                confidence=signal["payload"].get("confidence", 0.5),
                success=signal["payload"].get("success", True)
            )

        if signal_type == "IDLE":
            # Low priority background reflection
            self.reflection_engine.reflect(
                confidence=0.5,
                success=True
            )
