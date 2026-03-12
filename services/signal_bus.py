from collections import deque

class SignalBus:
    def __init__(self):
        self.queue = deque()

    def emit(self, signal_type: str, payload=None):
        self.queue.append({
            "type": signal_type,
            "payload": payload
        })

    def consume(self):
        if self.queue:
            return self.queue.popleft()
        return None
