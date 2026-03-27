import time
import threading

class Scheduler:
    def __init__(self, interval_seconds=5):
        self.interval = interval_seconds
        self.jobs = []
        self.running = False

    def add_job(self, fn):
        self.jobs.append(fn)

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            for job in self.jobs:
                try:
                    job()
                except Exception as e:
                    print(f"[Scheduler] Job error: {e}")
            time.sleep(self.interval)
