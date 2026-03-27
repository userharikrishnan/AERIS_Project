class SelfModel:
    def __init__(self):
        self.metrics = {
            "avg_confidence": 0.5,
            "failure_rate": 0.0,
            "successful_actions": 0,
            "failed_actions": 0
        }

    def update(self, confidence: float, success: bool):
        total = self.metrics["successful_actions"] + self.metrics["failed_actions"] + 1

        self.metrics["avg_confidence"] = (
            (self.metrics["avg_confidence"] * (total - 1)) + confidence
        ) / total

        if success:
            self.metrics["successful_actions"] += 1
        else:
            self.metrics["failed_actions"] += 1

        self.metrics["failure_rate"] = (
            self.metrics["failed_actions"] / total
        )

    def snapshot(self):
        return dict(self.metrics)
