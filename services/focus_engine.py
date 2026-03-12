from services.priority_models import PriorityLevel
import time


class FocusItem:
    def __init__(self, source: str, priority: PriorityLevel, timestamp=None):
        self.source = source  # "user", "goal", "system"
        self.priority = priority
        self.timestamp = timestamp or time.time()


class FocusEngine:
    """
    Manages what Aeris should focus on right now.
    """

    def __init__(self, max_active_items=1):
        self.max_active_items = max_active_items
        self.queue = []

    def submit(self, item: FocusItem):
        self.queue.append(item)
        self._reorder()

    def _reorder(self):
        # Highest priority first, oldest wins ties
        self.queue.sort(
            key=lambda x: (-x.priority.value, x.timestamp)
        )

    def allow_execution(self, source: str) -> bool:
        """
        Only allow execution if this source is within attention budget.
        """
        active = self.queue[: self.max_active_items]
        return any(item.source == source for item in active)

    def consume(self, source: str):
        """
        Remove an item once handled.
        """
        self.queue = [i for i in self.queue if i.source != source]
