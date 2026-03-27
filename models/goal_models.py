from enum import Enum
from typing import List, Dict, Any
import time


class GoalStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Goal:
    def __init__(self, goal_id: str, description: str):
        self.goal_id = goal_id
        self.description = description
        self.status = GoalStatus.ACTIVE
        self.objectives: List[Dict[str, Any]] = []
        self.progress: float = 0.0
        self.created_at: float = time.time()

    def add_objective(self, text: str):
        self.objectives.append({"text": text, "done": False})
        self.update_progress()

    def update_progress(self):
        if not self.objectives:
            self.progress = 0.0
            return
        done = sum(1 for o in self.objectives if o["done"])
        self.progress = done / len(self.objectives)
        if self.progress >= 1.0:
            self.status = GoalStatus.COMPLETED