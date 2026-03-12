from services.goal_models import Goal, GoalStatus
import uuid

class GoalEngine:
    def __init__(self):
        self.goals = {}

    def create_goal(self, description: str) -> Goal:
        goal_id = str(uuid.uuid4())
        goal = Goal(goal_id, description)
        self.goals[goal_id] = goal
        return goal

    def get_goal(self, goal_id: str):
        return self.goals.get(goal_id)

    def list_active_goals(self):
        return [
            g for g in self.goals.values()
            if g.status == GoalStatus.ACTIVE
        ]

    def mark_objective_done(self, goal_id: str, index: int):
        goal = self.get_goal(goal_id)
        if not goal:
            return False
        try:
            goal.objectives[index]["done"] = True
            goal.update_progress()
            return True
        except IndexError:
            return False

    def pause_goal(self, goal_id: str):
        goal = self.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.PAUSED

    def resume_goal(self, goal_id: str):
        goal = self.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.ACTIVE
