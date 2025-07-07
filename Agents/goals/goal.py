from dataclasses import dataclass, field

@dataclass
class Goal:
    name: str
    description: str
    priority: int
    achieved: bool = False

    def achieve(self):
        self.achieved = True

    def reset(self):
        self.achieved = False


class GoalManager:
    def __init__(self):
        self.goals = {}

    def add_goal(self, goal):
        self.goals[goal.name] = goal

    def get_goal(self, name):
        return self.goals[name]

    def reset_goals(self):
        for goal in self.goals.values():
            goal.reset()
