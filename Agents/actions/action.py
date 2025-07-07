from typing import Callable
from dataclasses import dataclass, field
import traceback

@dataclass
class Action:
    name: str
    description: str
    priority: int
    function: Callable
    parameters: dict
    terminal: bool = False
    is_executed: bool = field(default=False, init=False)

    def execute(self, **args):
        self.is_executed = True
        try:
            output = self.function(**self.parameters, **args)
            return True, output
        except Exception:
            return False, traceback.format_exc()

    def reset(self):
        self.is_executed = False


class ActionManager:
    def __init__(self):
        self.actions = {}

    def add_action(self, action):
        self.actions[action.name] = action

    def get_action(self, name):
        return self.actions[name]

    def get_unexecuted_actions(self):
        return [action for action in self.actions.values() if not action.is_executed]

    def reset_actions(self):
        for action in self.actions.values():
            action.reset()
