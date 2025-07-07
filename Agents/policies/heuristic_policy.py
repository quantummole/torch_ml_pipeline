from .base_policy import PolicyEngine

class HeuristicPolicy(PolicyEngine):
    def get_next_action(self, memory, goal_manager, action_manager, query):
        unexecuted = action_manager.get_unexecuted_actions()
        if not unexecuted:
            return None
        return sorted(unexecuted, key=lambda a: a.priority)[0].name
