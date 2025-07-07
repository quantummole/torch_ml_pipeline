from .base_policy import PolicyEngine
import random

class QLearningPolicy(PolicyEngine):
    def __init__(self):
        self.q_table = {}

    def get_next_action(self, memory, goal_manager, action_manager, query):
        state = str(query)
        available = list(action_manager.actions.keys())
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in available}
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn_from_memory(self, memory):
        pass  # Implement Q-learning update rule here
