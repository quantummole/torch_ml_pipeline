from .base_agent import BaseAgent

class RLAgent(BaseAgent):
    def update_policy(self):
        self.policy_engine.learn_from_memory(self.memory)
