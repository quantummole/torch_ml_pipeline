from .base_policy import PolicyEngine
import random

class LLMPolicy(PolicyEngine):
    def __init__(self, llm_call, max_memory=10):
        """
        llm_call: Callable that takes a prompt and returns a string output (next action name)
        """
        self.llm_call = llm_call
        self.max_memory = max_memory

    def get_next_action(self, memory, goal_manager, action_manager, query):
        goal_descriptions = [f"- {goal.name}: {goal.description}" for goal in goal_manager.goals.values() if not goal.achieved]
        actions = action_manager.get_unexecuted_actions()
        action_descriptions = [f"- {action.name}: {action.description}" for action in actions]

        last_memories = memory[max(0, len(memory)-self.max_memory):] if len(memory) > 0 else []
        memory_snippets = [f"{i+1}. {str(mem)}" for i, mem in enumerate(last_memories)]

        prompt = f"""
        Given the following goals:
{chr(10).join(goal_descriptions)}

Available actions:
{chr(10).join(action_descriptions)}

Recent memory:
{chr(10).join(memory_snippets)}

Query from environment: {query}

Choose the most appropriate action to fulfill the goal(s) and the query. Respond only with the action name.
"""

        action_name = self.llm_call(prompt).strip()
        if action_name in action_manager.actions:
            return action_name
        return None
