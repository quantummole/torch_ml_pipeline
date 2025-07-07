import time

class Environment:
    def __init__(self, name, reward_function, goal_function):
        self.name = name
        self.reward_function = reward_function
        self.goal_function = goal_function

    def fetch_state(self):
        raise NotImplementedError

    def execute_action(self, action):
        curr_state = self.fetch_state()
        success, result = action.execute()
        new_state = self.fetch_state()
        return success, {
            'initial_state': curr_state,
            'new_state': new_state,
            'reward': self.reward_function(result),
            'goal': self.goal_function(result),
            'output': result,
            'success': success
        }

    def get_query(self):
        return None  # Replace with actual query extraction
