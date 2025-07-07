class BaseAgent:
    def __init__(self, name, goal_manager, action_manager, memory, policy_engine, environment):
        self.name = name
        self.goal_manager = goal_manager
        self.action_manager = action_manager
        self.memory = memory
        self.policy_engine = policy_engine
        self.environment = environment

    def get_next_action(self, query):
        return self.policy_engine.get_next_action(self.memory, self.goal_manager, self.action_manager, query)

    def execute_action(self, action_name):
        return self.environment.execute_action(self.action_manager.get_action(action_name))

    def run(self):
        while True:
            query = self.environment.get_query()
            action_name = self.get_next_action(query)
            if not action_name:
                break
            output = self.execute_action(action_name)
            self.memory.add_memory(output)



