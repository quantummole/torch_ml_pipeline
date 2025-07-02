from typing import List, Callable, Dict
import traceback
import time

class Goal:
    def __init__(self, name, description, priority, is_achieved=False):
        self.name = name
        self.priority = priority
        self.description = description
        self.is_achieved = is_achieved
    def achieve(self):
        self.is_achieved = True
    def reset(self):
        self.is_achieved = False

    def update_priority(self, new_priority):
        self.priority = new_priority
    def __str__(self):
        return f"Goal(name={self.name}, description={self.description}, is_achieved={self.is_achieved}), priority={self.priority})"    
    def __repr__(self):
        return self.__str__()

class GoalManager:
    def __init__(self):
        self.goals = {}
    def add_goal(self, goal: Goal):
        if goal.name not in self.goals:
            self.goals[goal.name] = goal
        else:
            raise ValueError(f"Goal '{goal.name}' already exists in GoalManager.")
    def get_goal(self, goal_name: str) -> Goal:
        if goal_name in self.goals:
            return self.goals[goal_name]
        else:
            raise ValueError(f"Goal '{goal_name}' not found in GoalManager.")
    def remove_goal(self, goal_name: str):
        if goal_name in self.goals:
            del self.goals[goal_name]
        else:
            raise ValueError(f"Goal '{goal_name}' not found in GoalManager.")
    def get_goals(self):
        return self.goals
    def get_achieved_goals(self):
        return [goal for goal_name, goal in self.goals.items() if goal.is_achieved]
    def get_unachieved_goals(self):
        return [goal for goal_name, goal in self.goals.items() if not goal.is_achieved]
    def __str__(self):
        return f"GoalManager(goals={self.goals})"
    def __repr__(self):
        return self.__str__()
    def reset_goals(self):
        for goal in self.goals:
            goal.reset()

class Action:
    def __init__(self, name, description, priority, function: Callable, parameters: Dict, terminal: bool = False):

        self.name = name
        self.description = description
        self.priority = priority
        self.function = function
        self.parameters = parameters
        self.terminal = terminal  # Indicates if the action is terminal (i.e., it ends the game or process)
    def execute(self, **args):
        self.is_executed = True
        try:
            output = self.function(**self.parameters, **args)
            return True, output
        except Exception as e:
            return False, traceback.format_exc()         
    def reset(self):
        self.is_executed = False
    def update_priority(self, new_priority):
        self.priority = new_priority    
    def __str__(self):
        return f"Action(name={self.name}, description={self.description}, priority={self.priority}, terminal={self.terminal}, executed={self.is_executed})"
    def __repr__(self):
        return self.__str__()
    
class ActionManager:
    def __init__(self):
        self.actions = {}
    def add_action(self, action: Action):
        if action.name not in self.actions:
            self.actions[action.name] = action
        else:
            raise ValueError(f"Action '{action.name}' already exists in ActionManager.")
    def get_action(self, action_name: str) -> Action:   
        if action_name in self.actions:
            return self.actions[action_name]
        else:
            raise ValueError(f"Action '{action_name}' not found in ActionManager.")
    def remove_action(self, action_name: str):
        if action_name in self.actions:
            del self.actions[action_name]
        else:
            raise ValueError(f"Action '{action_name}' not found in ActionManager.")
    def get_actions(self):
        return self.actions
    def get_executed_actions(self):
        return [action for action_name, action in self.actions.items() if action.is_executed]
    def get_unexecuted_actions(self):
        return [action for action_name, action in self.actions.items() if not action.is_executed]
    def reset_actions(self):
        for _, action in self.actions.items():
            action.reset()
    def __str__(self):
        return f"ActionManager(actions={self.actions})"
    def __repr__(self):
        return self.__str__()
    
class Memory:
    def __init__(self):
        self.memory = []
    def add_memory(self, memory_item):
        self.memory.append(memory_item)
    def get_memory(self):
        return self.memory
    def clear_memory(self):
        self.memory = []
    def __str__(self):
        return f"Memory(memory={self.memory})"
    def __repr__(self):
        return self.__str__()

class Environment:
    def execute_action(self, action: Action):
        """
        Executes the given action in the environment.
        Returns a tuple of (success_state, output).
        """
        if not isinstance(action, Action):
            raise ValueError("The provided action is not an instance of Action class.")
        if action.is_executed:
            raise ValueError(f"Action '{action.name}' has already been executed.")
        print(f"Executing action in environment: {action.name}")
        output = self.format_output(action.execute())
        return output
    def fetch_reward(self, output, reward_function: Callable):
        return reward_function(output)
    def format_output(self, output):
        """
        Formats the output from the action execution.
        This can be customized based on the environment's requirements.
        """
        return {
            'output': output[1], 
            'success': output[0],
            "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S%z", time.localtime())
        }

class Agent:
    def __init__(self, name: str, goal_manager: GoalManager, action_manager: ActionManager, memory: Memory):
        self.name = name
        self.goal_manager = goal_manager
        self.action_manager = action_manager
        self.memory = memory
        self.environment = None
        self.qvalues = dict([(a,0) for a,_ in self.action_manager.actions.items()])

    
    def set_environment(self, environment: Environment):
        self.environment = environment
    
    def execute_action(self, action_name: str):
        action = self.action_manager.get_action(action_name)
        if not action:
            raise ValueError(f"Action '{action_name}' not found in ActionManager.")
        success_state, output = self.environment.execute_action(action)
        if success_state:
            action.is_executed = True
            self.memory.add_memory(output)
        return success_state, output
    
    def __str__(self):
        return f"Agent(name={self.name}, goals={self.goal_manager.get_goals()}, actions={self.action_manager.get_actions()})"
    
    def __repr__(self):
        return self.__str__()