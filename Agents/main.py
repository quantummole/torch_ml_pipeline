from agents.heuristic_agent import HeuristicAgent
from actions.action import Action, ActionManager
from goals.goal import Goal, GoalManager
from memory.memory import Memory
from policies.heuristic_policy import HeuristicPolicy
from environment.environment import Environment

if __name__ == '__main__':
    gm = GoalManager()
    am = ActionManager()
    mem = Memory()
    policy = HeuristicPolicy()

    def dummy_reward(output): return 1.0
    def dummy_goal(output): return 'dummy_goal'

    env = Environment("TestEnv", dummy_reward, dummy_goal)
    agent = HeuristicAgent("TestAgent", gm, am, mem, policy, env)

    am.add_action(Action("SampleAction", "Description", 1, lambda: "done", {}))
    gm.add_goal(Goal("SampleGoal", "Description", 1))

    agent.run()
