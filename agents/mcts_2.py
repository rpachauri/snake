import gym
import snake
import numpy as np
import copy
import random

class MCTSNode():
  def __init__(self, num_actions=5):
    self.children = {}  # dictionary of moves to MCTSNodes
    self.child_total_value = np.zeros(num_actions, dtype=np.float32)
    self.child_num_visits = np.zeros(num_actions, dtype=np.float32)

  def update_tree(self, model):
    '''Performs MCTS for this node and all children of this node.
    
      Args:
       - model is a copy of an OpenAI gym environment.
      Requires:
       - model's state is at the current node
      Effects:
       - Creates a new leaf node.
       - Updates the value estimate for each node along this path
      Returns:
       - the value of the new leaf node
    '''
    # SELECTION
    action = random.randint(0, model.action_space - 1)
    _, r, done, _ = model.step(action)  # Model is now at child.

    # Base case
    if action not in self.children:
      # EXPANSION
      self.children[action] = MCTSNode()
      value = r + self.children[action].rollout(model, done)
      self.child_total_value[action] += value
      self.child_num_visits[action] += 1
      return value

    if done:
      return r

    # Recursive case
    value = self.children[action].update_tree(model) + r
    # BACKUP
    self.child_total_value[action] += value
    self.child_num_visits[action] += 1
    return value

  def rollout(self, model, done):
    value = 0
    while not done:
      action = random.randint(0, model.action_space - 1)
      _, reward, done, _ = model.step(action)
      value += reward
    return value


class MCTS():
  '''
  '''

  def __init__(self):
    self.root = MCTSNode()

  def action(self, num_rollouts, env):
    '''Returns an action.
      
      Args:
        num_rollouts: the number of rollouts we simulate
        env: the environment we are in
      Requires:
        - env must be copyable with copy.deepcopy() in order to perform rollouts
        - env is a deterministic environment.
        - action space of env is finite.
        - current state of env MUST match self.root
      Effects:
        - selects an action and moves this tree's root to the resulting state
        - the original environment is not modified
      Returns:
        the best action after performing num_rollouts simulations
    '''
    for r in range(num_rollouts):
      self.root.update_tree(copy.deepcopy(env))
    # Select the action that had the most visits.
    action_values = np.divide(self.root.child_total_value, self.root.child_num_visits)
    print("action_values:", action_values)
    print("child_num_visits:", self.root.child_num_visits)
    action = np.argmax(action_values)

    # Move this tree to the state resulting from that action.
    self.root = self.root.children[action]
    return action


# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')

# Reset the environment to default beginning
# Default observation variable
obs = env.reset()
env.render()
score = 0
done = False

mcts = MCTS()

while not done:
  action = mcts.action(1000, env)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))