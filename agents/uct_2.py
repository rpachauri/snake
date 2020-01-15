import gym
import snake
import numpy as np
import copy
import random

class UCTNode():
  def __init__(self, num_actions=5):
    self.children = {}  # dictionary of moves to UCTNodes
    self.action_priors = np.ones(num_actions, dtype=np.float32)
    self.action_total_values = np.zeros(num_actions, dtype=np.float32)
    self.action_visits = np.zeros(num_actions, dtype=np.float32)

  def priors(self):
    return self.action_priors / np.sum(self.action_priors)

  def best_action(self, current_num_visits):
    '''Returns the best action based on each Q value and exploration value.
    
      Args:
      - current_num_visits is the number of times we have visited this node
    '''
    action_Q_value = self.action_total_values / (1 + self.action_visits)
    exploration_value = np.sqrt(current_num_visits) * self.priors() / (1 + self.action_visits)
    return np.argmax(action_Q_value + exploration_value)

  def update_tree(self, model, current_num_visits):
    '''Performs UCT for this node and all children of this node.
    
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
    action = self.best_action(current_num_visits)
    _, r, done, _ = model.step(action)  # Model is now at child.

    if done:
      self.children[action] = UCTNode()
      self.action_priors[action] = 0
      return r

    # Base case
    if action not in self.children:
      # EXPANSION
      self.children[action] = UCTNode()
      value = r + self.children[action].rollout(model)
      self.action_total_values[action] += value
      self.action_visits[action] += 1
      return value

    # Recursive case
    value = self.children[action].update_tree(model, self.action_visits[action]) + r
    # BACKUP
    self.action_total_values[action] += value
    self.action_visits[action] += 1
    return value

  def rollout(self, model):
    value = 0
    done = False
    while not done:
      action = random.randint(0, model.action_space - 1)
      _, r, done, _ = model.step(action)
      value += r
    return value


class UCT():
  '''
  '''

  def __init__(self):
    self.root = UCTNode()
    self.root_num_visits = 1  # number of times we've visited the root node

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
    for _ in range(num_rollouts):
      self.root_num_visits += 1
      self.root.update_tree(copy.deepcopy(env), self.root_num_visits)
    # Select the action that had the most visits.
    action_values = np.divide(self.root.action_total_values, self.root.action_visits)
    print("action_values:", action_values)
    print("action_visits:", self.root.action_visits)
    #assert self.root_num_visits - 1 == np.sum(self.root.action_visits)

    action = np.argmax(self.root.action_visits)

    # Move this tree to the state resulting from that action.
    self.root_num_visits = self.root.action_visits[action]
    self.root = self.root.children[action]
    return action

'''
'''
# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')

# Reset the environment to default beginning
# Default observation variable
obs = env.reset()
env.render()
score = 0
done = False

uct = UCT()

while not done:
  action = uct.action(1000, env)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))
'''
'''