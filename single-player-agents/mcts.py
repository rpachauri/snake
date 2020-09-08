import gym
import snake
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import copy
import random

class MCTSNode():
  def __init__(self, num_actions=5):
    self.children = {} # dictionary of moves to MCTSNodes
    # each action is on a range [0,1]
    # have to sum the probabilities and then divide by that sum
    self.action_priors = np.ones(num_actions)
    self.action_total_values = np.zeros(num_actions)
    self.action_visits = np.zeros(num_actions)

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
    action = random.choices(
      population=action_choices,
      weights=self.action_priors / np.sum(self.action_priors)
    )[0]
    _, r, done, _ = model.step(action)  # Model is now at child.
    
    if done:
      self.action_priors[action] = 0
      return r
    
    # Base case
    if action not in self.children:
      # EXPANSION
      self.children[action] = MCTSNode()
      value = r + self.children[action].rollout(model, done)
      self.action_total_values[action] += value
      self.action_visits[action] += 1
      return value
    
    

    # Recursive case
    value = self.children[action].update_tree(model) + r
    # BACKUP
    self.action_total_values[action] += value
    self.action_visits[action] += 1
    return value

  def rollout(self, model, done):
    value = 0
    while not done:
      action = random.randint(0, model.action_space - 1)
      _, r, done, _ = model.step(action)
      value += r
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
    action_values = np.divide(self.root.action_total_values, self.root.action_visits)
    print("action_values:", action_values)
    print("action_visits:", self.root.action_visits)

    # no available actions
    if np.sum(np.isnan(action_values)) == env.action_space:
      return 0

    action = np.nanargmax(action_values)

    # Move this tree to the state resulting from that action.
    self.root = self.root.children[action]
    return action

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')
action_choices = range(env.action_space)

# Reset the environment to default beginning
# Default observation variable
obs = env.reset()
env.render()
score = 0
done = False

mcts = MCTS()

while not done:
  action = mcts.action(600, env)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))