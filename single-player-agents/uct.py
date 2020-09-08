import gym
import snake
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import copy
import random

class UCTNode():
  WINNING_VALUE = 10000000
  LOSING_VALUE = -10000000

  EXPLORATION_CONSTANT = 1

  def __init__(self, num_actions=4):
    self.children = {}  # dictionary of moves to UCTNodes
    self.action_priors = np.ones(num_actions, dtype=np.float32) / num_actions
    self.action_total_values = np.zeros(num_actions, dtype=np.float32)
    self.action_visits = np.zeros(num_actions, dtype=np.float32)

  def Q_value(self):
    return self.action_total_values / (1 + self.action_visits)

  def U_value(self, current_num_visits):
    return np.sqrt(current_num_visits * UCTNode.EXPLORATION_CONSTANT) * self.action_priors / (1 + self.action_visits)

  def best_action(self, current_num_visits):
    '''Returns the best action based on each Q value and exploration value.
    
      Args:
      - current_num_visits is the number of times we have visited this node
    '''
    exploitation_exploration = self.Q_value() + self.U_value(current_num_visits)
    return np.argmax(exploitation_exploration)

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
    self.action_visits[action] += 1

    if done:
      if model.has_won():
        self.action_total_values[action] = UCTNode.WINNING_VALUE
      else:
        self.action_total_values[action] = UCTNode.LOSING_VALUE
      return r

    # Base case
    if action not in self.children:
      # EXPANSION
      self.children[action] = UCTNode()

      # ROLLOUT
      value = r + self.children[action].rollout(model)
      self.action_total_values[action] += value
      return value

    # Recursive case
    value = r + self.children[action].update_tree(model, self.action_visits[action])

    # BACKUP
    self.adjust_action_value(action, value)

    return value

  def adjust_action_value(self, action, value):
    child = self.children[action]
    # child's actions all lead to losing/winning states
    if np.all(np.logical_or(
        child.action_total_values == UCTNode.WINNING_VALUE,
        child.action_total_values == UCTNode.LOSING_VALUE)):
      self.action_total_values[action] = np.max(child.action_total_values)
    elif (self.action_total_values[action] != UCTNode.WINNING_VALUE and
        self.action_total_values[action] != UCTNode.LOSING_VALUE):
      self.action_total_values[action] += value

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
  def __init__(self, num_actions=4):
    self.num_actions = num_actions
    self.root = UCTNode(num_actions)
    self.root_num_visits = 1  # number of times we've visited the root node

  def _perform_rollouts(self, num_rollouts, env):
    for _ in range(num_rollouts):
      self.root.update_tree(copy.deepcopy(env), self.root_num_visits)
      self.root_num_visits += 1
    # print("action_total_values:", self.root.action_total_values)
    # print("action_visits:", self.root.action_visits)
    # assert self.root_num_visits - 1 == np.sum(self.root.action_visits)

  def _select_action(self):
    # Select the action that had the most visits.
    #print("exploitation_exploration:", self.root.Q_value() + self.root.U_value(self.root_num_visits))
    #print("self.root.action_visits:", self.root.action_visits)
    action = np.argmax(self.root.action_visits)

    # Move this tree to the state resulting from that action.
    self.root_num_visits = self.root.action_visits[action]
    self.root = self.root.children[action] if action in self.root.children else UCTNode(self.num_actions)
    #print("type(action):", type(action))
    return action

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
    self._perform_rollouts(num_rollouts, env)
    return self._select_action()



# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('multiplayer-snake-v0')

# Reset the environment to default beginning
# Default observation variable
obs = env.reset()
env.render()
score = 0
done = False

uct = UCT()

while not done:
  action = uct.action(600, env)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))