import gym
import snake
import numpy as np
import copy
import random

class UCTNode():
  def __init__(self, child_priors, num_actions=5):
    self.children = {}  # dictionary of moves to UCTNodes
    self.child_priors = child_priors
    self.child_total_value = np.zeros(num_actions, dtype=np.float32)
    self.child_num_visits = np.zeros(num_actions, dtype=np.float32)

  def best_action(self, current_num_visits):
    '''Returns the best action based on each Q value and exploration value.
    
      Args:
      - current_num_visits is the number of times we have visited this node
    '''
    action_Q_value = self.child_total_value / (1 + self.child_num_visits)
    exploration_value = np.sqrt(current_num_visits) * self.child_priors / (1 + self.child_num_visits)
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

    # Base case
    if action not in self.children:
      # EXPANSION
      priors = np.ones(model.action_space) / model.action_space
      self.children[action] = UCTNode(priors)
      value = r + self.children[action].rollout(model, done)
      self.child_total_value[action] += value
      self.child_num_visits[action] += 1
      return value

    if done:
      return r

    # Recursive case
    value = self.children[action].update_tree(model, self.child_num_visits[action]) + r
    # BACKUP
    self.child_total_value[action] += value
    self.child_num_visits[action] += 1

    if current_num_visits - 1 != np.sum(self.child_num_visits[action]):
      print("current_num_visits =", current_num_visits, "while chidren visits =", np.sum(self.child_num_visits[action]))

    return value

  def rollout(self, model, done):
    value = 0
    while not done:
      action = random.randint(0, model.action_space - 1)
      _, reward, done, _ = model.step(action)
      value += reward
    return value


class UCT():
  '''
  '''

  def __init__(self, model):
    priors = np.ones(model.action_space) / model.action_space
    self.root = UCTNode(priors)
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
    action_values = np.divide(self.root.child_total_value, self.root.child_num_visits)
    print("action_values:", action_values)
    print("child_num_visits:", self.root.child_num_visits)
    child_num_visits = int(np.sum(self.root.child_num_visits))
    if self.root_num_visits - 1 != child_num_visits:
      print("there's a problem here...")
    action = np.argmax(self.root.child_num_visits)

    # Move this tree to the state resulting from that action.
    self.root_num_visits = self.root.child_num_visits[action]
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

uct = UCT(env)

while not done:
  action = uct.action(10, env)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))