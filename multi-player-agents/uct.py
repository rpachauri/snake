import gym
import snake
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import copy
import random

from snake.envs.multiplayer_snake_env import MultiplayerSnakeEnv

class UCTNode():
  WINNING_VALUE = 10000000
  LOSING_VALUE = -10000000

  EXPLORATION_CONSTANT = 1

  def __init__(self, num_agents, num_actions=4):
    self.children = {}  # dictionary of moves to UCTNodes
    self.action_priors = np.ones((num_agents, num_actions), dtype=np.float32) / num_actions
    self.action_total_values = np.zeros((num_agents, num_actions), dtype=np.float32)
    self.action_visits = np.zeros((num_agents, num_actions), dtype=np.float32)

  def Q_value(self):
    # ndarrays allow for elementwise multiplication: https://stackoverflow.com/a/40035266
    # assuming that property extends to elementwise division
    return self.action_total_values / (1 + self.action_visits)

  def U_value(self, current_num_visits):
    # ndarrays allow for elementwise multiplication: https://stackoverflow.com/a/40035266
    # assuming that property extends to elementwise division
    u_value_factor = np.sqrt(current_num_visits * UCTNode.EXPLORATION_CONSTANT)
    return u_value_factor * self.action_priors / (1 + self.action_visits)

  def best_actions(self, current_num_visits):
    '''Returns the best action based on each Q value and exploration value.
    
      Args:
      - current_num_visits is the number of times we have visited this node
    '''
    exploitation_exploration = self.Q_value() + self.U_value(current_num_visits)
    # return the best action for each agent
    return np.argmax(exploitation_exploration, axis=1)

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
    actions = self.best_actions(current_num_visits)
    _, rewards, dones, _ = model.step(actions)  # Model is now at child.


    assert len(actions) == len(rewards)
    assert len(actions) == len(dones)

    num_agents = len(actions)

    for agent in range(num_agents):
      action = actions[agent]
      self.action_visits[agent, action] += 1


    # all_done should be set to True if all agents are done
    all_done = True
    for agent in range(num_agents):

      if dones[agent]:
        action = actions[agent]
        # this line may cause problems because has_won() only gets called when dones[agent]
        # is true but dones[agent] must be false for model.has_won() to ever be true
        if model.has_won(agent): 
          self.action_total_values[agent, action] = UCTNode.WINNING_VALUE
        else:
          self.action_total_values[agent, action] = UCTNode.LOSING_VALUE
      else:
        all_done = False

    if all_done:
      return rewards

    # Base case
    if tuple(actions.tolist()) not in self.children:
      # EXPANSION
      self.children[tuple(actions.tolist())] = UCTNode(num_agents)

      # ROLLOUT
      values = self.children[tuple(actions.tolist())].rollout(model, num_agents)
      for agent in range(num_agents): # len(values) == num_agents must be true
        action = actions[agent]
        values[agent] += rewards[agent]
        self.action_total_values[agent, action] += values[agent]
      return values

    # Recursive case
    action_visits = []
    for agent in range(num_agents):
      action = actions[agent]
      action_visits.append(self.action_visits[agent, action])
    action_visits = np.array(action_visits).reshape((num_agents, 1))

    values = self.children[tuple(actions.tolist())].update_tree(model, action_visits)
    for agent in range(num_agents):
      values[agent] += rewards[agent]
    # value = r + self.children[actions].update_tree(model, self.action_visits[action])

    # BACKUP
    self.adjust_action_values(actions, values)

    return values

  def adjust_action_values(self, actions, values):
    child = self.children[tuple(actions.tolist())]
    # child's actions all lead to losing/winning states
    for agent in range(len(actions)):
      action = actions[agent]
      if np.all(np.logical_or(
          child.action_total_values[agent] == UCTNode.WINNING_VALUE,
          child.action_total_values[agent] == UCTNode.LOSING_VALUE)):
        # if all actions for the agent are winning or losing, set the action-value for the agent to be
        # the largest action-value.
        self.action_total_values[agent, action] = np.max(child.action_total_values[agent])
      elif (self.action_total_values[agent, action] != UCTNode.WINNING_VALUE and
          self.action_total_values[agent, action] != UCTNode.LOSING_VALUE):
        # if the action for the agent isn't already completely winning/losing, add the value estimate.
        self.action_total_values[agent, action] += values[agent]

  def rollout(self, model, num_agents):
    values = [0 for agent in range(num_agents)]
    done = False
    while not done:
      actions = np.random.randint(0, model.action_space, num_agents)
      _, rewards, dones, _ = model.step(actions)

      # default done to True
      done = True
      # if there is a single agent that is not done, the model is not done
      for d in dones:
        if not d:
          done = False

      for agent in range(num_agents):
        values[agent] += rewards[agent]
    return values


class UCT():
  '''
  '''
  def __init__(self, num_agents, num_actions=4):
    self.num_agents = num_agents
    self.num_actions = num_actions
    self.root = UCTNode(num_agents)
    self.root_num_visits = np.ones((num_agents, 1)) # number of times we've visited the root node
    self.check_rep()

  def _perform_rollouts(self, num_rollouts, env):
    for _ in range(num_rollouts):
      self.root.update_tree(copy.deepcopy(env), self.root_num_visits)
      for i in range(len(self.root_num_visits)):
        self.root_num_visits[i] += 1

  def _select_actions(self):
    # Select the action that had the most visits.
    actions = np.argmax(self.root.action_visits, axis=1)

    assert len(actions) == self.num_agents

    # Move this tree to the state resulting from that action.
    num_visits = []
    for agent in range(self.num_agents):
      action = actions[agent]
      num_visits.append(self.root.action_visits[agent, action])
    self.root_num_visits = np.array(num_visits).reshape(self.root_num_visits.shape)
    self.root = self.root.children[tuple(actions.tolist())] if tuple(actions.tolist()) in self.root.children else UCTNode(self.num_agents, self.num_actions)

    self.check_rep()
    
    return actions

  def actions(self, num_rollouts, env):
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
    return self._select_actions()

  def check_rep(self):
    assert self.root_num_visits.shape == (self.num_agents, 1)



# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('multiplayer-snake-v0')
num_agents = MultiplayerSnakeEnv.num_agents
print("num_agents:", num_agents)

# Reset the environment to default beginning
# Default observation variable
env.reset()
env.render()
scores = [0 for agent in range(num_agents)]
done = False

uct = UCT(num_agents)

while not done:
  actions = uct.actions(100, env)
  print("Taking actions: ", actions)
  
  _, rewards, dones, _ = env.step(actions)

  # Update scores.
  for i in range(len(rewards)): # len(rewards) == num_agents must be true
    scores[i] += rewards[i]

  # Update done.
  # default done to true.
  done = True
  # if there is a single snake that is not done, then the env is not done.
  for d in dones:
    if not d:
      done = False

  # Render the env
  env.render()

for agent in range(num_agents):
  score = scores[agent]
  print("Score for agent", agent, "is:", int(score))