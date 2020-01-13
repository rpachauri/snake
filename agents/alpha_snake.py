import gym
import snake
import numpy as np
import copy
import random

class UCTNode():
  def __init__(self, state, child_priors, num_actions=5):
    self.state = state
    self.children = {}  # dictionary of moves to UCTNodes
    self.child_priors = child_priors
    self.child_total_value = np.zeros(num_actions, dtype=np.float32)
    self.child_num_visits = np.zeros(num_actions, dtype=np.float32)

  def __str__(self):
    return str(self.state) + "\n" + str(self._get_child_dict(0)) + "\n" + str(self._get_child_dict(1))

  def _get_child_dict(self, index):
    return {
      "Action": index,
      "Probability": self.child_priors[index],
      "Total value": self.child_total_value[index],
      "Number of Visits": self.child_num_visits[index]
    }

  def _child_Q(self):
    return self.child_total_value / (1 + self.child_num_visits)

  def _child_U(self, current_num_visits):
    return np.sqrt(current_num_visits) * self.child_priors / (1 + self.child_num_visits)

  def best_action(self, current_num_visits):
    '''Returns the best action based on each Q value and exploration value.
    
      Args:
      - current_num_visits is the number of times we have visited this node
    '''
    return np.argmax(self._child_Q() + self._child_U(current_num_visits))

  def update_tree(self, model, actor_critic, current_num_visits, eta=0.67):
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
    s, r, done, _ = model.step(action)  # Model is now at child.

    # Base case
    if action not in self.children:
        # Convert state to input that is readable for policy function.
        s = s[None, :]
        policy, value = actor_critic.policy_value(s)
        # EXPANSION
        self.children[action] = UCTNode(s, policy)
        # ROLLOUT
        # if done:
        #  value = 0
        self.child_total_value[action] = eta * value + (1 - eta) * r
        #if done or r == 50:
          #print("self.child_total_value[action] =", self.child_total_value[action])
        self.child_num_visits[action] += 1
        return value, 1

    if done:
        return 0, 1

    # Recursive case
    value, level = self.children[action].update_tree(
        model, actor_critic, self.child_num_visits[action])
    # BACKUP
    self.child_num_visits[action] += 1
    self.child_total_value[action] += value
    return value, 1 + level


class MCTS():
  '''
  '''

  def __init__(self, actor_critic, state):
    self.actor_critic = actor_critic
    # Convert state to input that is readable for policy function.
    state = state[None, :]
    policy, _ = self.actor_critic.policy_value(state)
    self.root = UCTNode(state, policy)
    self.root_num_visits = 0  # number of times we've visited the root node

  def action_value(self, num_rollouts, env):
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
        an action and an estimated value for the resulting state
    '''
    for r in range(num_rollouts):
        model = copy.deepcopy(env)
        _, depth = self.root.update_tree(model, self.actor_critic, self.root_num_visits)
        self.root_num_visits += 1
    # Select the action that had the most visits.
    action = np.argmax(self.root.child_num_visits)
    _, value = self.actor_critic.policy_value(self.root.state)

    # Move this tree to the state resulting from that action.
    self.root_num_visits = self.root.child_num_visits[action]
    self.root = self.root.children[action]

    return action, value


# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')

# Reset the environment to default beginning
# Default observation variable
obs = env.reset()
env.render()
score = 0
done = False

while not done:
  action = monte_carlo(obs, env, 1000)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))