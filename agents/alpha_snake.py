import gym
import snake
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import copy
import random

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

class UCTNode():
  WINNING_VALUE = 10000000
  LOSING_VALUE = -10000000

  EXPLORATION_CONSTANT = 10

  def __init__(self, priors, num_actions=5): # state, priors, num_actions=5):
    #self.state = state
    self.children = {}  # dictionary of moves to UCTNodes
    self.action_priors = priors
    self.action_total_values = np.zeros(num_actions, dtype=np.float32)
    self.action_visits = np.zeros(num_actions, dtype=np.float32)

  def best_action(self, current_num_visits):
    '''Returns the best action based on each Q value and exploration value.
    
      Args:
      - current_num_visits is the number of times we have visited this node
    '''
    q_value = self.action_total_values
    u_value = np.sqrt(current_num_visits * UCTNode.EXPLORATION_CONSTANT) * self.action_priors
    return np.argmax((q_value + u_value) / (1 + self.action_visits))

  def update_tree(self, model, actor_critic, current_num_visits):
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
    self.action_visits[action] += 1

    if done:
      if model.has_won():
        self.action_total_values[action] = UCTNode.WINNING_VALUE
      else:
        self.action_total_values[action] = UCTNode.LOSING_VALUE
      return r

    # Base case
    if action not in self.children:
      # Convert state to input that is readable for policy function.

      # EXPANSION
      s = s[None, :]
      policy, next_state_value = actor_critic.policy_value(s)
      action_value = r + next_state_value
      self.children[action] = UCTNode(policy) # UCTNode(s, policy)

      # ROLLOUT
      self.action_total_values[action] += action_value
      return action_value

    # Recursive case
    action_value = r + self.children[action].update_tree(
        model, actor_critic, self.action_visits[action])
    
    # BACKUP
    self.adjust_action_value(action, action_value)

    return action_value

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


class UCT():
  '''
  '''

  def __init__(self, actor_critic):
    self.actor_critic = actor_critic
    self.root = None
    self.root_num_visits = 0  # number of times we've visited the root node

  def _reset(self, state):
    # Convert state to input that is readable for policy function.
    policy, _ = self.actor_critic.policy_value(state[None, :])
    self.root = UCTNode(policy)
    self.root_num_visits = 1

  def action_value(self, num_rollouts, state, env):
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
    if self.root is None:
      self._reset(state)
    for _ in range(num_rollouts):
      model = copy.deepcopy(env)
      self.root.update_tree(model, self.actor_critic, self.root_num_visits)
      self.root_num_visits += 1
    # Select the action that had the most visits.
    action = np.argmax(self.root.action_visits)
    _, value = self.actor_critic.policy_value(state[None, :])

    # Move this tree to the state resulting from that action.
    self.root_num_visits = self.root.action_visits[action]
    self.root = self.root.children[action] if action in self.root.children else None

    return action, value

class ActorCritic(tf.keras.Model):
  def __init__(self, input_shape, num_actions):
    super().__init__('mlp_policy')
    self.batch1 = kl.BatchNormalization()
    self.conv1 = kl.Conv2D(32, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape)
    self.conv2 = kl.Conv2D(64, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape)
    self.flatten = kl.Flatten()
    self.hidden1 = kl.Dense(128, activation='relu')
    self.hidden2 = kl.Dense(128, activation='relu')
    self.value = kl.Dense(1, name='value')
    # logits are unnormalized log probabilities
    self.logits = kl.Dense(num_actions, name='policy_logits')

  def call(self, inputs):
    # inputs is a numpy array, convert to Tensor
    x = tf.convert_to_tensor(inputs)
    x = self.batch1(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)

    # separate hidden layers from the same input tensor
    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def policy_value(self, obs):
    # executes call() under the hood
    logits, value = self.predict(obs)
    policy = tf.nn.softmax(logits)
    return np.squeeze(policy), np.squeeze(value, axis=-1)

def returns_advantages(rewards, dones, values, next_value, gamma=0.9):
  # next_value is the bootstrap value estimate of a future state (the critic)
  returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
  # returns are calculated as discounted sum of future rewards
  for t in reversed(range(rewards.shape[0])):
    returns[t] = rewards[t] + gamma * returns[t+1] * (1-dones[t])
  returns = returns[:-1]
  # advantages are returns - baseline, value estimates in our case
  advantages = returns - values
  return returns, advantages

def train(uct, batch_sz=16, updates=100):
  actor_critic = uct.actor_critic
  # storage helpers for a single batch of data
  actions = np.empty((batch_sz,), dtype=np.int32)
  rewards, dones, values = np.empty((3, batch_sz))
  states = np.empty((batch_sz, env.observation_space[0], env.observation_space[1], env.observation_space[2]))
  
  # training loop: collect samples, send to optimizer, repeat updates times
  ep_rews = [0.0]
  state = env.reset()

  for update in range(updates):
    for step in range(batch_sz):
      states[step] = state.copy()
      actions[step], values[step] = uct.action_value(100, state, env)
      state, rewards[step], dones[step], _ = env.step(actions[step])

      ep_rews[-1] += rewards[step]
      if dones[step]:
        ep_rews.append(0.0)
        state = env.reset()
        print("Episode: %03d, Reward: %03d" % (len(ep_rews)-1, ep_rews[-2]))

    _, next_value = actor_critic.policy_value(state[None, :])
    returns, advantages = returns_advantages(rewards, dones, values, next_value)
    # performs a full training step on the collected batch
    losses = actor_critic.train_on_batch(states, [actions, returns], sample_weight={"output_1":advantages})
    print("[%d/%d] Losses: %s" % (update+1, updates, losses))
  print("Finished training")

env = gym.make('snake-v0')
actor_critic = ActorCritic(input_shape=env.observation_space, num_actions=env.action_space)
actor_critic.compile(
  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
  loss=[tf.losses.SparseCategoricalCrossentropy(from_logits=True), 'mean_squared_error']
)
uct = UCT(actor_critic)
train(uct, updates=1000)

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
  action, _ = uct.action_value(600, env)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))