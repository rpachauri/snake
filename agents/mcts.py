import tensorflow as tf
import gym
import snake
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import copy

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())

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

class ActorCritic(tf.keras.Model):
  def __init__(self, input_shape, num_actions):
    super().__init__('mlp_policy')
    self.conv1 = kl.Conv2D(16, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape)
    self.conv2 = kl.Conv2D(16, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape)
    self.flatten = kl.Flatten()
    self.hidden1 = kl.Dense(256, activation='relu')
    self.hidden2 = kl.Dense(256, activation='relu')
    self.value = kl.Dense(1, name='value')
    # logits are unnormalized log probabilities
    self.logits = kl.Dense(num_actions, name='policy_logits')

  def call(self, inputs):
    # inputs is a numpy array, convert to Tensor
    x = tf.convert_to_tensor(inputs)
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
        ''' Returns an action.
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

def train_actor_critic(actor_critic, batch_sz=16, updates=100):
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
      policy, values[step] = actor_critic.policy_value(state[None, :])
      actions[step] = np.random.choice(a=np.arange(len(policy)), p=policy)
      state, rewards[step], dones[step], _ = env.step(actions[step])
      #print(actions[step])
      #print(values[step])
      #env.render()

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

def train_mcts(actor_critic, batch_sz=32, updates=200, num_rollouts=30):
  # storage helpers for a single batch of data
  actions = np.empty((batch_sz,), dtype=np.int32)
  rewards, dones, values = np.empty((3, batch_sz))
  states = np.empty((batch_sz, env.observation_space[0], env.observation_space[1], env.observation_space[2]))

  # training loop: collect samples, send to optimizer, repeat updates times
  ep_rews = [0.0]
  state = env.reset()
  mcts = MCTS(actor_critic, state)
  env.render()

  for update in range(updates):
    for step in range(batch_sz):
      states[step] = state.copy()
      actions[step], values[step] = mcts.action_value(num_rollouts, env)
      state, rewards[step], dones[step], _ = env.step(actions[step])
      #print(actions[step])
      #print(values[step])
      env.render()

      ep_rews[-1] += rewards[step]
      if dones[step]:
        ep_rews.append(0.0)
        state = env.reset()
        mcts = MCTS(actor_critic, state)
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
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss=[tf.losses.SparseCategoricalCrossentropy(from_logits=True), 'mean_squared_error']
)
train_actor_critic(actor_critic, updates=500)
train_mcts(actor_critic, num_rollouts=50)