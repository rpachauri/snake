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

def returns_advantages(rewards, dones, values, next_value, gamma=0.99):
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

env = gym.make('snake-v0')
actor_critic = ActorCritic(input_shape=env.observation_space, num_actions=env.action_space)
actor_critic.compile(
  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
  loss=[tf.losses.SparseCategoricalCrossentropy(from_logits=True), 'mean_squared_error']
)
train_actor_critic(actor_critic, updates=10000000)