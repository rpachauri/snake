import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

tf.compat.v1.disable_eager_execution()

class ReplayBuffer():
  def __init__(self, mem_size, input_dims):
    self.mem_size = mem_size
    # Memory counter keeps track of the position of our first unsaved memory.
    # We use this to insert new memories into the replay buffer.
    # When the replay buffer becomes full, then we start rewriting over the
    # earliest memroies.
    self.mem_cntr = 0

    # State memory keeps track of the states the agent sees at each timestep.
    self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
    # Memory for actions.
    self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
    # Memory for the state transition so the agent can understand the
    # consequences of its actions.
    self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

  def store_transition(self, state, action, reward, new_state, done):
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.new_state_memory[index] = new_state
    # When the episode is done, then the done flag is true (1).
    # We want to multiply the reward in the terminal state by 0.
    self.terminal_memory[index] = 1 - done

    self.mem_cntr += 1
  
  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size)
    batch = np.random.choice(max_mem, batch_size, replace=False)

    states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    new_states = self.new_state_memory[batch]
    terminals = self.terminal_memory[batch]

    return states, actions, rewards, new_states, terminals

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
  model = keras.Sequential([
      keras.layers.Dense(fc1_dims, activation='relu'),
      keras.layers.Dense(fc2_dims, activation='relu'),
      keras.layers.Dense(n_actions, activation=None),
  ])
  model.compile(
      optimizer=Adam(learning_rate=lr),
      loss='mean_squared_error', 
  )
  return model

class Agent():
  def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
               input_dims, epsilon_dec=1e-4, epsilon_end=0.01, mem_size=1000000,
               fname='dqn_model.h5'):
    self.action_space = [i for i in range(n_actions)]
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_dec = epsilon_dec
    self.eps_min = epsilon_end
    self.batch_size = batch_size
    self.model_file = fname
    self.memory = ReplayBuffer(mem_size, input_dims)
    self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)
  
  def store_transition(self, state, action, reward, new_state, done):
    self.memory.store_transition(state, action, reward, new_state, done)
  
  def choose_action(self, observation):
    if np.random.random() < self.epsilon:
      action = np.random.choice(self.action_space)
    else:
      state = np.array([observation])
      actions = self.q_eval.predict(state)
      action = np.argmax(actions)

    return action
  
  def learn(self):
    # We don't want to perform learning if we haven't filled up at least
    # batch_size of memories.
    if self.memory.mem_cntr < self.batch_size:
      return
    
    states, actions, rewards, new_states, dones = \
        self.memory.sample_buffer(self.batch_size)
    
    # Need to get the values for our TD Learning update.
    q_eval = self.q_eval.predict(states)
    q_next = self.q_eval.predict(new_states)

    q_target = np.copy(q_eval)
    batch_index = np.arange(self.batch_size, dtype=np.int32)

    q_target[batch_index, actions] = rewards + \
        self.gamma * np.max(q_next, axis=1) * dones
    
    self.q_eval.train_on_batch(states, q_target)
    self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
        self.eps_min else self.eps_min

import gym
import snake

env = gym.make('snake-v1')

lr = 0.0001
n_games = 500
agent = Agent(gamma=0.99, epsilon=1.0, lr=lr,
              input_dims=env.observation_space,
              n_actions=env.action_space, batch_size=64)
scores = []
eps_history = []

for i in range(n_games):
  done = False
  score = 0
  observation = env.reset()
  while not done:
    action = agent.choose_action(observation)
    # print(action)
    new_observation, reward, done, info = env.step(action)
    score += reward
    agent.store_transition(observation, action, reward, new_observation, done)
    observation = new_observation
    agent.learn()
  eps_history.append(agent.epsilon)
  scores.append(score)

  avg_score = np.mean(scores[-100:])

  print("episode:", i, "score %.2f" % score,
        "average score %.2f" % avg_score,
        "epsilon %.2f" % agent.epsilon)