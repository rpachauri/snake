import gym
import snake
import numpy as np
import copy
import random

def monte_carlo(state, gym_env, num_rollouts):
  '''Performs the Monte Carlo Algorithm.
    Args:
      state: the state of the environment
      gym_env: the environment we're in
      num_rollouts: the number of rollouts we perform
    Requires:
      1. gym_env must be copyable
      2. gym_env must be in the given state
    Returns:
      action (int): the best action after performing the Monte Carlo Algorithm.
  '''
  action_returns = np.zeros(gym_env.action_space)
  action_visits = np.zeros(gym_env.action_space)

  for i in range(num_rollouts):
    model = copy.deepcopy(gym_env)
    # random.randint is inclusive on low and high.
    a0 = random.randint(0, model.action_space - 1)
    obs, reward, done, _ = model.step(a0)
    score = reward

    action_visits[a0] += 1

    while not done:
      action = random.randint(0, model.action_space - 1)
      obs, reward, done, _ = model.step(action)
      score += reward

    action_returns[a0] = ((action_returns[a0] * (action_visits[a0] - 1) / action_visits[a0]) +
      score * 1 / action_visits[a0])

  print("action_returns:",  action_returns)
  return np.argmax(action_returns)

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
  # take a random action 
  # random.randint is inclusive on low and high.
  action = monte_carlo(obs, env, 1000)
  print("Taking action: ", action)
  
  obs, reward, done, info = env.step(action)
  score += reward
  # Render the env
  env.render()

print("Score:", int(score))