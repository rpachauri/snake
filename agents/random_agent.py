# Gotta import gym!
import gym
import snake
import random

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')

# Reset the environment to default beginning
# Default observation variable
obs = env.reset()
env.render()

for i in range(1000):
  # take a random action 
  # random.randint is inclusive on low and high.
  action = random.randint(0, env.action_space - 1)
  obs, reward, done, info = env.step(action)

  print("Taking action: ", action)
  # Render the env
  env.render()
  if done:
    break