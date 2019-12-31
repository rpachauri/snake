# Gotta import gym!
import gym
import snake
import random

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')

def get_head(observation):
  """Given the current state of the snake environment, finds where the head of the snake is.
    Args:
      observation (numpy array): a 4xMxN ndarray.
    Returns:
      tuple: a tuple of length 2 representing the location of the head of the snake on a 2D grid.
  """
  one_hot_head = observation[0]
  for i in range(one_hot_head.shape[0]):
    for j in range(one_hot_head.shape[1]):
      if one_hot_head[i][j] == 1:
        return (i, j)
  return (-1, -1)

def get_action(head, M, N):
  """Given the location of the head of the snake, returns an action for that location.
    Args:
      head (tuple): a tuple of length 2 representing the location of the head of the snake on a 2D grid.
    Returns:
      int: an action
  """
  # top left corner goes down.
  if head == (0, 0):
    return 2
  # top row goes left.
  if head[0] == 0:
    return 1
  # last column goes up.
  if head[1] == N - 1:
    return 0

  # head is on an even column.
  if head[1] % 2 == 0:
    # turn right if at the bottom.
    if head[0] == M - 1:
      return 3
    # continue down otherwise.
    return 2

  #head is on an odd column.
  if head[0] == 1:
    # turn right if on the first row.
    # otherwise, will hit body.
    return 3
  # continue up otherwise.
  return 0

# Reset the environment to default beginning
# Default observation variable
obs = env.reset()
env.render()
done = False

fruit = 0
while not done:
  M = env.observation_space[1]
  N = env.observation_space[2]
  if N % 2 != 0: # This solution only works for when N % 2 == 0
    break
  action = get_action(get_head(obs), M, N)
  obs, reward, done, info = env.step(action)
  if reward == 50:
    fruit += 1

  print("Taking action: ", action)
  # Render the env
  env.render()
  if done:
    break
print("# of fruit consumed:", fruit)