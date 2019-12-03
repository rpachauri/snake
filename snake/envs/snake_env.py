import gym
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum
import random
import numpy as np
import tensorflow as tf

class Action(Enum):
    up = 0
    left = 1
    down = 2
    right = 3

class SnakeEnv(gym.Env):
  """Implements the gym.Env interface.
  
  https://github.com/openai/gym/blob/master/gym/core.py.
  """
  # Dimension of the snake environment.
  M = 10
  N = 20

  action_space = len(Action)
  observation_space = None


  def __init__(self):
    
    pass

  def step(self, action):
    """Accepts an action and returns a tuple (observation, reward, done, info).
      Args:
        action (object): an action provided by the agent
      Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended
        info (dict): contains auxiliary diagnostic information
    """
    pass

  def reset(self):
    """Resets the state of the environment and returns an initial observation.
      Returns:
        observation (object): the initial observation.
    """
    # locations are represented by tuples of length 2.
    # head is anywhere on the grid and not next to a wall.
    head = (random.randrange(SnakeEnv.M - 2) + 1, random.randrange(SnakeEnv.N - 2) + 1)
    self.body = [head]

    # create a random walk from that head to select the location for the neck.
    direction = random.choice(list(Action))
    if direction == Action.up:
      self.body.append((head[0] - 1, head[1]))
    elif direction == Action.down:
      self.body.append((head[0] + 1, head[1]))
    elif direction == Action.left:
      self.body.append((head[0], head[1] - 1))
    else: # direction == Action.right
      self.body.append((head[0], head[1] + 1))
   
    # select a location for the fruit.
    fruit_locs = self._get_next_fruit_location()

    # convert the body into 3 2D tensors that can be interpreted as an observation.
    #   1. location of head
    #   2. location of body adjacent to head (this should provide direction)
    #   3. all body parts
    head, neck, body = self._get_body_locations()
    return tf.stack([fruit_locs, head, neck, body])

  def _get_next_fruit_location(self):
    """Selects a location for a fruit anywhere on the grid that is not on the snake's body.
      Requires:
        There should be no fruit on the grid.
      Returns:
        An MxN tensor with a 1 placed at the location of the fruit.
    """
    # TODO: create a set of all available locations and choose from that set.
    # https://stackoverflow.com/questions/15837729/random-choice-from-set-python
    available_locs = set()
    # add all locations
    for row in range(SnakeEnv.M):
      for col in range(SnakeEnv.N):
        available_locs.add((row, col))
    # remove any location belonging to a body
    for row, col in self.body:
      available_locs.remove((row, col))
    # select a random available location
    row, col = random.choice(tuple(available_locs))
    fruit_locs = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    fruit_locs[row][col] = 1.
    return fruit_locs

  def _get_body_locations(self):
    """
      Returns:
        3 MxN tensors. Each tensor servers a different purpose:
        1. Location of head
        2. Location of neck
        3. Location of all body parts
    """
    head = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    head[self.body[0][0], self.body[0][1]] = 1.

    neck = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    neck[self.body[1][0], self.body[1][1]] = 1.

    body = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    for row, col in self.body:
      body[row][col] = 1.

    return head, neck, body


  def render(self, mode='human'):
    pass

  def close(self):
    pass