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
    nothing = 4

class SnakeEnv(gym.Env):
  """Implements the gym.Env interface.
  
  https://github.com/openai/gym/blob/master/gym/core.py.
  """
  # Dimension of the snake environment.
  M = 10
  N = 20

  action_space = list(Action)
  observation_space = (4, M, N)

  HIT_WALL = -100
  HIT_BODY = -100
  CONSUMED_FRUIT = 50
  DEFAULT_REWARD = -1


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
    reward = SnakeEnv.DEFAULT_REWARD
    done = False

    current_direction = self._current_direction()
    action = self._adjust_action(action, current_direction)
    next_head = self._get_next_head(action)

    # move the tail.
    tail = self.body[-1]
    self.body = self.body[:-1]

    # Hit wall.
    if next_head[0] < 0 or next_head[0] >= SnakeEnv.M or next_head[1] < 0 or next_head[1] >= SnakeEnv.N:
      return self._get_observation(), SnakeEnv.HIT_WALL, True, None
    
    # Hit body.
    if next_head in self.body:
      return self._get_observation(), SnakeEnv.HIT_BODY, True, None

    # move the head.
    self.body.insert(0, next_head)

    # Consumed fruit.
    if next_head == self.fruit:
      self.fruit = None
      # extend the snake's length by adding the location where the tail used to be.
      self.body.append(tail)

      done = False
      # game is over.
      if len(self.body) == SnakeEnv.M * SnakeEnv.N:
        done = True

      return self._get_observation(), SnakeEnv.CONSUMED_FRUIT, done, None

    # Simply moved.
    return self._get_observation(), SnakeEnv.DEFAULT_REWARD, False, None


  def _current_direction(self):
    """Gets the current direction of the snake depending on its head and neck.

      Returns:
        one action out of [up, left, down, right]
    """
    head = self.body[0]
    neck = self.body[1]

    row_diff = head[0] - neck[0]
    if row_diff == -1:
      return Action.up
    if row_diff == 1:
      return Action.down

    col_diff = head[1] - neck[1]
    if col_diff == -1:
      return Action.left
    # col_diff == 1 must be true
    return Action.right

  def _adjust_action(self, action, current_direction):
    """The action the user makes may not be compatible with game mechanics,
      so this function adjusts user-input.

      Args:
        action: a user-inputted action
        current_direction: the current direction of the snake
      Returns:
        an action that will work with Snake's game mechanics (either action or current_direction).
    """
    # A non-action results in a forward-action.
    if action == Action.nothing:
      return current_direction
    # An action in the opposite direction that the snake faces results in a forward-action.
    if ((action == Action.up and current_direction == Action.down) or
        (action == Action.left and current_direction == Action.right) or
        (action == Action.down and current_direction == Action.up) or
        (action == Action.right and current_direction == Action.left)
      ):
      return current_direction
    return action

  def _get_next_head(self, action):
    """Gets the next location the snake's head will be in given an action.

      Args:
        action: an action that works with Snake's game mechanics.
      Return:
        a tuple of length 2 representing a location (row, col)
    """
    row, col = self.body[0]
    if action == Action.up:
      return (row - 1, col)
    if action == Action.down:
      return (row + 1, col)
    if action == Action.left:
      return (row, col - 1)
    # action == Action.right must be true
    return (row, col + 1)

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
    self._set_next_fruit_location()

    return self._get_observation()

  def _set_next_fruit_location(self):
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
    self.fruit = random.choice(tuple(available_locs))

  def _get_observation(self):
    """
      Returns:
        4 MxN tensors. Each tensor servers a different purpose:
        1. Location of head
        2. Location of neck
        3. Location of all body parts
        4. Location of fruit
    """
    head = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    head[self.body[0][0], self.body[0][1]] = 1.

    neck = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    if len(self.body) > 1:
      neck[self.body[1][0], self.body[1][1]] = 1.

    body = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    for row, col in self.body:
      body[row][col] = 1.

    fruit = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    if self.fruit is not None:
      fruit[self.fruit[0]][self.fruit[1]] = 1.

    return tf.stack([head, neck, body, fruit])


  def render(self, mode='human'):
    pass

  def close(self):
    pass