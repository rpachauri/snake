import gym
from enum import IntEnum
import random
import numpy as np

class Action(IntEnum):
    up = 0
    left = 1
    down = 2
    right = 3

class SnakeEnv(gym.Env):
  """Implements the gym.Env interface.
  
  https://github.com/openai/gym/blob/master/gym/core.py.
  """
  # Dimension of the snake environment.
  M = 20
  N = 20

  action_space = len(Action)
  observation_space = np.array((4, M, N))

  HIT_WALL = -1
  HIT_BODY = -1
  CONSUMED_FRUIT = 10
  DEFAULT_REWARD = 0


  def __init__(self):
    self.done = True
    pass

  def step(self, action):
    """Accepts an action and returns a tuple (observation, reward, done, info).

      Args:
        action (int): an action provided by the agent [0,4]
      Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended
        info (dict): contains auxiliary diagnostic information
    """
    # if we are already in a terminal state, we cannot take a step
    assert not self.done

    reward = SnakeEnv.DEFAULT_REWARD

    current_direction = self._current_direction()
    action = self._adjust_action(list(Action)[action], current_direction)
    next_head = self._get_next_head(action)

    # move the tail.
    tail = self.body[-1]
    self.body = self.body[:-1]

    # Hit wall.
    if self.hit_wall(next_head):
      self.done = True
      return self._get_observation(), SnakeEnv.HIT_WALL, self.done, None
    
    # Hit body.
    if self.hit_body(next_head):
      self.done = True
      return self._get_observation(), SnakeEnv.HIT_BODY, self.done, None

    # Move the head.
    self.body.insert(0, next_head)

    # Consumed fruit.
    if next_head == self.fruit:
      self.fruit = None
      # extend the snake's length by adding the location where the tail used to be.
      self.body.append(tail)
      
      # game is over.
      if self.has_won():
        self.done = True
        return self._get_observation(), SnakeEnv.CONSUMED_FRUIT, self.done, None

      # keep playing.
      self._set_next_fruit_location()
      return self._get_observation(), SnakeEnv.CONSUMED_FRUIT, self.done, None

    # Simply moved.
    return self._get_observation(), SnakeEnv.DEFAULT_REWARD, self.done, None

  def hit_wall(self, next_head):
    return next_head[0] < 0 or next_head[0] >= SnakeEnv.M or next_head[1] < 0 or next_head[1] >= SnakeEnv.N

  def hit_body(self, next_head):
    return next_head in self.body

  def has_won(self):
    """ Returns true if this env is in a state that would be considered "won"

    Definition of "won": the length of the snake's body equals SnakeEnv.M * SnakeEnv.N
    """
    return len(self.body) == SnakeEnv.M * SnakeEnv.N


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
    self.done = False

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
    # create a set of all available locations and choose from that set.
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
        A 4xMxN tensor. Each tensor is essentially a bitmap with a different purpose:
        1. Location of head
        2. Location of neck
        3. Location of all body parts
        4. Location of fruit
    """
    head, neck, body, fruit = self._get_locations_as_2D_arrays()
    return np.stack([head, neck, body, fruit])

  def _get_locations_as_2D_arrays(self):
    """
      Returns:
        4 MxN 2D numpy arrays. Each array is essentially a bitmap with a different purpose:
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

    return head, neck, body, fruit


  def render(self, mode='human'):
    """Renders the current state of the environment.

    Args:
      mode (str): Supported modes: {'human'}
    """
    horizontal_wall = self._create_horizontal_wall()
    print(horizontal_wall)

    directions = {Action.up: "^", Action.left: "<", Action.right: ">", Action.down: "v"}
    head, _, body, fruit = self._get_locations_as_2D_arrays()
    for m in range(SnakeEnv.M):
      line = "|"
      for n in range(SnakeEnv.N):
        loc = " "
        if fruit[m][n] == 1:
          loc = "x"
        if body[m][n] == 1:
          loc = "o"
        if head[m][n] == 1 and len(self.body) > 1:
          loc = directions[self._current_direction()]
        line += loc
      print(line + "|")
    print(horizontal_wall)

  def _create_horizontal_wall(self):
    """Used to help render the top or bottom wall in human mode.
    Returns:
      str: a string that can be printed when rendering in human mode.
    """
    wall = "*"
    for i in range(SnakeEnv.N):
      wall += "="
    return wall + "*"

  def close(self):
    pass