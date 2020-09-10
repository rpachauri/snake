import gym
from enum import IntEnum
import random
import numpy as np

from snake.envs.snake_env import SnakeEnv
from snake.envs.snake_env import Action

class Snake():
  def __init__(self, body):
    self.body = body
    self.direction = random.choice(list(Action))
    self.done = False

class MultiplayerSnakeEnv(gym.Env):
  """Implements the gym.Env interface.
  
  https://github.com/openai/gym/blob/master/gym/core.py.
  """
  # Dimension of the snake environment.
  action_space = len(Action)
  observation_space = 10
  num_agents = 1


  def __init__(self):
    # self.done = True
    pass

  def step(self, actions):
    """Accepts a list of actions and returns a tuple (observations, rewards, dones, info).

      Each snake takes a turn moving all at once.

      Args:
        actions list(int): a list of actions provided by the agent [0,4]
      Returns:
        observations list(object): agent's observation of the current environment
        rewards list(float): amount of reward returned to the agent after previous action
        dones list(bool): whether the episode has ended for the agent
        info (dict): contains auxiliary diagnostic information
    """
    rewards = []
    dones = []
    # TODO: iterate through list of snakes in random order
    for i in range(len(actions)):
      snake = self.snakes[i]
      rewards.append(self._step(actions[i], snake, i))
      dones.append(snake.done)
    return self.get_observation(), rewards, dones, None


  def _step(self, action, snake, snake_index):
    """Accepts an action and moves the snake.

      If a snake dies, its body remains.

      Args:
        action (int): an action provided by the agent [0,4]
      Returns:
        rewards (float): amount of reward returned to the agent after previous action
    """
    # If we are already in a terminal state, we cannot take a step.
    # Otherwise, this would affect other snakes.
    reward = SnakeEnv.DEFAULT_REWARD
    if snake.done:
      return reward

    assert len(snake.body) > 0

    current_direction = snake.direction
    action = self._adjust_action(list(Action)[action], current_direction)
    next_head = self._get_next_head(action, snake.body[0])
    snake.direction = action

    # move the tail.
    tail = snake.body[-1]
    snake.body = snake.body[:-1]

    # Hit wall.
    if self.hit_wall(next_head):
      snake.done = True
      return SnakeEnv.HIT_WALL
    
    # Hit body.
    if self.hit_body(next_head):
      snake.done = True
      return SnakeEnv.HIT_BODY

    # Move the head.
    snake.body.insert(0, next_head)

    # Consumed fruit.
    if next_head == self.fruit:
      self.fruit = None
      # extend the snake's length by adding the location where the tail used to be.
      snake.body.append(tail)
      
      # game is over.
      if self.has_won(snake_index):
        snake.done = True
        return SnakeEnv.CONSUMED_FRUIT

      # keep playing.
      self._set_next_fruit_location()
      return SnakeEnv.CONSUMED_FRUIT

    # Simply moved.
    return SnakeEnv.DEFAULT_REWARD


  def hit_wall(self, next_head):
    return next_head[0] < 0 or next_head[0] >= SnakeEnv.M or next_head[1] < 0 or next_head[1] >= SnakeEnv.N

  def hit_body(self, next_head):
    """Returns true if this "next_head" would hit any snake body (including itself).
    """
    for snake in self.snakes:
      if next_head in snake.body:
        return True
    return False

  def has_won(self, snake_index):
    """Returns true if the given snake has "won"

    Requires: all snakes must be done; otherwise returns false

    Definition of "won": this snake is the only snake not done
     - the environment may not be "done" but this is a sufficient definition of "won"
    """
    snake = self.snakes[snake_index]
    if snake.done:
      return False
    for other_snake in self.snakes:
      # If there exists another snake that is still playing
      if snake != other_snake and not other_snake.done:
        return False
    # This snake is not done. All other snakes are done.
    return True

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

  def _get_next_head(self, action, current_head):
    """Gets the next location the snake's head will be in given an action.

      Args:
        action: an action that works with Snake's game mechanics.
        current_head: a tuple of length 2 representing a location (row, col)
      Return:
        next_head: a tuple of length 2 representing a location (row, col)
    """
    row, col = current_head
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
    # self.done = False

    available_locs = set()
    # add all locations not next to a wall.
    for row in range(1, SnakeEnv.M - 1):
      for col in range(1, SnakeEnv.N - 1):
        available_locs.add((row, col))

    assert len(available_locs) >= MultiplayerSnakeEnv.num_agents

    self.snakes = []
    for _ in range(MultiplayerSnakeEnv.num_agents):
      head = random.choice(tuple(available_locs))
      self.snakes.append(Snake(body=[head]
        ))
      available_locs.remove(head)

    # print(self.snakes)
   
    # select a location for the fruit.
    self._set_next_fruit_location()

    return self.get_observation()

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
    # remove any location belonging to a snake
    for snake in self.snakes:
      for row, col in snake.body:
        available_locs.remove((row, col))
    # select a random available location
    self.fruit = random.choice(tuple(available_locs))

  def get_observation(self):
    observations = []

    for snake in self.snakes:
      observations.append(self._get_observation(snake))

    return np.array(observations)

  def _get_observation(self, snake):
    """
      Returns:
        obs: An ndarray. Each value is essentially a bitmap with a different purpose:
        
        obs[0-3] indicate which direction the snake is currently facing
        obs[4-7] indicate whether (1) or not (0) there is an obstruction in that direction
        obs[8] indicates the relative row location of the fruit:
          1 if the fruit is above the snake head
          0 if the fruit is on the same row as the snake head
          -1 if the fruit is below the snake head
        obs[9] indicates the relative column location of the fruit:
          1 if the fruit is left of the snake head
          0 if the fruit is on the same column as the snake head
          -1 if the fruit is right of the snake head
    """
    obs = np.zeros(MultiplayerSnakeEnv.observation_space, dtype=np.int32)

    if snake.done:
      return obs

    # at this point, len(self.snake.body) > 1 must be true.

    # indicate which direction the snake is currently facing.
    current_direction = snake.direction
    obs[int(current_direction)] = 1

    head = snake.body[0]

    offset = len(Action)
    # indicate whether (1) or not (0) there is an obstruction in that direction
    for action in Action:
      next_head = self._get_next_head(action, head)
      if self.hit_wall(next_head) or self.hit_body(next_head):
        obs[offset + int(action)] = 1

    
    # indicates the relative row location of the fruit.
    obs[8] = head[0] - self.fruit[0]
    # indicates the relative column location of the fruit.
    obs[9] = head[1] - self.fruit[1]
    
    return obs

  def _get_locations_as_2D_arrays(self):
    """
      Returns:
        2 MxN 2D numpy arrays. Each array is essentially a bitmap with a different purpose:
        1. Location of all body parts
        2. Location of fruit
    """
    body = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    for row, col in self.snake.body:
      body[row][col] = 1.

    fruit = np.zeros(shape=(SnakeEnv.M, SnakeEnv.N))
    if self.fruit is not None:
      fruit[self.fruit[0]][self.fruit[1]] = 1.

    return body, fruit


  def render(self, mode='human'):
    """Renders the current state of the environment.

    Args:
      mode (str): Supported modes: {'human'}
    """
    # body, fruit = self._get_locations_as_2D_arrays()

    horizontal_wall = self._create_horizontal_wall()
    print(horizontal_wall)
    
    # default every location to be an empty space
    char_locs = []
    for m in range(SnakeEnv.M):
      locs = []
      for n in range(SnakeEnv.N):
        locs.append(" ")
      char_locs.append(locs)

    # set the location of the fruit if there is one.
    if self.fruit is not None:
      char_locs[self.fruit[0]][self.fruit[1]] = "x"

    # set the location of each snake's body.
    directions = {Action.up: "^", Action.left: "<", Action.right: ">", Action.down: "v"}

    # print(self.snakes)
    for snake in self.snakes:
      # print(snake)
      if len(snake.body) > 0:
        head = snake.body[0]
        char_locs[head[0]][head[1]] = directions[snake.direction]
      if len(snake.body) > 1:
        for body_part in snake.body[1:]:
          char_locs[body_part[0]][body_part[1]] = "o"

    # plot the environment.
    for m in range(SnakeEnv.M):
      line = "|"
      for n in range(SnakeEnv.N):
        # loc = " "
        # if fruit[m][n] == 1:
        #   loc = "x"
        # if body[m][n] == 1:
        #   loc = "o"
        # if len(self.snake.body) > 0 and self.snake.body[0] == (m, n):
        #   loc = directions[self.snake.direction]
        line += char_locs[m][n]
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