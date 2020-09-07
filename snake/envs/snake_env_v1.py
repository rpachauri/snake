import gym
from gym import error, spaces, utils
from gym.utils import seeding
from enum import IntEnum
import random
import numpy as np

from snake.envs.snake_env import SnakeEnv
from snake.envs.snake_env import Action

class SnakeEnvV1(SnakeEnv):
  """Implements the gym.Env interface.
  
  https://github.com/openai/gym/blob/master/gym/core.py.
  """
  # Dimension of the snake environment.
  observation_space = 10

  def __init__(self):
    super().__init__()
    pass

  def _get_observation(self):
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
    obs = np.zeros(SnakeEnvV1.observation_space, dtype=np.int32)

    # indicate which direction the snake is currently facing.
    if len(self.body) > 1:
      current_direction = super()._current_direction()
      obs[int(current_direction)] = 1

    offset = len(Action)
    # indicate whether (1) or not (0) there is an obstruction in that direction
    for action in Action:
      next_head = super()._get_next_head(action)
      if super().hit_wall(next_head) or super().hit_body(next_head):
        obs[offset + int(action)] = 1

    head = self.body[0]
    # indicates the relative row location of the fruit.
    obs[8] = head[0] - self.fruit[0]
    # indicates the relative column location of the fruit.
    obs[9] = head[1] - self.fruit[1]
    
    return obs