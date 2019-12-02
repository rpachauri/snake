import gym
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum

class Action(Enum):
    up = 0
    left = 1
    down = 2
    right = 3

class SnakeEnv(gym.Env):
  """Implements the gym.Env interface: https://github.com/openai/gym/blob/master/gym/core.py.
  """

  def __init__(self):
    self.action_space = len(Action)
    self.observation_space = None
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
    pass

  def render(self, mode='human'):
    pass

  def close(self):
    pass