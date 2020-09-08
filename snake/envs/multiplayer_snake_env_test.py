import unittest
import gym
import snake
import numpy as np

from snake.envs.snake_env import Action
from snake.envs.snake_env import SnakeEnv
from snake.envs.multiplayer_snake_env import MultiplayerSnakeEnv
from snake.envs.multiplayer_snake_env import Snake


class TestMultiplayerSnakeEnv(unittest.TestCase):

  def setUp(self):
    SnakeEnv.M = 4
    SnakeEnv.N = 4
    self.env = MultiplayerSnakeEnv()
    self.env.reset()

  def test_facing_down_at_bottom_turn_left(self):
    self.env.snakes = [
      Snake(body=[(3,0), (2,0)]), # single snake of length two, facing down the first column, at the bottom
    ]
    self.env.fruit = (3,3)

    _, rewards, dones, _ = self.env.step([1]) # turn left
    self.assertEqual(rewards, [SnakeEnv.HIT_WALL])
    self.assertEqual(dones, [True])
    self.assertEqual(self.env.snakes[0].body, [(3,0)])
    self.assertEqual(self.env.fruit, (3,3))

  def test_has_won(self):
    # set all snakes except the first one to done
    for snake in self.env.snakes[1:]:
      snake.done = True
    self.assertTrue(self.env.has_won(0))


if __name__ == '__main__':
  unittest.main()