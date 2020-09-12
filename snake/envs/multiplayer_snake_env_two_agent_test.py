import unittest
import gym
import snake
import numpy as np

from snake.envs.snake_env import Action
from snake.envs.snake_env import SnakeEnv
from snake.envs.multiplayer_snake_env import MultiplayerSnakeEnv
from snake.envs.multiplayer_snake_env import Snake


class TestTwoAgentMultiplayerSnakeEnv(unittest.TestCase):

  def setUp(self):
    SnakeEnv.M = 4
    SnakeEnv.N = 4
    MultiplayerSnakeEnv.num_agents = 2
    self.env = MultiplayerSnakeEnv()
    self.env.reset()

  def test_second_snake_blocks_first_snake(self):
    # snake_1 is the top two rows of the first column facing up
    snake_1 = Snake(body=[(0,0), (1,0)])
    snake_1.direction = Action.up
    # snake_2 is the top two rows of the second column facing up
    snake_2 = Snake(body=[(0,1), (1,1)])
    snake_2.direction = Action.up

    self.env.snakes = [snake_1, snake_2]
    self.env.fruit = (0,2)

    # make both snakes go up and die.
    _, rewards, dones, _ = self.env.step([0, 0])
    self.assertEqual(rewards, [SnakeEnv.HIT_WALL, SnakeEnv.HIT_WALL])
    self.assertEqual(dones, [True, True])
    self.assertEqual(self.env.snakes[0].body, [(0,0)])
    self.assertEqual(self.env.snakes[1].body, [(0,1)])
    self.assertEqual(self.env.fruit, (0,2))
    # snake_1 should die first since order matters.
    self.assertFalse(self.env.has_won(0))
    self.assertTrue(self.env.has_won(1))


if __name__ == '__main__':
  unittest.main()