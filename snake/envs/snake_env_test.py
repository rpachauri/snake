import unittest
import gym
import snake
import numpy as np

class TestSnakeEnv(unittest.TestCase):

  def setUp(self):
    self.env = gym.make('snake-v0')
    snake.envs.snake_env.SnakeEnv.M = 4
    snake.envs.snake_env.SnakeEnv.N = 4

  def test_facing_down_at_bottom_turn_left(self):
    self.env.body = [(3,0), (2,0)]
    self.env.fruit = (3,3)

    _, reward, done, _ = self.env.step(1)
    self.assertEqual(reward, snake.envs.snake_env.SnakeEnv.HIT_WALL)
    self.assertTrue(done)
    self.assertEqual(self.env.body, [(3,0)])
    self.assertEqual(self.env.fruit, (3,3))

  def test_facing_down_at_bottom_go_up(self):
    self.env.body = [(3,0), (2,0)]
    self.env.fruit = (3,3)

    _, reward, done, _ = self.env.step(0)
    self.assertEqual(self.env.body, [(3,0)])
    self.assertEqual(self.env.fruit, (3,3))
    self.assertTrue(done)
    self.assertEqual(reward, snake.envs.snake_env.SnakeEnv.HIT_WALL)

if __name__ == '__main__':
  unittest.main()