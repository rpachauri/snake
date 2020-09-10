import unittest
import gym
import snake
import numpy as np

from snake.envs.snake_env import Action
from snake.envs.snake_env import SnakeEnv
from snake.envs.multiplayer_snake_env import MultiplayerSnakeEnv
from snake.envs.multiplayer_snake_env import Snake


class TestSingleAgentMultiplayerSnakeEnv(unittest.TestCase):

  def setUp(self):
    SnakeEnv.M = 4
    SnakeEnv.N = 4
    MultiplayerSnakeEnv.num_agents = 1
    self.env = MultiplayerSnakeEnv()
    self.env.reset()

  def test_facing_down_at_bottom_turn_left(self):
    snake = Snake(body=[(3,0), (2,0)]) # single snake of length two, facing down the first column, at the bottom
    snake.direction = Action.down
    self.env.snakes = [snake]
    self.env.fruit = (3,3)

    _, rewards, dones, _ = self.env.step([1]) # turn left
    self.assertEqual(rewards, [SnakeEnv.HIT_WALL])
    self.assertEqual(dones, [True])
    self.assertEqual(self.env.snakes[0].body, [(3,0)])
    self.assertEqual(self.env.fruit, (3,3))

  def test_go_up_when_facing_right_at_top_right(self):
    snake = Snake(body=[(0,3), (1,3)])
    snake.direction = Action.right
    self.env.snakes = [snake]
    self.env.fruit = (0,2)

    # Snake is currently facing right and we tell it to move up.
    _, rewards, dones, _ = self.env.step([0])
    self.assertEqual(rewards, [SnakeEnv.HIT_WALL])
    self.assertEqual(dones, [True])
    self.assertEqual(self.env.snakes[0].body, [(0,3)])
    self.assertEqual(self.env.fruit, (0,2))

  def test_hit_wall(self):
    # hit wall
    self.assertTrue(self.env.hit_wall(next_head=(-1, 0)))
    self.assertTrue(self.env.hit_wall(next_head=(0, -1)))
    self.assertTrue(self.env.hit_wall(next_head=(SnakeEnv.M, 0)))
    self.assertTrue(self.env.hit_wall(next_head=(0, SnakeEnv.N)))

    # did not hit wall
    self.assertFalse(self.env.hit_wall(next_head=(0, 0)))
    self.assertFalse(self.env.hit_wall(next_head=(SnakeEnv.M - 1, 0)))
    self.assertFalse(self.env.hit_wall(next_head=(0, SnakeEnv.N - 1)))
    self.assertFalse(self.env.hit_wall(next_head=(SnakeEnv.M - 1, SnakeEnv.N - 1)))

  def test_hit_body(self):
    # there is no way out. Left and Down hit a wall, but Right hits the body.
    snake = Snake(body=[(3,0), (2,0), (2,1), (3,1), (3,2)])
    self.env.snakes = [snake]
    self.env.fruit = (0,0)

    # turn right
    _, rewards, dones, _ = self.env.step([3])
    self.assertEqual(rewards, [SnakeEnv.HIT_BODY])
    self.assertEqual(dones, [True])
    self.assertEqual(self.env.snakes[0].body, [(3,0), (2,0), (2,1), (3,1)])
    self.assertEqual(self.env.fruit, (0,0))

  def test_has_won(self):
    # since there only exists a single snake, the snake should already have "won"
    self.assertTrue(self.env.has_won(0))


if __name__ == '__main__':
  unittest.main()