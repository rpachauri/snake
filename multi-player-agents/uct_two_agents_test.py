import unittest
import gym
import snake
import numpy as np
import uct

from snake.envs.snake_env import Action
from snake.envs.snake_env import SnakeEnv
from snake.envs.multiplayer_snake_env import MultiplayerSnakeEnv
from snake.envs.multiplayer_snake_env import Snake

class TestUCTTwoAgents(unittest.TestCase):

  def setUp(self):
    SnakeEnv.M = 4
    SnakeEnv.N = 4
    MultiplayerSnakeEnv.num_agents = 2
    self.env = MultiplayerSnakeEnv()
    self.env.reset()

  def test_second_snake_blocks_first_snake_best_actions_force_up(self):
    # snake_1 is the top two rows of the first column facing up
    snake_1 = Snake(body=[(0,0), (1,0)])
    snake_1.direction = Action.up
    # snake_2 is the top two rows of the second column facing up
    snake_2 = Snake(body=[(0,1), (1,1)])
    snake_2.direction = Action.up

    self.env.snakes = [snake_1, snake_2]
    self.env.fruit = (0,2)

    agent = uct.UCT(MultiplayerSnakeEnv.num_agents)
    # force both snakes to only move upwards
    agent.root.action_priors = np.array([
      [1, 0, 0, 0],
      [1, 0, 0, 0],
    ])
    actions = agent.root.best_actions(agent.root_num_visits)
    # all actions for snake_1 should result in a loss.
    self.assertIsNone(np.testing.assert_array_equal(
      actions,
      np.array([0, 0], dtype=np.float32)
    ))

    _, rewards, dones, _ = self.env.step(actions)
    self.assertEqual(rewards, [SnakeEnv.HIT_WALL, SnakeEnv.HIT_WALL])
    self.assertEqual(dones, [True, True])

  def test_second_snake_blocks_first_snake_single_rollout_force_up(self):
    # snake_1 is the top two rows of the first column facing up
    snake_1 = Snake(body=[(0,0), (1,0)])
    snake_1.direction = Action.up
    # snake_2 is the top two rows of the second column facing up
    snake_2 = Snake(body=[(0,1), (1,1)])
    snake_2.direction = Action.up

    self.env.snakes = [snake_1, snake_2]
    self.env.fruit = (0,2)

    agent = uct.UCT(MultiplayerSnakeEnv.num_agents)
    # force both snakes to only move upwards
    agent.root.action_priors = np.array([
      [1, 0, 0, 0],
      [1, 0, 0, 0],
    ])
    agent._perform_rollouts(num_rollouts=1, env=self.env)

    # both agents should have visited Action.up once.
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
      ], dtype=np.float32)
    ))

    # going up for snake_1 should result in a loss.
    # going up for snake_2 should result in a win.
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_total_values,
      np.array([
        [uct.UCTNode.LOSING_VALUE, 0, 0, 0],
        [uct.UCTNode.WINNING_VALUE, 0, 0, 0],
      ], dtype=np.float32)
    ))

  # DOES NOT WORK
  def test_second_snake_blocks_first_snake(self):
    # snake_1 is the top two rows of the first column facing up
    snake_1 = Snake(body=[(0,0), (1,0)])
    snake_1.direction = Action.up
    # snake_2 is the top two rows of the second column facing up
    snake_2 = Snake(body=[(0,1), (1,1)])
    snake_2.direction = Action.up

    self.env.snakes = [snake_1, snake_2]
    self.env.fruit = (0,2)

    agent = uct.UCT(MultiplayerSnakeEnv.num_agents)
    agent._perform_rollouts(num_rollouts=16, env=self.env)

    # all actions for snake_1 should result in a loss.
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_total_values[0],
      np.ones(4, dtype=np.float32) * uct.UCTNode.LOSING_VALUE
    ))
    # up, down, and left should all result in a win because
    # the snake is the last one to die

    # however, UCT will stop exploring when it finds a WINNING action,
    # and it could be any one of the three actions.
    action_total_value_sum = np.sum(np.delete(agent.root.action_total_values[1], 3))
    self.assertEqual(action_total_value_sum, uct.UCTNode.WINNING_VALUE)


# If you want to run this test, make sure to comment out the bottom of uct.py,
# because that will play a game when we import the file.
if __name__ == '__main__':
  unittest.main()