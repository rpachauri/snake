import unittest
import gym
import snake
import numpy as np
import uct

from snake.envs.snake_env import Action
from snake.envs.snake_env import SnakeEnv
from snake.envs.multiplayer_snake_env import MultiplayerSnakeEnv
from snake.envs.multiplayer_snake_env import Snake

class TestUCTSingleAgent(unittest.TestCase):

  def setUp(self):
    SnakeEnv.M = 4
    SnakeEnv.N = 4
    MultiplayerSnakeEnv.num_agents = 1
    self.env = MultiplayerSnakeEnv()
    self.env.reset()

  def test_facing_up_at_bottom_after_single_rollout(self):
    snake = Snake(body=[(0,3), (1,3)])
    self.env.snakes = [snake]
    self.env.fruit = (0,2)

    agent = uct.UCT(MultiplayerSnakeEnv.num_agents)
    agent._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([[1, 0, 0, 0]], dtype=np.float32)
    ))
    
  def test_facing_up_at_top_right(self):
    snake = Snake(body=[(0,3), (1,3)])
    snake.direction = Action.up
    self.env.snakes = [snake]
    self.env.fruit = (0,2)

    agent = uct.UCT(MultiplayerSnakeEnv.num_agents)
    agent._perform_rollouts(num_rollouts=10, env=self.env)
    
    # agent.root.children should only contain left.
    self.assertTrue((1,) in agent.root.children)
    self.assertEqual(len(agent.root.children), 1)

    # root_num_visits for the first agent should be 1 + num_rollouts
    self.assertEqual(agent.root_num_visits[0], 11)
    # up, down, and right should all result in a loss.
    # this means we should visit left 7 out of 10 times.
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits[0],
      np.array([1, 7, 1, 1], dtype=np.float32)
    ))
    # up, down, and right should all result in a LOSING_VALUE
    self.assertIsNone(np.testing.assert_array_equal(
      np.delete(agent.root.action_total_values[0], 1),
      np.ones(3, dtype=np.float32) * uct.UCTNode.LOSING_VALUE
    ))
    
    got_actions = agent._select_actions()
    self.assertEqual(got_actions, [1])

  def test_best_action(self):
    # test UCTNode initialization
    node = uct.UCTNode(num_agents=MultiplayerSnakeEnv.num_agents, num_actions=2)
    expected_action_priors = np.array([[1,1]]) / 2
    self.assertIsNone(np.testing.assert_array_equal(node.action_priors, expected_action_priors))

    # test UCTNode.best_action()
    node.child_total_value = np.array([[1, 10]])
    node.child_num_visits = np.array([[10, 1]])
    got_best_actions = node.best_actions(current_num_visits=11)
    self.assertEqual(got_best_actions, [0])

  def test_perform_rollout_with_no_way_out(self):
    snake = Snake(body=[(3,0), (2,0), (2,1), (3,1), (3,2)])
    self.env.snakes = [snake]
    self.env.fruit = (0,0)

    agent = uct.UCT(MultiplayerSnakeEnv.num_agents)
    agent._perform_rollouts(num_rollouts=4, env=self.env)

    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_total_values,
      np.ones((1,4), dtype=np.float32) * uct.UCTNode.LOSING_VALUE
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.ones((1,4), dtype=np.float32)
    ))

# If you want to run this test, make sure to comment out the bottom of uct.py,
# because that will play a game when we import the file.
if __name__ == '__main__':
  unittest.main()